import AVFoundation
import CoreVideo
import Metal
import VideoToolbox

/// Bounded IOSurface buffers feed the hardware encoder while the next frame renders.
/// Pixel data never crosses into a Swift array, PNG, or a second encoding pass.
@MainActor
final class DirectVideoWriter {
    let writer: AVAssetWriter
    let input: AVAssetWriterInput
    let adaptor: AVAssetWriterInputPixelBufferAdaptor
    let queue: MTLCommandQueue
    let cache: CVMetalTextureCache
    let pool: CVPixelBufferPool
    let width: Int
    let height: Int
    let fps: Int
    private(set) var frames = 0
    private(set) var backpressureSeconds = 0.0
    private(set) var copyGPUSeconds = 0.0
    private(set) var submissionSeconds = 0.0
    private(set) var drainSeconds = 0.0
    private var pendingBuffer: CVPixelBuffer?

    init(device: MTLDevice, url: URL, width: Int, height: Int, fps: Int = 30,
         bitrate: Int = 60_000_000) throws {
        self.width = width
        self.height = height
        self.fps = fps
        queue = device.makeCommandQueue()!
        var textureCache: CVMetalTextureCache?
        try Self.check(CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache), "texture cache")
        cache = textureCache!
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        var pixelPool: CVPixelBufferPool?
        try Self.check(CVPixelBufferPoolCreate(nil,
            [kCVPixelBufferPoolMinimumBufferCountKey as String: 3] as CFDictionary,
            attributes as CFDictionary, &pixelPool), "pixel buffer pool")
        pool = pixelPool!
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        var transfer = AVVideoTransferFunction_ITU_R_709_2
        if #available(macOS 15, *) { transfer = AVVideoTransferFunction_IEC_sRGB }
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoEncoderSpecificationKey: [
                kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder as String: true,
            ],
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoExpectedSourceFrameRateKey: fps,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                AVVideoAllowFrameReorderingKey: false,
                AVVideoMaxKeyFrameIntervalKey: fps * 2,
            ],
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: transfer,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
        ]
        input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
                                                       sourcePixelBufferAttributes: attributes)
        guard writer.canAdd(input) else { throw failure("Hardware H.264 settings are unsupported") }
        writer.add(input)
        guard writer.startWriting() else { throw failure("Could not start video writer") }
        writer.startSession(atSourceTime: .zero)
    }

    private static func check(_ status: OSStatus, _ operation: String) throws {
        guard status == noErr else {
            throw NSError(domain: "HeroDirectVideo", code: Int(status),
                          userInfo: [NSLocalizedDescriptionKey: "Failed \(operation): \(status)"])
        }
    }

    private func failure(_ detail: String) -> NSError {
        NSError(domain: "HeroDirectVideo", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "\(detail): \(String(describing: writer.error))"])
    }

    /// Reserve before drawing so backpressure never drops or duplicates a frame.
    func prepare() async throws {
        precondition(pendingBuffer == nil)
        let start = Date()
        while true {
            guard writer.status == .writing else { throw failure("Video writer stopped") }
            if input.isReadyForMoreMediaData {
                var buffer: CVPixelBuffer?
                let status = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(nil, pool,
                    [kCVPixelBufferPoolAllocationThresholdKey as String: 6] as CFDictionary, &buffer)
                if status == noErr {
                    pendingBuffer = buffer!
                    break
                }
                if status != kCVReturnWouldExceedAllocationThreshold {
                    try Self.check(status, "pixel buffer allocation")
                }
            }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        backpressureSeconds += Date().timeIntervalSince(start)
    }

    /// The renderer owns its callback texture; finish the small GPU blit before
    /// that texture is recycled. Hardware compression proceeds asynchronously.
    func append(_ texture: MTLTexture) throws {
        let start = Date()
        guard let buffer = pendingBuffer else { throw failure("No reserved output buffer") }
        defer { pendingBuffer = nil }
        precondition(texture.width == width && texture.height == height)
        precondition(texture.pixelFormat == .bgra8Unorm_srgb || texture.pixelFormat == .bgra8Unorm)
        CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey,
                              kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey,
                              kCVImageBufferTransferFunction_sRGB, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey,
                              kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        var cvTexture: CVMetalTexture?
        try Self.check(CVMetalTextureCacheCreateTextureFromImage(nil, cache, buffer, nil,
            texture.pixelFormat, width, height, 0, &cvTexture), "IOSurface Metal texture")
        let destination = CVMetalTextureGetTexture(cvTexture!)!
        let command = queue.makeCommandBuffer()!
        let blit = command.makeBlitCommandEncoder()!
        blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: .init(x: 0, y: 0, z: 0),
                  sourceSize: .init(width: width, height: height, depth: 1),
                  to: destination, destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: .init(x: 0, y: 0, z: 0))
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw failure("Video GPU copy failed: \(String(describing: command.error))") }
        copyGPUSeconds += command.gpuEndTime - command.gpuStartTime
        guard adaptor.append(buffer, withPresentationTime: CMTime(value: Int64(frames), timescale: Int32(fps)))
        else { throw failure("Video encoder rejected frame \(frames)") }
        frames += 1
        submissionSeconds += Date().timeIntervalSince(start)
    }

    func finish() async throws {
        precondition(pendingBuffer == nil && frames > 0)
        let start = Date()
        writer.endSession(atSourceTime: CMTime(value: Int64(frames), timescale: Int32(fps)))
        input.markAsFinished()
        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }
        drainSeconds = Date().timeIntervalSince(start)
        guard writer.status == .completed else { throw failure("Video finalization failed") }
    }

    var metrics: [String: Any] {
        ["encoder": "VideoToolbox hardware H.264 (required)", "cpu_pixel_readbacks": 0,
         "png_frames": 0, "frames": frames, "pool_buffer_limit": 6,
         "backpressure_s": backpressureSeconds, "copy_gpu_s": copyGPUSeconds,
         "submission_s": submissionSeconds, "drain_s": drainSeconds]
    }
}

import AppKit
import Foundation
import GPUSimRenderer
import ImageIO
import MetalKit
import simd

struct MaterialDescription: Decodable {
    let name: String
    let color: [Float]
    let roughness: Float
    let transmission: Float
    let ior: Float
    let clearcoat: Float
}

struct SceneDescription: Decodable {
    let builtinGround: Bool?
    let bodyCount: Int
    let sampleCount: Int
    let recordingFps: Int
    let substeps: Int
    let iterations: Int
    let indexCount: Int
    let vertexCount: Int
    let vertexStride: Int
    let materials: [MaterialDescription]
    let traceSha256: String
}

struct CameraKeyframe: Decodable {
    let time: Double
    let position: [Float]
    let target: [Float]
    let fov: Float
}

struct CameraDescription: Decodable {
    let name: String
    let position: [Float]
    let target: [Float]
    let fov: Float
    let endPosition: [Float]?
    let endTarget: [Float]?
    let endFov: Float?
    let moveStart: Double?
    let moveEnd: Double?
    let keyframes: [CameraKeyframe]?

    func pose(at time: Double) -> (position: SIMD3<Float>, target: SIMD3<Float>, fov: Float) {
        if let keyframes, let first = keyframes.first, let last = keyframes.last {
            if time <= first.time { return (vector(first.position), vector(first.target), first.fov) }
            for (start, end) in zip(keyframes, keyframes.dropFirst()) {
                precondition(end.time > start.time)
                if time <= end.time {
                    let u = Float((time - start.time) / (end.time - start.time))
                    let blend = u * u * u * (10 + u * (-15 + 6 * u))
                    return (mix(vector(start.position), vector(end.position), t: SIMD3(repeating: blend)),
                            mix(vector(start.target), vector(end.target), t: SIMD3(repeating: blend)),
                            start.fov + (end.fov - start.fov) * blend)
                }
            }
            return (vector(last.position), vector(last.target), last.fov)
        }
        guard let endPosition, let endTarget, let moveStart, let moveEnd else {
            return (vector(position), vector(target), fov)
        }
        precondition(moveEnd > moveStart)
        let u = Float(min(1, max(0, (time - moveStart) / (moveEnd - moveStart))))
        let blend = u * u * u * (10 + u * (-15 + 6 * u))
        return (mix(vector(position), vector(endPosition), t: SIMD3(repeating: blend)),
                mix(vector(target), vector(endTarget), t: SIMD3(repeating: blend)),
                fov + ((endFov ?? fov) - fov) * blend)
    }
}

func vector(_ values: [Float]) -> SIMD3<Float> {
    precondition(values.count == 3)
    return SIMD3(values[0], values[1], values[2])
}

// This adapter supplies recorded poses to the renderer. It has no solver.
final class ReplayScene: GPUSimRenderableScene {
    let renderDevice: MTLDevice
    let description: SceneDescription
    let renderBodyCount: Int
    let renderRigidInstanceCount = 0
    let rendererStateIsValid = true
    let renderCameraHint = GPUSimRenderCameraHint()
    let softRenderSurface: GPUSimSoftRenderSurface? = nil
    let skinnedRenderSurface: GPUSimSkinnedRenderSurface? = nil
    let convexDebugRenderSurface: GPUSimConvexDebugRenderSurface? = nil
    let rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface?
    let renderContentBounds: GPUSimContentBounds? = .init(center: SIMD3(0.15, 0, 0.4), radius: 1.5)
    var renderStateRevision: UInt64? = 0
    let positions: MTLBuffer
    let rotations: MTLBuffer
    let positionData: Data
    let rotationData: Data

    init(device: MTLDevice, directory: URL) throws {
        renderDevice = device
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        description = try decoder.decode(SceneDescription.self, from: Data(contentsOf: directory.appendingPathComponent("scene.json")))
        renderBodyCount = description.bodyCount
        precondition(description.substeps == 8 && description.iterations == 10)
        precondition(description.vertexStride == MemoryLayout<GPUSimRigidMeshRenderVertex>.stride)
        func loadBuffer(_ name: String) throws -> MTLBuffer {
            let data = try Data(contentsOf: directory.appendingPathComponent(name))
            return data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared)! }
        }
        let vertices = try loadBuffer("vertices.bin")
        let indices = try loadBuffer("indices.bin")
        precondition(vertices.length == description.vertexCount * description.vertexStride)
        precondition(indices.length == description.indexCount * 4)
        positions = device.makeBuffer(length: renderBodyCount * 16, options: .storageModeShared)!
        rotations = device.makeBuffer(length: renderBodyCount * 16, options: .storageModeShared)!
        positionData = try Data(contentsOf: directory.appendingPathComponent("positions.bin"))
        rotationData = try Data(contentsOf: directory.appendingPathComponent("rotations.bin"))
        precondition(positionData.count == description.sampleCount * renderBodyCount * 16)
        precondition(rotationData.count == positionData.count)
        rigidMeshRenderSurface = .init(vertices: vertices, indices: indices, indexCount: description.indexCount,
                                       positions: positions, rotations: rotations)
        setSample(0)
    }

    func setSample(_ sample: Int) {
        precondition((0..<description.sampleCount).contains(sample))
        let bytes = renderBodyCount * 16
        positionData.withUnsafeBytes { source in
            _ = memcpy(positions.contents(), source.baseAddress!.advanced(by: sample * bytes), bytes)
        }
        rotationData.withUnsafeBytes { source in
            _ = memcpy(rotations.contents(), source.baseAddress!.advanced(by: sample * bytes), bytes)
        }
        renderStateRevision = UInt64(sample)
    }

    func encodeRenderInstances(_ commandBuffer: MTLCommandBuffer, instances: MTLBuffer,
                               colorMode: GPUSimRenderColorMode, appearanceOverrides: MTLBuffer?) throws {}
}

@MainActor
final class FrameCapture {
    var pixels = [UInt8]()
    var continuation: CheckedContinuation<Void, Never>?
    let renderer: GPUSimRenderer
    let view: MTKView
    let width: Int
    let height: Int

    init(renderer: GPUSimRenderer, device: MTLDevice, width: Int, height: Int) {
        self.renderer = renderer
        self.width = width
        self.height = height
        view = MTKView(frame: NSRect(x: 0, y: 0, width: width, height: height), device: device)
        renderer.configure(view)
        view.isPaused = true
        view.autoResizeDrawable = false
        view.drawableSize = CGSize(width: width, height: height)
        renderer.frameCompletionHandler = { [weak self] texture, _ in
            guard let self else { return }
            self.pixels = [UInt8](repeating: 0, count: width * height * 4)
            self.pixels.withUnsafeMutableBytes {
                texture.getBytes($0.baseAddress!, bytesPerRow: width * 4,
                                 from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
            }
            self.continuation?.resume()
            self.continuation = nil
        }
    }

    func draw() async {
        await withCheckedContinuation { pending in
            continuation = pending
            view.draw()
            if let error = renderer.runtimeFailure { fatalError(error) }
        }
        precondition(renderer.activeLightingMode == .qualityBeta, "HQ unexpectedly fell back to Fast")
    }

    func save(_ url: URL, outputWidth: Int, outputHeight: Int) throws {
        var rgba = pixels
        for i in stride(from: 0, to: rgba.count, by: 4) { rgba.swapAt(i, i + 2) }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let resolved: CGImage
        if outputWidth == width && outputHeight == height { resolved = image }
        else {
            let context = CGContext(data: nil, width: outputWidth, height: outputHeight,
                bitsPerComponent: 8, bytesPerRow: outputWidth * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: outputWidth, height: outputHeight))
            resolved = context.makeImage()!
        }
        let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, resolved, nil)
        precondition(CGImageDestinationFinalize(destination), "PNG write failed")
    }
}

@main
struct KitchenHQReplay {
    @MainActor
    static func main() async throws {
        func option(_ key: String, default fallback: String) -> String {
            guard let i = CommandLine.arguments.firstIndex(of: key) else { return fallback }
            return CommandLine.arguments[i + 1]
        }
        let input = URL(fileURLWithPath: option("--data", default: "data"))
        let output = URL(fileURLWithPath: option("--output", default: "previews"))
        let cameraURL = URL(fileURLWithPath: option("--cameras", default: "cameras.json"))
        let cameras = try JSONDecoder().decode([CameraDescription].self, from: Data(contentsOf: cameraURL))
        let width = Int(option("--width", default: "1280"))!
        let height = Int(option("--height", default: "720"))!
        let supersample = Double(option("--supersample", default: "1"))!
        precondition(supersample.isFinite && supersample >= 1 && supersample <= 2)
        let renderWidth = Int((Double(width) * supersample).rounded())
        let renderHeight = Int((Double(height) * supersample).rounded())
        let interfaces = Int(option("--interfaces", default: "24"))!
        let accumulation = Int(option("--accumulation", default: "32"))!
        let times = option("--times", default: "12,20").split(separator: ",").map { Double($0)! }
        let videoCamera = option("--video-camera", default: "")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        guard let device = MTLCreateSystemDefaultDevice(), GPUSimRenderer.supportsHQ(device: device)
        else { fatalError("HQ requires supported Metal ray tracing and MetalFX") }
        let scene = try ReplayScene(device: device, directory: input)
        let materials = scene.description.materials.map { source in
            var material = GPUSimSurfaceMaterial()
            material.baseColor = vector(source.color)
            material.roughness = source.roughness
            material.previewOptics.transmission = source.transmission
            material.previewOptics.indexOfRefraction = source.ior
            material.previewOptics.clearcoat = source.clearcoat
            return material
        }
        let library = try GPUSimMaterialLibrary(device: device, materials: materials)
        let renderer = try GPUSimRenderer(device: device, scene: scene, materials: library)
        renderer.automaticallyFramesScene = false
        renderer.sceneLengthScale = 0.1
        renderer.nearClipDistance = Float(option("--near-clip", default: "0.1"))! / renderer.sceneLengthScale
        renderer.farClipDistance = Float(option("--far-clip", default: "30"))! / renderer.sceneLengthScale
        renderer.options = .qualityBeta
        renderer.options.reconstructionScale = Float(option("--scale", default: "1"))!
        let quality = option("--quality", default: "balanced")
        renderer.options.rayTracingQuality = quality == "high" ? .high : .balanced
        renderer.options.rayTracingDenoising = !CommandLine.arguments.contains("--no-denoise")
        renderer.options.rayTracingQuality.transmissionInterfaces = interfaces
        renderer.options.rayTracingQuality.secondaryAreaLightSamples = 1
        renderer.options.showsGroundPlane = scene.description.builtinGround ?? false
        renderer.options.sunAngularRadius = 0.06
        renderer.options.sunDirection = normalize(SIMD3<Float>(0.4, 0.5, -1))
        renderer.options.sunIntensity = 0.6
        renderer.options.ambientExposure = 0.2
        renderer.options.displayExposure = 0
        let capture = FrameCapture(renderer: renderer, device: device, width: renderWidth, height: renderHeight)
        var results = [[String: Any]]()
        for camera in cameras where videoCamera.isEmpty || camera.name == videoCamera {
            renderer.verticalFieldOfView = camera.fov
            renderer.setCamera(position: vector(camera.position), target: vector(camera.target), resetTemporalHistory: true)
            let samples = videoCamera.isEmpty
                ? times.map { Int(($0 * Double(scene.description.recordingFps)).rounded()) }
                : Array(stride(from: 0, to: scene.description.sampleCount, by: 2))
            if !videoCamera.isEmpty {
                scene.setSample(0)
                for _ in 0..<accumulation { await capture.draw() }
            }
            let started = Date()
            var gpuMS = 0.0
            for (index, sample) in samples.enumerated() {
                scene.setSample(sample)
                let cameraPose = camera.pose(at: Double(sample) / Double(scene.description.recordingFps))
                renderer.verticalFieldOfView = cameraPose.fov
                renderer.setCamera(position: cameraPose.position, target: cameraPose.target, resetTemporalHistory: false)
                if videoCamera.isEmpty { renderer.resetTemporalHistory() }
                for _ in 0..<(videoCamera.isEmpty ? accumulation : 1) { await capture.draw() }
                gpuMS += renderer.lastFrameGPUMilliseconds
                let name = videoCamera.isEmpty
                    ? String(format: "%@-%02ds.png", camera.name, sample / scene.description.recordingFps)
                    : String(format: "frame_%04d.png", index + 1)
                try capture.save(output.appendingPathComponent(name), outputWidth: width, outputHeight: height)
                if videoCamera.isEmpty || index % 60 == 0 {
                    print("Saved \(name), HQ GPU \(String(format: "%.1f", renderer.lastFrameGPUMilliseconds)) ms")
                    fflush(stdout)
                }
            }
            results.append(["camera": camera.name, "position": camera.position, "target": camera.target,
                            "fov": camera.fov, "end_position": camera.endPosition ?? camera.position,
                            "end_target": camera.endTarget ?? camera.target, "end_fov": camera.endFov ?? camera.fov,
                            "move_start_s": camera.moveStart ?? 0, "move_end_s": camera.moveEnd ?? 0,
                            "keyframes": camera.keyframes?.map { ["time_s": $0.time, "position": $0.position,
                                "target": $0.target, "fov": $0.fov] as [String: Any] } ?? [],
                            "images": samples.count, "wall_s": Date().timeIntervalSince(started),
                            "mean_final_frame_gpu_ms": gpuMS / Double(samples.count)])
        }
        let report: [String: Any] = ["renderer": "avbd-metal GPUSimRenderer HQ", "device": device.name,
            "substeps": scene.description.substeps, "iterations": scene.description.iterations,
            "trace_sha256": scene.description.traceSha256, "simulation_steps_executed": 0,
            "width": width, "height": height, "quality": quality,
            "reconstruction_scale": renderer.options.reconstructionScale,
            "denoising": renderer.options.rayTracingDenoising,
            "builtin_ground": renderer.options.showsGroundPlane,
            "near_clip_m": renderer.nearClipDistance * renderer.sceneLengthScale,
            "far_clip_m": renderer.farClipDistance * renderer.sceneLengthScale,
            "render_width": renderWidth, "render_height": renderHeight, "supersample": supersample,
            "transmission_interfaces": interfaces, "accumulation_frames": accumulation, "results": results]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("render-report.json"))
    }
}

import Foundation
import Metal

/// Exercise the actual capture/encoder path at native resolution, including
/// moving content and saturated patches for channel/order decoding checks.
@MainActor
func benchmarkEncoding(output: URL, width: Int, height: Int, seconds: Int) async throws {
    precondition(seconds > 0 && width > 0 && height > 0)
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let device = MTLCreateSystemDefaultDevice()!
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void pattern(texture2d<float, access::write> image [[texture(0)]],
                        constant uint &frame [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
        if (p.x >= image.get_width() || p.y >= image.get_height()) return;
        float2 uv = float2(p) / float2(image.get_width(), image.get_height());
        float moving = float(frame) / 30.0;
        bool checker = ((uint(p.x + frame * 3) / 32 + p.y / 32) & 1) != 0;
        float3 color = float3(uv.x, uv.y, 0.25 + 0.2 * sin(uv.x * 40.0 + moving));
        color = mix(color, float3(checker ? 0.8 : 0.15), 0.3);
        if (uv.x < 0.2 && uv.y < 0.2) color = float3(1, 0, 0);
        if (uv.x > 0.8 && uv.y < 0.2) color = float3(0, 0, 1);
        if (uv.x < 0.2 && uv.y > 0.8) color = float3(0, 1, 0);
        if (uv.x > 0.8 && uv.y > 0.8) color = float3(1);
        if (uv.y > 0.20 && uv.y < 0.24) {
            uint bit = min(10u, uint(uv.x * 11));
            color = float3(((frame >> bit) & 1) != 0 ? 0.9 : 0.1);
        }
        image.write(float4(color, 1), p);
    }
    """
    let library = try await device.makeLibrary(source: source, options: nil)
    let pipeline = try await device.makeComputePipelineState(function: library.makeFunction(name: "pattern")!)
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
        width: width, height: height, mipmapped: false)
    descriptor.storageMode = .private
    descriptor.usage = [.shaderRead, .shaderWrite]
    let texture = device.makeTexture(descriptor: descriptor)!
    let queue = device.makeCommandQueue()!
    let writer = try DirectVideoWriter(device: device, url: output.appendingPathComponent("encoding-benchmark.mp4"),
                                       width: width, height: height)
    let start = Date()
    var patternGPUSeconds = 0.0
    for i in 0..<(seconds * 30) {
        try await writer.prepare()
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        var frame = UInt32(i)
        encoder.setBytes(&frame, length: MemoryLayout<UInt32>.stride, index: 0)
        encoder.dispatchThreads(.init(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: .init(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        await withCheckedContinuation { continuation in
            command.addCompletedHandler { _ in continuation.resume() }
            command.commit()
        }
        precondition(command.status == .completed, String(describing: command.error))
        patternGPUSeconds += command.gpuEndTime - command.gpuStartTime
        try writer.append(texture)
    }
    try await writer.finish()
    let wall = Date().timeIntervalSince(start)
    var report = writer.metrics
    report["width"] = width
    report["height"] = height
    report["fps"] = 30
    report["video_duration_s"] = seconds
    report["wall_s"] = wall
    report["export_fps"] = Double(writer.frames) / wall
    report["pattern_gpu_s"] = patternGPUSeconds
    report["renderer_included"] = false
    let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: output.appendingPathComponent("encoding-benchmark.json"))
    print(String(data: json, encoding: .utf8)!)
}

import AppKit
import Foundation
import GPUSimRenderer
import ImageIO
import MetalKit
import simd

struct MaterialDescription: Decodable {
    let color: [Float]
    let roughness: Float
    let metallic: Float
    let texture: String?
}

struct MeshDescription: Decodable {
    let world: Int
    let firstIndex: Int
    let indexCount: Int
}

struct WorldDescription: Decodable {
    let id: String
    let displayOffset: [Float]
}

struct SceneDescription: Decodable {
    let builtinGround: Bool
    let bodyCount: Int
    let sampleCount: Int
    let recordingFps: Int
    let substeps: Int
    let iterations: Int
    let indexCount: Int
    let vertexCount: Int
    let vertexStride: Int
    let materials: [MaterialDescription]
    let meshes: [MeshDescription]
    let worlds: [WorldDescription]
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
    let worlds: [String]
    let startTime: Double
    let duration: Double
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
    let renderContentBounds: GPUSimContentBounds?
    var renderStateRevision: UInt64? = 0
    let positions: MTLBuffer
    let rotations: MTLBuffer
    let positionData: Data
    let rotationData: Data

    init(device: MTLDevice, directory: URL, selectedWorlds: [String]) throws {
        renderDevice = device
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let description = try decoder.decode(SceneDescription.self, from: Data(contentsOf: directory.appendingPathComponent("scene.json")))
        self.description = description
        renderBodyCount = description.bodyCount
        precondition(description.substeps > 0 && description.iterations > 0)
        let selected = description.worlds.indices.filter { selectedWorlds.isEmpty || selectedWorlds.contains(description.worlds[$0].id) }
        precondition(!selected.isEmpty)
        let center = selected.map { vector(description.worlds[$0].displayOffset) }.reduce(SIMD3<Float>.zero, +) / Float(selected.count)
        let radius = selected.map { length(vector(description.worlds[$0].displayOffset) - center) + 1.8 }.max()!
        renderContentBounds = .init(center: center, radius: radius)
        precondition(description.vertexStride == MemoryLayout<GPUSimRigidMeshRenderVertex>.stride)
        let vertexData = try Data(contentsOf: directory.appendingPathComponent("vertices.bin"))
        precondition(vertexData.count == description.vertexCount * description.vertexStride)
        let indexData = try Data(contentsOf: directory.appendingPathComponent("indices.bin"))
        var chosenIndices = [UInt32]()
        indexData.withUnsafeBytes { raw in
            let source = raw.bindMemory(to: UInt32.self)
            for mesh in description.meshes where selected.contains(mesh.world) {
                chosenIndices.append(contentsOf: source[mesh.firstIndex..<(mesh.firstIndex + mesh.indexCount)])
            }
        }
        // Keep exactly the selected geometry, but avoid transforming vertices
        // from nineteen invisible worlds on every close-up render pass.
        let vertices: MTLBuffer
        if selectedWorlds.isEmpty {
            vertices = vertexData.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
            }
        } else {
            let used = Set(chosenIndices).sorted()
            var remap = [UInt32](repeating: .max, count: description.vertexCount)
            vertices = device.makeBuffer(length: used.count * description.vertexStride, options: .storageModeShared)!
            vertexData.withUnsafeBytes { raw in
                for (destination, source) in used.enumerated() {
                    remap[Int(source)] = UInt32(destination)
                    memcpy(vertices.contents().advanced(by: destination * description.vertexStride),
                           raw.baseAddress!.advanced(by: Int(source) * description.vertexStride), description.vertexStride)
                }
            }
            for i in chosenIndices.indices {
                let original = chosenIndices[i]
                chosenIndices[i] = remap[Int(original)]
                precondition(used[Int(chosenIndices[i])] == original)
            }
        }
        let indices = chosenIndices.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        precondition(indices.length == chosenIndices.count * 4)
        positions = device.makeBuffer(length: renderBodyCount * 16, options: .storageModeShared)!
        rotations = device.makeBuffer(length: renderBodyCount * 16, options: .storageModeShared)!
        positionData = try Data(contentsOf: directory.appendingPathComponent("positions.bin"), options: .mappedIfSafe)
        rotationData = try Data(contentsOf: directory.appendingPathComponent("rotations.bin"), options: .mappedIfSafe)
        precondition(positionData.count == description.sampleCount * renderBodyCount * 16)
        precondition(rotationData.count == positionData.count)
        rigidMeshRenderSurface = .init(vertices: vertices, indices: indices, indexCount: chosenIndices.count,
                                       positions: positions, rotations: rotations)
        setSample(0)
    }

    func setSample(_ sample: Int) { setTime(Double(sample) / Double(description.recordingFps)) }

    func setTime(_ time: Double) {
        let sample = min(Double(description.sampleCount - 1), max(0, time * Double(description.recordingFps)))
        let lo = Int(sample), hi = min(lo + 1, description.sampleCount - 1)
        let blend = Float(sample - Double(lo))
        let count = renderBodyCount
        let p = positions.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        let q = rotations.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        positionData.withUnsafeBytes { raw in
            let source = raw.bindMemory(to: SIMD4<Float>.self)
            for i in 0..<count { p[i] = mix(source[lo * count + i], source[hi * count + i], t: SIMD4(repeating: blend)) }
        }
        rotationData.withUnsafeBytes { raw in
            let source = raw.bindMemory(to: SIMD4<Float>.self)
            for i in 0..<count {
                let a = simd_normalize(simd_quatf(vector: source[lo * count + i]))
                var b = simd_normalize(simd_quatf(vector: source[hi * count + i]))
                if dot(a.vector, b.vector) < 0 { b = simd_quatf(vector: -b.vector) }
                q[i] = simd_slerp(a, b, blend).vector
            }
        }
        renderStateRevision = (renderStateRevision ?? 0) &+ 1
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
    var textureConsumer: ((MTLTexture) throws -> Void)?
    var captureError: Error?

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
            do {
                if let consume = self.textureConsumer {
                    try consume(texture)
                } else {
                    self.pixels = [UInt8](repeating: 0, count: width * height * 4)
                    self.pixels.withUnsafeMutableBytes {
                        texture.getBytes($0.baseAddress!, bytesPerRow: width * 4,
                                         from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
                    }
                }
            } catch {
                self.captureError = error
            }
            self.continuation?.resume()
            self.continuation = nil
        }
    }

    func draw(captureOutput: Bool = true,
              textureConsumer: ((MTLTexture) throws -> Void)? = nil) async throws {
        self.textureConsumer = textureConsumer
        captureError = nil
        if captureOutput {
            await withCheckedContinuation { pending in
                continuation = pending
                view.draw()
            }
        } else {
            // ReplayScene uses live pose buffers and requires synchronous
            // retirement. Intermediate accumulation passes need no readback.
            precondition(renderer.scene!.renderSceneRequiresFrameRetirement)
            let callback = renderer.frameCompletionHandler
            renderer.frameCompletionHandler = nil
            view.draw()
            renderer.frameCompletionHandler = callback
        }
        if let error = captureError { throw error }
        if let error = renderer.runtimeFailure { fatalError(error) }
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
struct HeroHQReplay {
    @MainActor
    static func main() async throws {
        func option(_ key: String, default fallback: String) -> String {
            guard let i = CommandLine.arguments.firstIndex(of: key) else { return fallback }
            return CommandLine.arguments[i + 1]
        }
        let input = URL(fileURLWithPath: option("--data", default: "data"))
        let output = URL(fileURLWithPath: option("--output", default: "renders"))
        if CommandLine.arguments.contains("--benchmark-encoding") {
            try await benchmarkEncoding(output: output,
                width: Int(option("--width", default: "2560"))!,
                height: Int(option("--height", default: "1440"))!,
                seconds: Int(option("--benchmark-seconds", default: "60"))!)
            return
        }
        let cameraURL = URL(fileURLWithPath: option("--cameras", default: "shots.json"))
        let cameras = try JSONDecoder().decode([CameraDescription].self, from: Data(contentsOf: cameraURL))
        let width = Int(option("--width", default: "2560"))!
        let height = Int(option("--height", default: "1440"))!
        let directVideo = CommandLine.arguments.contains("--direct-video")
            || (CommandLine.arguments.contains("--video") && !CommandLine.arguments.contains("--png-frames"))
        let accumulation = Int(option("--accumulation", default: directVideo ? "1" : "12"))!
        let samplesPerFrame = Int(option("--samples-per-frame", default: directVideo ? "1" : "2"))!
        precondition(accumulation > 0 && samplesPerFrame > 0)
        let firstFrame = Int(option("--first-frame", default: "0"))!
        let frameCount = Int(option("--frame-count", default: "0"))!
        let resetEveryFrame = CommandLine.arguments.contains("--reset-history-per-frame")
        let requested = option("--shot", default: "")
        let video = CommandLine.arguments.contains("--video") || directVideo
        let quality = option("--quality", default: directVideo ? "realtime" : "high")
        precondition(["high", "balanced", "realtime"].contains(quality), "Unknown ray tracing quality")
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let description = try decoder.decode(SceneDescription.self, from: Data(contentsOf: input.appendingPathComponent("scene.json")))
        guard let device = MTLCreateSystemDefaultDevice(), GPUSimRenderer.supportsHQ(device: device)
        else { fatalError("HQ requires supported Metal ray tracing") }
        var textureCache = [String: MTLTexture]()
        let materials = try description.materials.map { source in
            var material = GPUSimSurfaceMaterial()
            func linear(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
            material.baseColor = SIMD3(linear(source.color[0]), linear(source.color[1]), linear(source.color[2]))
            material.roughness = source.roughness
            material.metallic = source.metallic
            if let name = source.texture {
                if textureCache[name] == nil {
                    textureCache[name] = try GPUSimMaterialLibrary.loadTexture(device: device, url: input.appendingPathComponent(name), sRGB: true)
                }
                material.baseColorTexture = textureCache[name]
            }
            return material
        }
        let library = try GPUSimMaterialLibrary(device: device, materials: materials)
        let startup = Date()
        let renderer = try GPUSimRenderer(device: device, materials: library)
        renderer.automaticallyFramesScene = false
        renderer.sceneLengthScale = 0.1
        let requestedNearClip = CommandLine.arguments.contains("--near-clip")
            ? Float(option("--near-clip", default: "0.5"))! : nil
        if let requestedNearClip { precondition(requestedNearClip.isFinite && requestedNearClip > 0 && requestedNearClip < 450) }
        renderer.farClipDistance = 450
        renderer.options = .qualityBeta
        renderer.options.reconstructionScale = 1
        switch quality {
        case "high": renderer.options.rayTracingQuality = .high
        case "balanced": renderer.options.rayTracingQuality = .balanced
        default: renderer.options.rayTracingQuality = .realtime
        }
        renderer.options.rayTracingQuality.secondaryAreaLightSamples = 1
        if directVideo && quality == "realtime" {
            renderer.options.rayTracingQuality.areaLightSampling = .powerWeighted
            // Bound indirect work too: zero adaptively expands to four rays.
            // Motion-aware history and denoising retain the temporal estimate.
            renderer.options.rayTracingQuality.diffuseSamples = 1
        }
        if CommandLine.arguments.contains("--diffuse-samples") {
            let diffuseSamples = Int(option("--diffuse-samples", default: "4"))!
            precondition(diffuseSamples > 0)
            renderer.options.rayTracingQuality.diffuseSamples = diffuseSamples
        }
        if CommandLine.arguments.contains("--power-weighted-area-lights") {
            renderer.options.rayTracingQuality.areaLightSampling = .powerWeighted
        }
        renderer.options.rayTracingDenoising = !CommandLine.arguments.contains("--no-denoise")
        renderer.options.showsGroundPlane = true
        renderer.options.sunAngularRadius = 0.07
        renderer.options.sunDirection = normalize(SIMD3<Float>(0.4, 0.5, -1))
        renderer.options.sunIntensity = 0.35
        renderer.options.ambientExposure = 0.15
        renderer.options.displayExposure = 0
        let lighting = option("--lighting", default: "soft-studio")
        func panel(_ position: SIMD3<Float>, _ target: SIMD3<Float>, _ size: SIMD2<Float>,
                   _ radiance: SIMD3<Float>) throws -> GPUSimAreaLight {
            try GPUSimAreaLight(position: position, normal: normalize(target - position),
                                up: SIMD3(0, 1, 0), size: size, radiance: radiance)
        }
        switch lighting {
        case "soft-studio": break
        case "warm-window":
            renderer.options.sunIntensity = 0.85
            renderer.options.sunAngularRadius = 0.035
            renderer.options.sunDirection = normalize(SIMD3<Float>(-0.65, 0.30, -0.65))
            renderer.options.ambientExposure = -0.65
            renderer.options.displayExposure = -0.15
            renderer.options.areaLights = [try panel(SIMD3(-10, -8, 15), .zero, SIMD2(18, 16), SIMD3(3.5, 2.6, 1.8))]
        case "cool-lab":
            renderer.options.sunIntensity = 0.12
            renderer.options.ambientExposure = -0.7
            renderer.options.areaLights = [
                try panel(SIMD3(0, -4, 16), .zero, SIMD2(32, 20), SIMD3(2.1, 2.5, 3.2)),
                try panel(SIMD3(13, 2, 10), .zero, SIMD2(12, 16), SIMD3(1.3, 1.6, 2.0))]
        case "warm-gallery":
            renderer.options.sunIntensity = 0.45
            renderer.options.sunAngularRadius = 0.09
            renderer.options.sunDirection = normalize(SIMD3<Float>(0.75, -0.4, -0.8))
            renderer.options.ambientExposure = -0.5
            renderer.options.areaLights = [
                try panel(SIMD3(-7, -9, 14), .zero, SIMD2(22, 18), SIMD3(2.8, 2.5, 2.1)),
                try panel(SIMD3(10, 6, 12), .zero, SIMD2(15, 16), SIMD3(1.0, 1.3, 1.8))]
        default: preconditionFailure("Unknown lighting preset")
        }
        let capture = FrameCapture(renderer: renderer, device: device, width: width, height: height)
        let startupSeconds = Date().timeIntervalSince(startup)
        let selectedCameras = cameras.filter { requested.isEmpty || $0.name == requested }
        func sceneKey(_ camera: CameraDescription) -> String {
            camera.worlds.sorted().joined(separator: "\u{0}")
        }
        let sceneUseCounts = Dictionary(grouping: selectedCameras, by: sceneKey).mapValues(\.count)
        var sceneCache: [String: ReplayScene] = [:]
        for camera in selectedCameras {
            let setupStarted = Date()
            let folder = output.appendingPathComponent(camera.name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let key = sceneKey(camera)
            let scene: ReplayScene
            if let cached = sceneCache[key] {
                scene = cached
            } else {
                scene = try ReplayScene(device: device, directory: input, selectedWorlds: camera.worlds)
                if sceneUseCounts[key]! > 1 { sceneCache[key] = scene }
            }
            // Buffer identity already distinguishes different geometry. A
            // camera cut must reset history without rebuilding cached AS assets.
            try renderer.setScene(scene, resetCamera: false)
            let center = scene.renderContentBounds!.center
            let large = camera.worlds.isEmpty
            let spread: Float = large ? 2.2 : 1
            if lighting == "soft-studio" { renderer.options.areaLights = [
                try GPUSimAreaLight(position: center + SIMD3(-3, -4, 6) * spread, normal: SIMD3(3, 4, -6), size: SIMD2(4, 3) * spread, radiance: SIMD3(5, 4.7, 4.3)),
                try GPUSimAreaLight(position: center + SIMD3(3, 1, 4) * spread, normal: SIMD3(-3, -1, -4), size: SIMD2(3, 3) * spread, radiance: SIMD3(2.3, 2.6, 3.0))
            ]
            }
            let videoWriter = directVideo ? try DirectVideoWriter(device: device,
                url: folder.appendingPathComponent("video.mp4"), width: width, height: height) : nil
            let setupSeconds = Date().timeIntervalSince(setupStarted)
            let totalFrames = Int((camera.duration * 30).rounded())
            let endFrame = frameCount > 0 ? min(totalFrames, firstFrame + frameCount) : totalFrames
            precondition(firstFrame >= 0 && firstFrame < endFrame)
            let frameTimes = video ? (firstFrame..<endFrame).map { Double($0) / 30 } : (frameCount == 1 ? [0.0] : [0.15, 0.6, 0.95].map { $0 * camera.duration })
            // The renderer scales clipping distances by sceneLengthScale.
            // A fixed 5 cm near plane loses depth precision on distant thin
            // shells. Use 2% of the closest focus distance for this shot, with
            // the original close-up minimum. Keep it constant during motion
            // so changing the projection does not invalidate temporal history.
            let closestFocus = (0...totalFrames).map { frame -> Float in
                let pose = camera.pose(at: Double(frame) / 30)
                return length(pose.position - pose.target)
            }.min()!
            renderer.nearClipDistance = requestedNearClip ?? max(0.5, min(20, closestFocus * 0.2))
            let started = Date()
            var finalPassGPUMS = 0.0, totalGPUMS = 0.0, outputSeconds = 0.0
            var drawCount = 0
            for (index, t) in frameTimes.enumerated() {
                let cameraPose = camera.pose(at: t)
                scene.setTime(camera.startTime + t)
                renderer.verticalFieldOfView = cameraPose.fov
                renderer.setCamera(position: cameraPose.position, target: cameraPose.target, resetTemporalHistory: resetEveryFrame || !video || index == 0)
                try await videoWriter?.prepare()
                let passes = (!video || index == 0) ? accumulation : samplesPerFrame
                for pass in 0..<passes {
                    let consumer: ((MTLTexture) throws -> Void)? = videoWriter.map { writer in
                        { texture in try writer.append(texture) }
                    }
                    try await capture.draw(captureOutput: pass == passes - 1, textureConsumer: consumer)
                    totalGPUMS += renderer.lastFrameGPUMilliseconds
                    drawCount += 1
                }
                finalPassGPUMS += renderer.lastFrameGPUMilliseconds
                if !directVideo {
                    let outputStarted = Date()
                    let name = String(format: "frame_%04d.png", video ? firstFrame + index : index)
                    try capture.save(folder.appendingPathComponent(name), outputWidth: width, outputHeight: height)
                    outputSeconds += Date().timeIntervalSince(outputStarted)
                }
                if !video || index % 30 == 0 {
                    let elapsed = Date().timeIntervalSince(started)
                    print("\(camera.name) \(index + 1)/\(frameTimes.count), last pass GPU \(String(format: "%.1f", renderer.lastFrameGPUMilliseconds)) ms; output \(String(format: "%.1f", elapsed * 1000 / Double(index + 1))) ms/frame")
                    fflush(stdout)
                }
            }
            try await videoWriter?.finish()
            var report: [String: Any] = [
                "renderer": "avbd-metal GPUSimRenderer HQ", "device": device.name, "lighting": lighting,
                "shot": camera.name, "worlds": camera.worlds, "trace_sha256": description.traceSha256,
                "simulation_steps_executed": 0, "source_start_s": camera.startTime,
                "duration_s": camera.duration, "output_fps": 30, "playback_speed": 1,
                "pose_interpolation": "linear translation and quaternion SLERP from 50 Hz trace",
                "width": width, "height": height, "quality": quality,
                "reconstruction_scale": 1, "denoising": renderer.options.rayTracingDenoising,
                "near_clip_world": renderer.nearClipDistance * renderer.sceneLengthScale,
                "far_clip_world": renderer.farClipDistance * renderer.sceneLengthScale,
                "frames": frameTimes.count, "wall_s": Date().timeIntervalSince(started),
                "mean_final_frame_gpu_ms": finalPassGPUMS / Double(frameTimes.count), "samples_per_frame": samplesPerFrame,
                "gpu_passes": drawCount, "gpu_total_ms": totalGPUMS,
                "mean_output_frame_gpu_ms": totalGPUMS / Double(frameTimes.count),
                "png_output_s": outputSeconds, "renderer_startup_s": startupSeconds,
                "shot_setup_s": setupSeconds, "direct_video": directVideo,
                "area_light_sampling": renderer.options.rayTracingQuality.areaLightSampling == .powerWeighted ? "power_weighted" : "all_lights",
                "diffuse_samples": renderer.options.rayTracingQuality.diffuseSamples,
                "first_frame": firstFrame, "reset_history_per_frame": resetEveryFrame
            ]
            if let videoWriter {
                report["video_file"] = "video.mp4"
                report["encoder_metrics"] = videoWriter.metrics
            }
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: folder.appendingPathComponent("render-report.json"))
        }
    }
}

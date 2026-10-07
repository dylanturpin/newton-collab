// swift-tools-version: 5.10
import Foundation
import PackageDescription

let rendererPath = ProcessInfo.processInfo.environment["AVBD_METAL_REPO"] ?? "../../../../avbd-metal"
let package = Package(
    name: "KitchenHQReplay",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "gpu-sim", path: rendererPath)],
    targets: [
        .executableTarget(name: "KitchenHQReplay", dependencies: [
            .product(name: "GPUSimRenderer", package: "gpu-sim"),
        ]),
    ],
    swiftLanguageVersions: [.v5]
)

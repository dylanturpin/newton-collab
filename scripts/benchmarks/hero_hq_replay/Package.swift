// swift-tools-version: 5.10
import Foundation
import PackageDescription

let rendererPath = ProcessInfo.processInfo.environment["AVBD_METAL_REPO"] ?? "../../../../avbd-metal"
let package = Package(
    name: "HeroHQReplay",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "gpu-sim", path: rendererPath)],
    targets: [
        .executableTarget(name: "HeroHQReplay", dependencies: [
            .product(name: "GPUSimRenderer", package: "gpu-sim"),
        ], swiftSettings: [.unsafeFlags(["-parse-as-library"])]),
    ],
    swiftLanguageVersions: [.v5]
)

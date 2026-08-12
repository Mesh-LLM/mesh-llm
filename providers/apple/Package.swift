// swift-tools-version: 6.4

import PackageDescription

let package = Package(
  name: "MeshAppleRuntime",
  platforms: [
    .macOS(.v27)
  ],
  products: [
    .library(
      name: "MeshAppleRuntime",
      targets: ["MeshAppleRuntime"]
    ),
    .executable(
      name: "mesh-apple-runtime",
      targets: ["MeshAppleRuntimeCLI"]
    ),
  ],
  targets: [
    .target(
      name: "MeshAppleRuntime"
    ),
    .executableTarget(
      name: "MeshAppleRuntimeCLI",
      dependencies: ["MeshAppleRuntime"]
    ),
    .testTarget(
      name: "MeshAppleRuntimeTests",
      dependencies: ["MeshAppleRuntime"]
    ),
  ]
)

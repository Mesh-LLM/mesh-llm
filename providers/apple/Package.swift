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
  dependencies: [
    .package(
      url: "https://github.com/apple/coreai-models.git",
      revision: "f401272cd3b8574c27cf5071c56409ad772f91fb"
    )
  ],
  targets: [
    .target(
      name: "MeshAppleRuntime",
      dependencies: [
        .product(name: "CoreAILM", package: "coreai-models")
      ]
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

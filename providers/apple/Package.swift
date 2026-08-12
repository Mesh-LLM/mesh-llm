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
      revision: "25a093b9fb05c99d90bd2d4ddbca44d95cbc6af8"
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

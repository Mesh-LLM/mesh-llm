plugins {
    `java-library`
    `maven-publish`
}

group = "ai.meshllm"
version = "0.72.1"

publishing {
    publications {
        create<MavenPublication>("runtimeJar") {
            artifactId = "meshllm-apple-runtime-macos-arm64"
            from(components["java"])
            pom {
                name.set("MeshLLM Apple runtime for macOS arm64")
                description.set("Signed Apple provider sidecar resources for the MeshLLM Kotlin/JVM SDK.")
                url.set("https://github.com/Mesh-LLM/mesh-llm")
            }
        }
    }
}

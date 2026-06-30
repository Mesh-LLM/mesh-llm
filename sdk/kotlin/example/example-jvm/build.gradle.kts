plugins {
    kotlin("jvm") version "2.0.21"
    application
}

kotlin {
    jvmToolchain(21)
}

group = "ai.meshllm.example"
version = "0.72.1"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("net.java.dev.jna:jna:5.14.0")
}

val repoRoot = projectDir.resolve("../../../..").canonicalFile

val generateKotlinBindings by tasks.registering(Exec::class) {
    description = "Generate Kotlin UniFFI bindings from the mesh-llm FFI UDL"
    group = "build"

    workingDir = repoRoot
    commandLine("bash", "sdk/kotlin/scripts/generate-kotlin-bindings.sh")
    inputs.file(repoRoot.resolve("crates/mesh-llm-ffi/src/mesh_ffi.udl"))
    outputs.file(projectDir.resolve("src/main/kotlin/uniffi/mesh_ffi/mesh_ffi.kt"))
    outputs.file(repoRoot.resolve("sdk/kotlin/src/main/kotlin/uniffi/mesh_ffi/mesh_ffi.kt"))
}

// Include parent binding sources directly — avoids triggering the Android NDK native build
sourceSets {
    main {
        kotlin {
            srcDir("../../src/main/kotlin/ai/meshllm")
        }
    }
}

tasks.named("compileKotlin") {
    dependsOn(generateKotlinBindings)
}

application {
    mainClass.set("ai.meshllm.example.ExampleMainKt")
}

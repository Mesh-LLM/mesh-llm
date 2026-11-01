import java.io.File

plugins {
    kotlin("jvm") version "2.0.21"
    application
}

kotlin {
    jvmToolchain(21)
}

group = "ai.meshllm.example"
version = "0.78.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    implementation("net.java.dev.jna:jna:5.14.0")
}

val repoRoot = projectDir.resolve("../../../../..").canonicalFile

val generateKotlinBindings by tasks.registering(Exec::class) {
    description = "Generate Kotlin UniFFI bindings from the prebuilt mesh-llm FFI library"
    group = "build"

    workingDir = repoRoot
    commandLine("bash", "mesh/sdk/kotlin/scripts/generate-kotlin-bindings.sh")
    // Mirror generate-kotlin-bindings.sh: a blank MESH_LLM_FFI_LIBRARY or
    // CARGO_TARGET_DIR behaves as unset, and the default target directory is
    // <repo>/target. Resolved without nullable provider mappings so the
    // configuration never emits DSL warnings.
    inputs.file(
        providers.environmentVariable("MESH_LLM_FFI_LIBRARY").orElse("").zip(
            providers.environmentVariable("CARGO_TARGET_DIR").orElse("")
        ) { ffiLibrary, cargoTarget ->
            ffiLibrary.takeIf(String::isNotBlank)
                ?: File(cargoTarget.takeIf(String::isNotBlank) ?: repoRoot.resolve("target").absolutePath)
                    .resolve("release/libmeshllm_ffi.a").absolutePath
        }
    )
    outputs.file(projectDir.resolve("src/main/kotlin/uniffi/mesh_ffi/mesh_ffi.kt"))
    outputs.file(repoRoot.resolve("mesh/sdk/kotlin/src/main/kotlin/uniffi/mesh_ffi/mesh_ffi.kt"))
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

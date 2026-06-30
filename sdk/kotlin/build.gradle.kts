import java.io.File

plugins {
    kotlin("jvm") version "2.0.21"
    `maven-publish`
}

group = "ai.meshllm"
version = "0.72.1"

val androidArtifactId = "meshllm-android"

repositories {
    mavenCentral()
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("net.java.dev.jna:jna:5.14.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    testImplementation("junit:junit:4.13.2")
    testImplementation("io.mockk:mockk:1.13.8")
}

val repoRoot = rootProject.projectDir.parentFile.parentFile

val generateKotlinBindings by tasks.registering(Exec::class) {
    description = "Generate Kotlin UniFFI bindings from the mesh-llm FFI UDL"
    group = "build"

    workingDir = repoRoot
    commandLine("bash", "sdk/kotlin/scripts/generate-kotlin-bindings.sh")
    inputs.file(repoRoot.resolve("crates/mesh-llm-ffi/src/mesh_ffi.udl"))
    outputs.file(projectDir.resolve("src/main/kotlin/uniffi/mesh_ffi/mesh_ffi.kt"))
    outputs.file(projectDir.resolve("example/example-jvm/src/main/kotlin/uniffi/mesh_ffi/mesh_ffi.kt"))
}

tasks.named("compileKotlin") {
    dependsOn(generateKotlinBindings)
}

tasks.named("compileTestKotlin") {
    dependsOn(generateKotlinBindings)
}

fun resolveAndroidNdkHome(): String {
    val env = System.getenv()
    val direct = listOf("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT")
        .mapNotNull { env[it] }
        .firstOrNull { File(it).isDirectory }
    if (direct != null) {
        return direct
    }

    val sdkRoots = buildList {
        env["ANDROID_SDK_ROOT"]?.let(::add)
        env["ANDROID_HOME"]?.let(::add)
        add("${System.getProperty("user.home")}/Library/Android/sdk")
        add("${System.getProperty("user.home")}/Android/Sdk")
    }

    sdkRoots
        .map(::File)
        .filter(File::isDirectory)
        .forEach { sdkRoot ->
            val ndkBundle = sdkRoot.resolve("ndk-bundle")
            if (ndkBundle.isDirectory) {
                return ndkBundle.absolutePath
            }

            val ndkDir = sdkRoot.resolve("ndk")
            if (ndkDir.isDirectory) {
                val versions = ndkDir.listFiles()
                    ?.filter(File::isDirectory)
                    ?.sortedByDescending(File::getName)
                    .orEmpty()
                if (versions.isNotEmpty()) {
                    return versions.first().absolutePath
                }
            }
        }

    error("Android NDK not found. Set ANDROID_NDK_HOME or ANDROID_SDK_ROOT/ANDROID_HOME.")
}

// Task to build native libraries for all Android ABIs
val buildNativeLibs by tasks.registering {
    description = "Build mesh-llm-ffi shared libraries with embedded serving for all Android ABIs"
    group = "build"

    val androidPlatform = System.getenv("MESH_LLM_ANDROID_PLATFORM") ?: "android-26"
    val buildTargets = listOf(
        Triple("arm64-v8a", "aarch64-linux-android", "libmeshllm_ffi.so"),
        Triple("armeabi-v7a", "armv7-linux-androideabi", "libmeshllm_ffi.so"),
        Triple("x86_64", "x86_64-linux-android", "libmeshllm_ffi.so"),
    )

    doLast {
        val ndkHome = resolveAndroidNdkHome()
        val rustc = System.getenv("RUSTC")
        val baseEnv = mutableMapOf(
            "ANDROID_NDK_HOME" to ndkHome,
            "ANDROID_NDK_ROOT" to ndkHome,
        )
        if (!rustc.isNullOrBlank()) {
            baseEnv["RUSTC"] = rustc
        }

        exec {
            workingDir = repoRoot
            commandLine(
                "bash",
                "scripts/prepare-llama.sh",
                System.getenv("MESH_LLM_LLAMA_PIN_SHA") ?: "pinned"
            )
        }

        buildTargets.forEach { (abi, target, _) ->
            val llamaBuildDir = repoRoot.resolve(".deps/llama-build/build-stage-abi-android-$abi-cpu")
            exec {
                workingDir = repoRoot
                environment(baseEnv)
                environment(
                    mapOf(
                        "LLAMA_STAGE_BACKEND" to "cpu",
                        "LLAMA_STAGE_BUILD_DIR" to llamaBuildDir.absolutePath,
                        "LLAMA_BUILD_DIR" to llamaBuildDir.absolutePath,
                    )
                )
                commandLine(
                    "bash",
                    "scripts/build-llama.sh",
                    "-DCMAKE_TOOLCHAIN_FILE=$ndkHome/build/cmake/android.toolchain.cmake",
                    "-DANDROID_ABI=$abi",
                    "-DANDROID_PLATFORM=$androidPlatform",
                )
            }

            exec {
                workingDir = repoRoot
                environment(baseEnv)
                environment(
                    mapOf(
                        "LLAMA_STAGE_BACKEND" to "cpu",
                        "LLAMA_STAGE_BUILD_DIR" to llamaBuildDir.absolutePath,
                    )
                )
                commandLine(
                    "cargo", "ndk",
                    "-t", abi,
                    "build",
                    "--release",
                    "-p", "mesh-llm-ffi",
                    "--no-default-features",
                    "--features", "host,embedded-runtime"
                )
            }

            copy {
                from(repoRoot.resolve("target/$target/release/libmeshllm_ffi.so"))
                into(projectDir.resolve("src/main/jniLibs/$abi"))
            }
        }
    }

    outputs.files(
        "${projectDir}/src/main/jniLibs/arm64-v8a/libmeshllm_ffi.so",
        "${projectDir}/src/main/jniLibs/armeabi-v7a/libmeshllm_ffi.so",
        "${projectDir}/src/main/jniLibs/x86_64/libmeshllm_ffi.so"
    )
}

// Assemble a distributable AAR artifact (ZIP format) containing:
//   classes.jar              — compiled Kotlin classes
//   jni/<abi>/libmeshllm_ffi.so — native shared libraries
//   consumer-proguard-rules.pro
//   AndroidManifest.xml      — minimal manifest required by AAR spec
val assembleAar by tasks.registering(Zip::class) {
    description = "Assemble AAR artifact with native libs and consumer ProGuard rules"
    group = "build"

    dependsOn(buildNativeLibs)
    dependsOn("jar")

    archiveFileName.set("$androidArtifactId.aar")
    destinationDirectory.set(layout.buildDirectory.dir("outputs/aar"))

    // Compiled Kotlin classes, renamed to the standard AAR entry name
    from(tasks.named<Jar>("jar")) {
        rename { "classes.jar" }
    }

    // Native shared libraries under jni/<abi>/
    from("src/main/jniLibs") {
        into("jni")
    }

    // Consumer ProGuard rules consumed by downstream Android projects
    from("consumer-proguard-rules.pro")

    // Console web assets exposed to Android apps as package assets.
    from("src/main/resources/mesh-llm/console") {
        into("assets/mesh-llm/console")
    }

    // Minimal AndroidManifest required by the AAR format
    from("src/main/AndroidManifest.xml")
}

val sourcesJar by tasks.registering(Jar::class) {
    description = "Assemble Kotlin sources jar for Maven publication"
    group = "build"

    dependsOn(generateKotlinBindings)
    archiveClassifier.set("sources")
    from("src/main/kotlin")
}

publishing {
    publications {
        create<MavenPublication>("aar") {
            groupId = project.group.toString()
            artifactId = androidArtifactId
            version = project.version.toString()

            artifact(assembleAar) {
                extension = "aar"
            }
            artifact(sourcesJar)

            pom {
                name.set("MeshLLM Android SDK")
                description.set("Android/Kotlin bindings for mesh-llm model management, local serving, and mesh inference.")
                url.set("https://github.com/Mesh-LLM/mesh-llm")

                licenses {
                    license {
                        name.set("MIT")
                        url.set("https://github.com/Mesh-LLM/mesh-llm/blob/main/LICENSE")
                    }
                }

                scm {
                    url.set("https://github.com/Mesh-LLM/mesh-llm")
                    connection.set("scm:git:https://github.com/Mesh-LLM/mesh-llm.git")
                    developerConnection.set("scm:git:ssh://git@github.com/Mesh-LLM/mesh-llm.git")
                }

                withXml {
                    val projectNode = asNode()
                    val dependenciesNode = (projectNode.get("dependencies") as? groovy.util.NodeList)
                        ?.firstOrNull() as? groovy.util.Node
                        ?: projectNode.appendNode("dependencies")

                    fun dependencyNodes(): List<groovy.util.Node> =
                        dependenciesNode.children().filterIsInstance<groovy.util.Node>()

                    fun childText(node: groovy.util.Node, name: String): String? =
                        node.get(name).let { children ->
                            (children as? groovy.util.NodeList)
                                ?.firstOrNull()
                                ?.let { it as? groovy.util.Node }
                                ?.text()
                        }

                    fun ensureDependency(group: String, artifactId: String, version: String, scope: String) {
                        val dependencyNode = dependencyNodes().firstOrNull {
                            childText(it, "groupId") == group && childText(it, "artifactId") == artifactId
                        } ?: dependenciesNode.appendNode("dependency").also {
                            it.appendNode("groupId", group)
                            it.appendNode("artifactId", artifactId)
                            it.appendNode("version", version)
                        }

                        val scopeNode = (dependencyNode.get("scope") as? groovy.util.NodeList)
                            ?.firstOrNull() as? groovy.util.Node
                        if (scopeNode == null) {
                            dependencyNode.appendNode("scope", scope)
                        } else {
                            scopeNode.setValue(scope)
                        }
                    }

                    ensureDependency(
                        group = "org.jetbrains.kotlinx",
                        artifactId = "kotlinx-coroutines-core",
                        version = "1.7.3",
                        scope = "compile"
                    )
                    ensureDependency(
                        group = "net.java.dev.jna",
                        artifactId = "jna",
                        version = "5.14.0",
                        scope = "runtime"
                    )
                }
            }
        }
    }

    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/Mesh-LLM/mesh-llm")
            credentials {
                username = providers.environmentVariable("GITHUB_ACTOR")
                    .orElse(providers.gradleProperty("gpr.user"))
                    .orNull
                password = providers.environmentVariable("GITHUB_TOKEN")
                    .orElse(providers.gradleProperty("gpr.key"))
                    .orNull
            }
        }
    }
}

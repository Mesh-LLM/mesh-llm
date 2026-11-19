// Resolve the Kotlin Gradle plugin from Maven Central first. The plugin portal
// serves it by redirecting to Central and intermittently reports the plugin's
// BOM as missing, which fails the build before any code compiles.
pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

rootProject.name = "example-jvm"

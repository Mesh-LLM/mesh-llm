-keep class ai.meshllm.** { *; }
-keep class uniffi.mesh_ffi.** { *; }
-keepclassmembers class * {
    native <methods>;
}

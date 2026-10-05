# Container build cache benchmarks

## llama.cpp compiler cache

Measured 2026-08-02 on `mesh1.patio51.com` (Linux ARM64) with Debian Bookworm,
GCC 12.2, sccache 0.16.0, and BuildKit 0.31.2. Each timed sample rebuilt the
same patched llama.cpp CPU/static source tree in a clean build directory. The
compiler cache was the only persistent native-build state.

| Sample | sccache result | Compile step |
|---|---:|---:|
| Control 1 | disabled | 122.1s |
| Control 2 | disabled | 121.2s |
| Cache population | 0/273 hits, 273 misses, 0 errors | 125.8s |
| Warm cache | 273/273 hits, 0 misses, 0 errors | 3.55s |

The warm compiler-cache path reduced the forced compile step by 97.1% versus
the faster control. The one-time population cost was 3.8% above that control.
This supports keeping the mount while full llama.cpp compilation remains in
the container path.

The cache ID separates Debian Bookworm, GCC 12, the CPU backend, and target
architecture. Package downloads use a separate architecture-specific mount.

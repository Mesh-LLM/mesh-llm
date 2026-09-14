$ErrorActionPreference = "Stop"

$sccacheVersion = if ($env:MESH_LLM_SCCACHE_VERSION) {
    $env:MESH_LLM_SCCACHE_VERSION
} else {
    "0.16.0"
}

& rustup component add llvm-tools-preview
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the Rust LLVM linker tools."
}

$installedVersion = ""
if (Get-Command sccache -ErrorAction SilentlyContinue) {
    $installedVersion = ((& sccache --version) -split '\s+')[-1]
}
if ($installedVersion -ne $sccacheVersion) {
    $hadRustcWrapper = Test-Path Env:RUSTC_WRAPPER
    $previousRustcWrapper = $env:RUSTC_WRAPPER
    try {
        $env:RUSTC_WRAPPER = ""
        & cargo install sccache --version $sccacheVersion --locked --force
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install sccache $sccacheVersion."
        }
    } finally {
        if ($hadRustcWrapper) {
            $env:RUSTC_WRAPPER = $previousRustcWrapper
        } else {
            Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
        }
    }
}

& sccache --version
& (Join-Path $PSScriptRoot "cargo-linker.cmd") --mesh-probe
if ($LASTEXITCODE -ne 0) {
    throw "The Windows lld linker probe failed."
}

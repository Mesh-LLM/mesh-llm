param(
    [string]$Backend = "",
    [string]$CudaArch = "",
    [string]$RocmArch = "",
    [string]$BuildProfile = "",
    [switch]$DynamicHost,
    [switch]$HostOnly,
    [switch]$SkippyOnly
)

# Stable workspace entrypoint; implementation belongs to Mesh.
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "../mesh/scripts/build-windows.ps1") @PSBoundParameters
exit $LASTEXITCODE

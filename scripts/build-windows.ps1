param(
    [string]$BuildProfile = "",
    [string]$TargetArch = ""
)

# Stable workspace entrypoint; implementation belongs to Mesh.
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "../mesh/scripts/build-windows.ps1") @PSBoundParameters
exit $LASTEXITCODE

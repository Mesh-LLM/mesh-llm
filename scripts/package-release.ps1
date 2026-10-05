param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$OutputDir = "dist",
    [string]$Flavor = ""
)

# Stable workspace entrypoint; implementation belongs to Mesh.
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "../mesh/scripts/package-release.ps1") @PSBoundParameters
exit $LASTEXITCODE

[CmdletBinding()]
param(
  [ValidateSet('cpu','cuda','rocm','vulkan')][string]$Backend = 'cpu',
  [ValidateSet('debug','dev','release')][string]$BuildProfile = 'release',
  [string]$CudaArch = '',
  [string]$RocmArch = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
# This wrapper intentionally prevents a headless/no-UI artifact.
Remove-Item Env:MESH_LLM_SKIP_UI -ErrorAction SilentlyContinue
$build = Join-Path $PSScriptRoot 'build-windows.ps1'
& $build -Backend $Backend -BuildProfile $BuildProfile -CudaArch $CudaArch -RocmArch $RocmArch
if ($LASTEXITCODE -ne 0) { throw "Windows webchat build failed: $LASTEXITCODE" }
$binary = if ($BuildProfile -eq 'release') { Join-Path $root 'target\release\mesh-llm.exe' } else { Join-Path $root 'target\debug\mesh-llm.exe' }
& (Join-Path $PSScriptRoot 'verify-webchat-build.ps1') -Binary $binary -ExpectedVersion '0.71.0'

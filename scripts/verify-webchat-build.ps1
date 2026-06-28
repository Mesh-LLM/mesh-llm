[CmdletBinding()]
param(
  [string]$Binary = (Join-Path $PSScriptRoot '..\target\release\mesh-llm.exe'),
  [string]$UiRoot = (Join-Path $PSScriptRoot '..\crates\mesh-llm-ui'),
  [string]$ExpectedVersion = '0.71.0'
)
$ErrorActionPreference = 'Stop'
$connection = Join-Path $UiRoot 'src\features\chat\api\mesh-connection.ts'
$toolLoop = Join-Path $UiRoot 'src\features\chat\api\flushnet-tool-loop.ts'
$package = Join-Path $UiRoot 'package.json'
$dist = Join-Path $UiRoot 'dist'
foreach ($path in @($connection,$toolLoop,$package,(Join-Path $dist 'index.html'),$Binary)) {
  if (-not (Test-Path $path)) { throw "Required webchat build artifact is missing: $path" }
}
$connectionText = Get-Content -LiteralPath $connection -Raw
$toolText = Get-Content -LiteralPath $toolLoop -Raw
if ($connectionText -notmatch 'runFlushnetToolChat' -or $connectionText -notmatch 'extractFlushnetAccessCode') {
  throw 'Authenticated Flushnet tool-chat route is absent from mesh-connection.ts.'
}
if ($connectionText -notmatch 'createResponsesStream|buildResponsesInput') {
  throw 'Normal Mesh-LLM webchat route is absent from mesh-connection.ts.'
}
if ($toolText -notmatch 'FLUSHNET_TOOL_MODE_PROMPT' -or $toolText -notmatch 'gatewayExecute') {
  throw 'Flushnet canonical tool loop is incomplete.'
}
$version = (& $Binary --version 2>&1 | Select-Object -First 1).ToString().Trim()
if ($version -ne "mesh-llm $ExpectedVersion") { throw "Unexpected binary version: $version" }
$distFiles = @(Get-ChildItem -LiteralPath $dist -Recurse -File)
if ($distFiles.Count -lt 2) { throw 'Embedded UI dist is unexpectedly empty.' }
[pscustomobject]@{
  binary = $Binary
  version = $version
  ui_dist_files = $distFiles.Count
  normal_mesh_webchat = $true
  flushnet_tool_webchat = $true
} | ConvertTo-Json -Compress

param(
    [string]$Repository = "",
    [string]$InstallDirectory = "$env:LOCALAPPDATA\Mesh-LLM\bin"
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw "GitHub CLI (gh) is required" }
$repoArgs = if ($Repository) { @("--repo", $Repository) } else { @() }
$runId = gh run list @repoArgs --workflow fast-reusable-binaries.yml --status success --limit 1 --json databaseId --jq '.[0].databaseId'
if (-not $runId) { throw "No successful fast binary workflow run found" }

$temp = Join-Path ([System.IO.Path]::GetTempPath()) "mesh-llm-fast-$([guid]::NewGuid())"
try {
    New-Item -ItemType Directory -Force $temp | Out-Null
    gh run download $runId @repoArgs --name mesh-llm-win3-x64-cpu --dir $temp
    $expected = ((Get-Content (Join-Path $temp "mesh-llm-win3-x64-cpu.zip.sha256") -Raw) -split '\s+')[0].ToLowerInvariant()
    $actual = (Get-FileHash (Join-Path $temp "mesh-llm-win3-x64-cpu.zip") -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "SHA-256 mismatch: expected $expected, got $actual" }
    Expand-Archive (Join-Path $temp "mesh-llm-win3-x64-cpu.zip") -DestinationPath $temp -Force
    New-Item -ItemType Directory -Force $InstallDirectory | Out-Null
    Copy-Item (Join-Path $temp "mesh-llm-win3-x64-cpu\mesh-llm.exe") (Join-Path $InstallDirectory "mesh-llm.exe") -Force
    & (Join-Path $InstallDirectory "mesh-llm.exe") --version
} finally {
    Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
}

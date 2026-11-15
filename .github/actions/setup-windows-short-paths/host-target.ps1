$ErrorActionPreference = "Stop"

function Initialize-HostCargoTarget {
    param([string]$Workspace, [string]$TargetRoot)

    if (-not [string]::IsNullOrEmpty($env:CARGO_TARGET_DIR)) {
        throw "Refusing to replace an existing Cargo target override"
    }
    $workspaceTarget = Join-Path $Workspace "target"
    foreach ($path in @($workspaceTarget, $TargetRoot)) {
        if (Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue) {
            throw "Refusing existing Windows host target entry: $path"
        }
    }
    New-Item -ItemType Directory -Path $TargetRoot -ErrorAction Stop | Out-Null
    New-Item -ItemType Junction -Path $workspaceTarget -Target $TargetRoot -ErrorAction Stop | Out-Null
    $junction = Get-Item -LiteralPath $workspaceTarget -Force
    if ($junction.LinkType -ne "Junction" -or
        [IO.Path]::GetFullPath(@($junction.Target)[0]) -ne [IO.Path]::GetFullPath($TargetRoot)) {
        throw "Windows host target junction did not resolve to the short target"
    }
    "CARGO_TARGET_DIR=$TargetRoot" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    "cargo_target=$TargetRoot" | Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
    Write-Host "Windows host Cargo target: $TargetRoot"
}

$workspace = [IO.Path]::GetFullPath($env:GITHUB_WORKSPACE)
$targetRoot = Join-Path ([IO.Path]::GetPathRoot($workspace)) "b"
Initialize-HostCargoTarget -Workspace $workspace -TargetRoot $targetRoot

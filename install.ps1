param(
    [switch]$PreRelease,
    [string]$InstallDir = $env:MESH_LLM_INSTALL_DIR,
    [string]$Flavor,
    [switch]$NoPathUpdate,
    [switch]$NoSetup,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Repo = if ($env:MESH_LLM_INSTALL_REPO) { $env:MESH_LLM_INSTALL_REPO } else { "Mesh-LLM/mesh-llm" }
$HostArchive = "mesh-llm-x86_64-pc-windows-msvc.zip"
$ReleaseUrlBase = $env:MESH_LLM_INSTALL_URL_BASE

function Test-Truthy {
    param([string]$Value)

    if (-not $Value) {
        return $false
    }

    return @("1", "true", "yes", "on") -contains $Value.Trim().ToLowerInvariant()
}

if (Test-Truthy $env:MESH_LLM_INSTALL_PRERELEASE) {
    $PreRelease = $true
}

$RequireChecksum = Test-Truthy $env:MESH_LLM_REQUIRE_CHECKSUM

if (-not $Flavor -and $env:MESH_LLM_INSTALL_FLAVOR) {
    $Flavor = $env:MESH_LLM_INSTALL_FLAVOR
}

if (-not $InstallDir) {
    $localAppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME "AppData\Local" }
    $InstallDir = Join-Path $localAppData "mesh-llm\bin"
}

function Show-Usage {
    @"
Usage: install.ps1 [-PreRelease] [-InstallDir <DIR>] [-Flavor <FLAVOR>] [-NoPathUpdate] [-NoSetup]

Options:
  -PreRelease             Install the latest published GitHub prerelease instead of the latest stable release.
  -InstallDir <DIR>       Install directory. Defaults to %LOCALAPPDATA%\mesh-llm\bin.
  -Flavor <FLAVOR>        Legacy compatibility flag. The installer always installs the Windows x64 host binary and ``mesh-llm.exe setup`` now chooses the runtime.
  -NoPathUpdate           Do not add the install directory to the user Path.
  -NoSetup                Do not run ``mesh-llm.exe setup``; print the exact command instead.
  -Help                   Show this help text.

Environment overrides:
  MESH_LLM_INSTALL_DIR
  MESH_LLM_INSTALL_FLAVOR
  MESH_LLM_INSTALL_PRERELEASE=1
  MESH_LLM_INSTALL_REPO=Mesh-LLM/mesh-llm
  MESH_LLM_REQUIRE_CHECKSUM=1
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

function Require-WindowsX64 {
    if (Test-Truthy $env:MESH_LLM_INSTALL_TEST_ALLOW_NONWINDOWS) {
        return
    }

    if (-not $IsWindows -and $PSVersionTable.PSEdition -eq "Core") {
        throw "install.ps1 only supports native Windows. Use install.sh on macOS or Linux."
    }

    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    if ($arch -ne "X64") {
        throw "unsupported Windows architecture: $arch. Published Windows release bundles target x86_64."
    }
}

function Get-GitHubHeaders {
    $headers = @{
        "Accept" = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
        "User-Agent" = "mesh-llm-installer"
    }
    if ($env:GITHUB_TOKEN) {
        $headers["Authorization"] = "Bearer $env:GITHUB_TOKEN"
    } elseif ($env:GH_TOKEN) {
        $headers["Authorization"] = "Bearer $env:GH_TOKEN"
    }
    return $headers
}

function Get-LatestPrereleaseTag {
    $apiUrl = "https://api.github.com/repos/$Repo/releases?per_page=20"
    $releases = Invoke-RestMethod -Uri $apiUrl -Headers (Get-GitHubHeaders)
    foreach ($release in $releases) {
        if ($release.prerelease -and -not $release.draft) {
            return $release.tag_name
        }
    }
    throw "could not find a published prerelease for $Repo"
}

function Join-UrlPath {
    param(
        [string]$Base,
        [string]$Child
    )

    if ($Base.EndsWith("/")) {
        return "$Base$Child"
    }
    return "$Base/$Child"
}

function Get-ReleaseUrl {
    param([string]$Asset)

    if ($ReleaseUrlBase) {
        return Join-UrlPath -Base $ReleaseUrlBase -Child $Asset
    }

    if ($PreRelease) {
        $tag = Get-LatestPrereleaseTag
        return "https://github.com/$Repo/releases/download/$tag/$Asset"
    }

    return "https://github.com/$Repo/releases/latest/download/$Asset"
}

function Get-ChecksumUrl {
    param([string]$Url)
    return "$Url.sha256"
}

function Read-ExpectedSha256 {
    param([string]$Path)

    $content = Get-Content -Path $Path -Raw
    $match = [regex]::Match($content, "[A-Fa-f0-9]{64}")
    if (-not $match.Success) {
        throw "checksum sidecar did not contain a SHA-256 digest: $Path"
    }
    return $match.Value.ToLowerInvariant()
}

function Test-MissingChecksumResponse {
    param([object]$ErrorRecord)

    $response = $ErrorRecord.Exception.Response
    if (-not $response) {
        return $ErrorRecord.Exception -is [System.Net.WebException]
    }

    $statusCode = [int]$response.StatusCode
    return $statusCode -eq 404 -or $statusCode -eq 410
}

function Assert-DownloadedFileChecksum {
    param(
        [string]$Path,
        [string]$Url,
        [bool]$RequireSidecar = $RequireChecksum
    )

    $checksumPath = "$Path.sha256"
    $checksumUrl = Get-ChecksumUrl $Url
    try {
        Invoke-WebRequest -Uri $checksumUrl -OutFile $checksumPath
    } catch {
        if (Test-Path $checksumPath) {
            Remove-Item $checksumPath -Force
        }
        if (Test-MissingChecksumResponse $_) {
            if ($RequireSidecar) {
                throw "checksum sidecar is required but missing: $checksumUrl"
            }
            Write-Warning "Checksum sidecar not found; continuing without archive verification: $checksumUrl"
            return
        }
        throw "could not download checksum sidecar: $checksumUrl"
    }

    $expected = Read-ExpectedSha256 $checksumPath
    $actual = (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "checksum mismatch for $(Split-Path -Leaf $Path): expected $expected, got $actual"
    }
    Write-Host "Verified checksum: $(Split-Path -Leaf $Path)"
}

function Write-FlavorCompatibilityWarning {
    if (-not $Flavor) {
        return
    }

    $legacyFlavor = $Flavor.Trim().ToLowerInvariant()
    if (-not $legacyFlavor) {
        return
    }

    Write-Warning "Ignoring legacy -Flavor '$legacyFlavor'. The Windows installer now always installs the x64 host binary; run ``mesh-llm.exe setup`` to choose the recommended runtime."
}

function Get-StaleBinaryNames {
    @(
        "mesh-llm.exe",
        "mesh-llm-cpu.exe",
        "mesh-llm-cuda.exe",
        "mesh-llm-cuda-blackwell.exe",
        "mesh-llm-rocm.exe",
        "mesh-llm-vulkan.exe",
        "rpc-server.exe",
        "llama-server.exe",
        "llama-moe-split.exe"
    )
}

function Remove-StaleBinaries {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    foreach ($name in Get-StaleBinaryNames) {
        $path = Join-Path $InstallDir $name
        if (Test-Path $path) {
            Remove-Item $path -Force
        }
    }
}

function Install-MeshBinary {
    param([string]$BundleDir)

    $meshBinarySource = Join-Path $BundleDir "mesh-llm.exe"
    if (-not (Test-Path $meshBinarySource)) {
        throw "release archive did not contain mesh-bundle/mesh-llm.exe"
    }

    Remove-StaleBinaries
    Copy-Item -Path $meshBinarySource -Destination (Join-Path $InstallDir "mesh-llm.exe") -Force
}

function Add-InstallDirToPath {
    if ($NoPathUpdate) {
        return $false
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @()
    if ($userPath) {
        $parts = $userPath -split ";"
    }

    foreach ($part in $parts) {
        if ($part.TrimEnd([char]'\') -ieq $InstallDir.TrimEnd([char]'\')) {
            return $false
        }
    }

    $newPath = if ($userPath) { "$InstallDir;$userPath" } else { $InstallDir }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    $env:Path = "$InstallDir;$env:Path"
    Write-Host "Added $InstallDir to your user Path."
    Write-Host "Open a new PowerShell session before running mesh-llm from PATH."
    return $true
}

function Test-InteractiveSession {
    if ($env:MESH_LLM_INSTALL_INTERACTIVE) {
        return Test-Truthy $env:MESH_LLM_INSTALL_INTERACTIVE
    }

    if (-not [Environment]::UserInteractive) {
        return $false
    }

    return -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected
}

function Format-SetupCommand {
    param([string]$MeshBinary)
    return "& `"$MeshBinary`" setup"
}

function Invoke-SetupOrPrint {
    param([string]$MeshBinary)

    $setupCommand = Format-SetupCommand -MeshBinary $MeshBinary
    if ($NoSetup -or -not (Test-InteractiveSession)) {
        Write-Host "Run this next:"
        Write-Host $setupCommand
        return
    }

    Write-Host "Running: $setupCommand"
    & $MeshBinary setup
    if ($LASTEXITCODE -ne 0) {
        throw "mesh-llm.exe setup exited with code $LASTEXITCODE"
    }
}

Require-WindowsX64
Write-FlavorCompatibilityWarning

$asset = $HostArchive
$url = Get-ReleaseUrl $asset
$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("mesh-llm-install-" + [System.Guid]::NewGuid().ToString("N"))
$archive = Join-Path $tmpRoot $asset

New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null

try {
    Write-Host "Installing Windows x64 host binary"
    if ($PreRelease) {
        Write-Host "Release channel: prerelease"
    } else {
        Write-Host "Release channel: stable"
    }
    Write-Host "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $archive
    Assert-DownloadedFileChecksum -Path $archive -Url $url

    Expand-Archive -Path $archive -DestinationPath $tmpRoot -Force

    $bundleDir = Join-Path $tmpRoot "mesh-bundle"
    if (-not (Test-Path $bundleDir)) {
        throw "release archive did not contain mesh-bundle/"
    }

    Install-MeshBinary -BundleDir $bundleDir
    $pathUpdated = Add-InstallDirToPath

    $meshBinary = Join-Path $InstallDir "mesh-llm.exe"
    Write-Host "Installed $asset to $InstallDir"
    & $meshBinary --version

    if ($NoPathUpdate -and -not $pathUpdated) {
        Write-Host "Install directory was not added to PATH. Use the full command below until you add $InstallDir to PATH."
    }

    Invoke-SetupOrPrint -MeshBinary $meshBinary
} finally {
    if (Test-Path $tmpRoot) {
        Remove-Item $tmpRoot -Recurse -Force
    }
}

param(
    [switch]$PreRelease,
    [string]$InstallDir = $env:MESH_LLM_INSTALL_DIR,
    [ValidateSet("cpu", "cuda", "cuda-blackwell", "rocm", "vulkan")]
    [string]$Flavor = $env:MESH_LLM_INSTALL_FLAVOR,
    [switch]$NoPathUpdate,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Repo = if ($env:MESH_LLM_INSTALL_REPO) { $env:MESH_LLM_INSTALL_REPO } else { "Mesh-LLM/mesh-llm" }

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

if (-not $InstallDir) {
    $localAppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME "AppData\Local" }
    $InstallDir = Join-Path $localAppData "mesh-llm\bin"
}

function Show-Usage {
    @"
Usage: install.ps1 [-PreRelease] [-InstallDir <DIR>] [-Flavor <FLAVOR>] [-NoPathUpdate]

Options:
  -PreRelease             Install the latest published GitHub prerelease instead of the latest stable release.
  -InstallDir <DIR>       Install directory. Defaults to %LOCALAPPDATA%\mesh-llm\bin.
  -Flavor <FLAVOR>        Release bundle flavor: cpu, cuda, cuda-blackwell, rocm, or vulkan.
  -NoPathUpdate           Do not add the install directory to the user Path.
  -Help                   Show this help text.

Environment overrides:
  MESH_LLM_INSTALL_DIR
  MESH_LLM_INSTALL_FLAVOR
  MESH_LLM_INSTALL_PRERELEASE=1
  MESH_LLM_INSTALL_REPO=Mesh-LLM/mesh-llm
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

function Require-WindowsX64 {
    if (-not $IsWindows -and $PSVersionTable.PSEdition -eq "Core") {
        throw "install.ps1 only supports native Windows. Use install.sh on macOS or Linux."
    }

    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    if ($arch -ne "X64") {
        throw "unsupported Windows architecture: $arch. Published Windows release bundles target x86_64."
    }
}

function Test-Command {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Normalize-CudaCapability {
    param([string]$Value)
    return ($Value -replace "[^0-9]", "")
}

function Test-CudaBlackwell {
    if (Test-Command "nvidia-smi") {
        try {
            $caps = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
            foreach ($cap in $caps) {
                $normalized = Normalize-CudaCapability $cap
                if ($normalized -match "^[0-9]+$" -and [int]$normalized -ge 100 -and [int]$normalized -lt 200) {
                    return $true
                }
            }
        } catch {
        }

        try {
            $names = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
            foreach ($name in $names) {
                if (Test-NvidiaModelBlackwell $name) {
                    return $true
                }
            }
        } catch {
        }
    }

    try {
        $controllers = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        foreach ($controller in $controllers) {
            if ($controller.Name -and (Test-NvidiaModelBlackwell $controller.Name)) {
                return $true
            }
        }
    } catch {
    }

    return $false
}

function Test-NvidiaModelBlackwell {
    param([string]$Name)
    $upper = $Name.ToUpperInvariant()
    return $upper -match "BLACKWELL|GB300|B300|GB200|B200|B100|GB10|THOR|RTX 5090|RTX 5080|RTX 5070|RTX 5060|RTX 5050|RTX PRO 6000"
}

function Test-Nvidia {
    if ((Test-Command "nvidia-smi") -or (Test-Command "nvcc")) {
        return $true
    }

    try {
        $controllers = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        foreach ($controller in $controllers) {
            if ($controller.Name -and $controller.Name.ToUpperInvariant().Contains("NVIDIA")) {
                return $true
            }
        }
    } catch {
    }

    return $false
}

function Test-Rocm {
    if ((Test-Command "rocm-smi") -or (Test-Command "rocminfo") -or (Test-Command "hipcc")) {
        return $true
    }

    foreach ($path in @(
        "$env:ProgramFiles\AMD\ROCm",
        "$env:ProgramFiles\AMD\ROCm*\bin\hipcc.exe",
        "$env:ProgramFiles\AMD\ROCm*\bin\rocminfo.exe"
    )) {
        if ($path -and (Test-Path $path)) {
            return $true
        }
    }

    return $false
}

function Test-Vulkan {
    if ((Test-Command "vulkaninfo") -or (Test-Command "glslc")) {
        return $true
    }
    if ($env:VULKAN_SDK -and (Test-Path $env:VULKAN_SDK)) {
        return $true
    }
    return $false
}

function Get-SupportedFlavors {
    return @("cuda-blackwell", "cuda", "rocm", "vulkan", "cpu")
}

function Get-RecommendedFlavor {
    if (Test-CudaBlackwell) {
        return "cuda-blackwell"
    }
    if (Test-Nvidia) {
        return "cuda"
    }
    if (Test-Rocm) {
        return "rocm"
    }
    if (Test-Vulkan) {
        return "vulkan"
    }
    return "cpu"
}

function Get-RecommendationReason {
    param([string]$SelectedFlavor)
    switch ($SelectedFlavor) {
        "cuda-blackwell" { return "Blackwell NVIDIA hardware was detected." }
        "cuda" { return "NVIDIA tooling or devices were detected." }
        "rocm" { return "ROCm/HIP tooling was detected." }
        "vulkan" { return "Vulkan tooling was detected." }
        "cpu" { return "No supported GPU runtime was detected." }
        default { return "Flavor was selected explicitly." }
    }
}

function Choose-Flavor {
    if ($Flavor) {
        if ((Get-SupportedFlavors) -notcontains $Flavor) {
            throw "unsupported Windows flavor '$Flavor'"
        }
        return $Flavor
    }

    $recommended = Get-RecommendedFlavor
    if ([Console]::IsInputRedirected -or [Console]::IsOutputRedirected) {
        return $recommended
    }

    $flavors = Get-SupportedFlavors
    Write-Host "Mesh LLM installer"
    Write-Host "Platform: Windows/x86_64"
    Write-Host "Recommended flavor: $recommended"
    Write-Host "Reason: $(Get-RecommendationReason $recommended)"
    Write-Host ""
    Write-Host "Available flavors:"
    for ($i = 0; $i -lt $flavors.Count; $i++) {
        $label = $flavors[$i]
        if ($label -eq $recommended) {
            $label = "$label (recommended)"
        }
        Write-Host ("  {0}. {1}" -f ($i + 1), $label)
    }
    Write-Host ""

    $reply = Read-Host "Install which flavor? [$recommended]"
    if (-not $reply) {
        return $recommended
    }
    if ($reply -match "^[0-9]+$") {
        $index = [int]$reply - 1
        if ($index -ge 0 -and $index -lt $flavors.Count) {
            return $flavors[$index]
        }
    }
    if ($flavors -contains $reply) {
        return $reply
    }
    throw "unsupported Windows flavor '$reply'"
}

function Get-AssetName {
    param([string]$SelectedFlavor)
    if ($SelectedFlavor -eq "cpu") {
        return "mesh-llm-x86_64-pc-windows-msvc.zip"
    }
    return "mesh-llm-x86_64-pc-windows-msvc-$SelectedFlavor.zip"
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

function Get-ReleaseUrl {
    param([string]$Asset)
    if ($PreRelease) {
        $tag = Get-LatestPrereleaseTag
        return "https://github.com/$Repo/releases/download/$tag/$Asset"
    }
    return "https://github.com/$Repo/releases/latest/download/$Asset"
}

function Get-StaleBinaryNames {
    $names = @(
        "mesh-llm",
        "mesh-llm-cpu",
        "mesh-llm-cuda",
        "mesh-llm-cuda-blackwell",
        "mesh-llm-rocm",
        "mesh-llm-vulkan",
        "rpc-server",
        "llama-server",
        "llama-moe-split",
        "rpc-server-cpu",
        "llama-server-cpu",
        "rpc-server-cuda",
        "llama-server-cuda",
        "rpc-server-rocm",
        "llama-server-rocm",
        "rpc-server-vulkan",
        "llama-server-vulkan"
    )
    foreach ($name in $names) {
        "$name.exe"
    }
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

function Install-Bundle {
    param([string]$BundleDir)
    Remove-StaleBinaries
    Get-ChildItem -Path $BundleDir -Force | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination (Join-Path $InstallDir $_.Name) -Recurse -Force
    }
}

function Install-RecommendedNativeRuntime {
    param([string]$TempDir)
    $meshBinary = Join-Path $InstallDir "mesh-llm.exe"
    if (-not (Test-Path $meshBinary)) {
        return
    }

    $manifestPath = Join-Path $TempDir "native-runtimes.json"
    $manifestUrl = Get-ReleaseUrl "native-runtimes.json"
    try {
        Invoke-WebRequest -Uri $manifestUrl -OutFile $manifestPath
    } catch {
        Write-Host "Native runtime manifest was not available for this release; skipping runtime install."
        return
    }

    & $meshBinary runtime install --manifest $manifestPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Native runtime install did not complete successfully."
        return
    }
    & $meshBinary runtime prune --active-only
}

function Add-InstallDirToPath {
    if ($NoPathUpdate) {
        return
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @()
    if ($userPath) {
        $parts = $userPath -split ";"
    }
    $alreadyPresent = $false
    foreach ($part in $parts) {
        if ($part.TrimEnd([char]'\') -ieq $InstallDir.TrimEnd([char]'\')) {
            $alreadyPresent = $true
            break
        }
    }

    if (-not $alreadyPresent) {
        $newPath = if ($userPath) { "$InstallDir;$userPath" } else { $InstallDir }
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        $env:Path = "$InstallDir;$env:Path"
        Write-Host "Added $InstallDir to your user Path."
        Write-Host "Open a new PowerShell session before running mesh-llm from PATH."
    }
}

Require-WindowsX64

$selectedFlavor = Choose-Flavor
$asset = Get-AssetName $selectedFlavor
$url = Get-ReleaseUrl $asset
$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("mesh-llm-install-" + [System.Guid]::NewGuid().ToString("N"))
$archive = Join-Path $tmpRoot $asset

New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null

try {
    Write-Host "Installing flavor: $selectedFlavor"
    if ($PreRelease) {
        Write-Host "Release channel: prerelease"
    } else {
        Write-Host "Release channel: stable"
    }
    Write-Host "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $archive

    Expand-Archive -Path $archive -DestinationPath $tmpRoot -Force

    $bundleDir = Join-Path $tmpRoot "mesh-bundle"
    if (-not (Test-Path $bundleDir)) {
        throw "release archive did not contain mesh-bundle/"
    }

    Install-Bundle $bundleDir
    Install-RecommendedNativeRuntime $tmpRoot
    Add-InstallDirToPath

    Write-Host "Installed $asset to $InstallDir"
    $meshBinary = Join-Path $InstallDir "mesh-llm.exe"
    if (Test-Path $meshBinary) {
        & $meshBinary --version
    }
} finally {
    if (Test-Path $tmpRoot) {
        Remove-Item $tmpRoot -Recurse -Force
    }
}

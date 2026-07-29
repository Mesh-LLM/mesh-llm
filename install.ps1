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

    # Windows PowerShell 5.1 can expose RuntimeInformation while returning a
    # null OSArchitecture value. Casting avoids calling ToString() on null and
    # PROCESSOR_ARCHITECTURE provides the legacy Windows fallback.
    $arch = [string][System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    if (-not $arch) {
        $arch = [string]$env:PROCESSOR_ARCHITECTURE
    }

    if (-not $arch -or @("X64", "AMD64", "X86_64") -notcontains $arch) {
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
        if ($name -eq "mesh-llm.exe") {
            continue
        }
        $path = Join-Path $InstallDir $name
        if (Test-Path $path) {
            Remove-Item $path -Force
        }
    }
}

function Convert-HexToBytes {
    param([string]$Hex)

    $bytes = New-Object byte[] ($Hex.Length / 2)
    for ($index = 0; $index -lt $bytes.Length; $index++) {
        $bytes[$index] = [Convert]::ToByte($Hex.Substring($index * 2, 2), 16)
    }
    return $bytes
}

function Convert-UInt64ToBigEndianBytes {
    param([UInt64]$Value)

    $bytes = [BitConverter]::GetBytes($Value)
    if ([BitConverter]::IsLittleEndian) {
        [Array]::Reverse($bytes)
    }
    return $bytes
}

function Update-Sha256Bytes {
    param(
        [System.Security.Cryptography.HashAlgorithm]$Hasher,
        [byte[]]$Bytes
    )

    if ($Bytes.Length -eq 0) {
        return
    }
    [void]$Hasher.TransformBlock($Bytes, 0, $Bytes.Length, $Bytes, 0)
}

function Complete-Sha256 {
    param([System.Security.Cryptography.HashAlgorithm]$Hasher)

    [void]$Hasher.TransformFinalBlock([byte[]]@(), 0, 0)
    return ([System.BitConverter]::ToString($Hasher.Hash).Replace("-", "")).ToLowerInvariant()
}

function Get-DeterministicTreeSha256 {
    param([string]$Path)

    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $root = (Resolve-Path -LiteralPath $Path).ProviderPath.TrimEnd([char]'\')
        $files = Get-ChildItem -LiteralPath $Path -Recurse -File | Sort-Object FullName
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($root.Length).TrimStart([char]'\') -replace '\\', '/'
            $relativeBytes = [System.Text.Encoding]::UTF8.GetBytes($relative)
            $relativeLength = Convert-UInt64ToBigEndianBytes ([UInt64]$relativeBytes.Length)
            $fileDigest = Convert-HexToBytes ((Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant())
            Update-Sha256Bytes -Hasher $hasher -Bytes $relativeLength
            Update-Sha256Bytes -Hasher $hasher -Bytes $relativeBytes
            Update-Sha256Bytes -Hasher $hasher -Bytes $fileDigest
        }
        return Complete-Sha256 -Hasher $hasher
    } finally {
        $hasher.Dispose()
    }
}

function Assert-JsonProperty {
    param(
        [object]$Object,
        [string]$Name,
        [string]$Label
    )

    if (-not $Object -or -not ($Object.PSObject.Properties.Name -contains $Name)) {
        throw "product-manifest.json missing $Label"
    }
    return $Object.$Name
}

function Assert-StringField {
    param(
        [object]$Value,
        [string]$Label
    )

    if ($Value -isnot [string] -or -not $Value) {
        throw "product-manifest.json field $Label must be a non-empty string"
    }
    return $Value
}

function Assert-Sha256Field {
    param(
        [object]$Value,
        [string]$Label
    )

    $digest = Assert-StringField -Value $Value -Label $Label
    if ($digest -cnotmatch '^[0-9a-f]{64}$') {
        throw "product-manifest.json field $Label must be a lowercase SHA-256 digest"
    }
    return $digest
}

function Assert-SafeRelativePath {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not $Path -or [System.IO.Path]::IsPathRooted($Path) -or $Path.Contains("\")) {
        throw "product-manifest.json field $Label must be a safe relative POSIX path"
    }
    $parts = $Path.Split('/')
    foreach ($part in $parts) {
        if (-not $part -or $part -eq "." -or $part -eq "..") {
            throw "product-manifest.json field $Label must be a safe relative POSIX path"
        }
    }
    return $Path
}

function Assert-ProductBundle {
    param([string]$BundleDir)

    $productManifestSource = Join-Path $BundleDir "product-manifest.json"
    $runtimeRoot = Join-Path $BundleDir "native-runtimes"
    if (-not (Test-Path $productManifestSource -PathType Leaf) -or -not (Test-Path $runtimeRoot -PathType Container)) {
        throw "release archive did not contain a composed native runtime bundle"
    }

    $manifest = Get-Content -Path $productManifestSource -Raw | ConvertFrom-Json
    $schemaVersion = Assert-JsonProperty -Object $manifest -Name "schema_version" -Label "schema_version"
    if ([int]$schemaVersion -ne 2) {
        throw "product-manifest.json schema_version must be 2"
    }
    $contract = Assert-StringField -Value (Assert-JsonProperty -Object $manifest -Name "contract" -Label "contract") -Label "contract"
    if ($contract -ne "mesh-llm-product-v2") {
        throw "product-manifest.json contract must be mesh-llm-product-v2"
    }
    [void](Assert-StringField -Value (Assert-JsonProperty -Object $manifest -Name "mesh_version" -Label "mesh_version") -Label "mesh_version")
    [void](Assert-StringField -Value (Assert-JsonProperty -Object $manifest -Name "backend" -Label "backend") -Label "backend")

    $host = Assert-JsonProperty -Object $manifest -Name "host" -Label "host"
    $hostPath = Assert-SafeRelativePath -Path (Assert-StringField -Value (Assert-JsonProperty -Object $host -Name "path" -Label "host.path") -Label "host.path") -Label "host.path"
    if ($hostPath -ne "mesh-llm.exe") {
        throw "product-manifest.json host.path must be mesh-llm.exe"
    }
    $hostSha256 = Assert-Sha256Field -Value (Assert-JsonProperty -Object $host -Name "sha256" -Label "host.sha256") -Label "host.sha256"
    $hostSource = Join-Path $BundleDir $hostPath
    if (-not (Test-Path $hostSource -PathType Leaf)) {
        throw "product-manifest.json referenced host path was not found: $hostPath"
    }
    $actualHostSha256 = (Get-FileHash -Path $hostSource -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHostSha256 -ne $hostSha256) {
        throw "host.sha256 mismatch: expected $hostSha256, got $actualHostSha256"
    }

    $runtime = Assert-JsonProperty -Object $manifest -Name "runtime" -Label "runtime"
    $runtimeId = Assert-StringField -Value (Assert-JsonProperty -Object $runtime -Name "id" -Label "runtime.id") -Label "runtime.id"
    $runtimePath = Assert-SafeRelativePath -Path (Assert-StringField -Value (Assert-JsonProperty -Object $runtime -Name "path" -Label "runtime.path") -Label "runtime.path") -Label "runtime.path"
    $runtimeSha256 = Assert-Sha256Field -Value (Assert-JsonProperty -Object $runtime -Name "sha256" -Label "runtime.sha256") -Label "runtime.sha256"
    $runtimeManifestSha256 = Assert-Sha256Field -Value (Assert-JsonProperty -Object $runtime -Name "manifest_sha256" -Label "runtime.manifest_sha256") -Label "runtime.manifest_sha256"
    $runtimeParts = $runtimePath.Split('/')
    if ($runtimeParts.Length -ne 2 -or $runtimeParts[0] -ne "native-runtimes" -or $runtimeParts[1] -ne $runtimeId) {
        throw "product-manifest.json runtime.path must be native-runtimes/$runtimeId"
    }
    $runtimeSource = Join-Path $BundleDir $runtimePath
    if (-not (Test-Path $runtimeSource -PathType Container)) {
        throw "product-manifest.json referenced runtime path was not found: $runtimePath"
    }
    $runtimeChildren = @(Get-ChildItem -LiteralPath $runtimeRoot)
    if ($runtimeChildren.Count -ne 1 -or -not $runtimeChildren[0].PSIsContainer -or $runtimeChildren[0].Name -ne $runtimeId) {
        throw "release archive must contain exactly one selected native runtime tree"
    }
    $runtimeManifestSource = Join-Path $runtimeSource "manifest.json"
    if (-not (Test-Path $runtimeManifestSource -PathType Leaf)) {
        throw "product-manifest.json referenced runtime manifest was not found: $runtimePath/manifest.json"
    }
    $actualRuntimeManifestSha256 = (Get-FileHash -Path $runtimeManifestSource -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualRuntimeManifestSha256 -ne $runtimeManifestSha256) {
        throw "runtime.manifest_sha256 mismatch: expected $runtimeManifestSha256, got $actualRuntimeManifestSha256"
    }
    $actualRuntimeSha256 = Get-DeterministicTreeSha256 -Path $runtimeSource
    if ($actualRuntimeSha256 -ne $runtimeSha256) {
        throw "runtime.sha256 mismatch: expected $runtimeSha256, got $actualRuntimeSha256"
    }

    return [PSCustomObject]@{
        HostSource = $hostSource
        RuntimeSource = $runtimeSource
        ProductManifestSource = $productManifestSource
        HostImportsSource = Join-Path $BundleDir "host-imports.json"
    }
}

function Remove-InstallStagingPath {
    param([string]$Path)

    if (Test-Path $Path) {
        Remove-Item $Path -Recurse -Force
    }
}

function Move-IfExists {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (Test-Path $Source) {
        Move-Item -Path $Source -Destination $Destination -Force
        return $true
    }
    return $false
}

function Restore-InstallBackup {
    param([object]$Paths)

    Remove-InstallStagingPath -Path $Paths.MeshBinaryDestination
    Remove-InstallStagingPath -Path $Paths.RuntimeDestination
    Remove-InstallStagingPath -Path $Paths.ProductManifestDestination
    Remove-InstallStagingPath -Path $Paths.HostImportsDestination
    if (Test-Path $Paths.MeshBinaryBackup) {
        Move-Item -Path $Paths.MeshBinaryBackup -Destination $Paths.MeshBinaryDestination -Force
    }
    if (Test-Path $Paths.RuntimeBackup) {
        Move-Item -Path $Paths.RuntimeBackup -Destination $Paths.RuntimeDestination -Force
    }
    if (Test-Path $Paths.ProductManifestBackup) {
        Move-Item -Path $Paths.ProductManifestBackup -Destination $Paths.ProductManifestDestination -Force
    }
    if (Test-Path $Paths.HostImportsBackup) {
        Move-Item -Path $Paths.HostImportsBackup -Destination $Paths.HostImportsDestination -Force
    }
}

function Remove-InstallBackups {
    param([object]$Paths)

    Remove-InstallStagingPath -Path $Paths.MeshBinaryBackup
    Remove-InstallStagingPath -Path $Paths.RuntimeBackup
    Remove-InstallStagingPath -Path $Paths.ProductManifestBackup
    Remove-InstallStagingPath -Path $Paths.HostImportsBackup
}

function Stage-IncomingBundle {
    param(
        [object]$Bundle,
        [object]$Paths
    )

    try {
        Copy-Item -Path $Bundle.HostSource -Destination $Paths.MeshBinaryStaging -Force
        Copy-Item -Path $Bundle.RuntimeSource -Destination $Paths.RuntimeStaging -Recurse -Force
        Copy-Item -Path $Bundle.ProductManifestSource -Destination $Paths.ProductManifestStaging -Force
        if (Test-Path $Bundle.HostImportsSource -PathType Leaf) {
            Copy-Item -Path $Bundle.HostImportsSource -Destination $Paths.HostImportsStaging -Force
        }
    } catch {
        Remove-InstallStagingPath -Path $Paths.MeshBinaryStaging
        Remove-InstallStagingPath -Path $Paths.RuntimeStaging
        Remove-InstallStagingPath -Path $Paths.ProductManifestStaging
        Remove-InstallStagingPath -Path $Paths.HostImportsStaging
        throw
    }
}

function Install-MeshBinary {
    param([string]$BundleDir)

    $bundle = Assert-ProductBundle -BundleDir $BundleDir

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $paths = [PSCustomObject]@{
        MeshBinaryDestination = Join-Path $InstallDir "mesh-llm.exe"
        RuntimeDestination = Join-Path $InstallDir "native-runtimes"
        ProductManifestDestination = Join-Path $InstallDir "product-manifest.json"
        HostImportsDestination = Join-Path $InstallDir "host-imports.json"
        MeshBinaryStaging = Join-Path $InstallDir "mesh-llm.exe.incoming"
        RuntimeStaging = Join-Path $InstallDir "native-runtimes.incoming"
        ProductManifestStaging = Join-Path $InstallDir "product-manifest.json.incoming"
        HostImportsStaging = Join-Path $InstallDir "host-imports.json.incoming"
        MeshBinaryBackup = Join-Path $InstallDir "mesh-llm.exe.backup"
        RuntimeBackup = Join-Path $InstallDir "native-runtimes.backup"
        ProductManifestBackup = Join-Path $InstallDir "product-manifest.json.backup"
        HostImportsBackup = Join-Path $InstallDir "host-imports.json.backup"
    }

    Remove-InstallStagingPath -Path $paths.MeshBinaryStaging
    Remove-InstallStagingPath -Path $paths.RuntimeStaging
    Remove-InstallStagingPath -Path $paths.ProductManifestStaging
    Remove-InstallStagingPath -Path $paths.HostImportsStaging
    Remove-InstallBackups -Paths $paths

    Stage-IncomingBundle -Bundle $bundle -Paths $paths

    $hostImportsDestination = $paths.HostImportsDestination
    try {
        [void](Move-IfExists -Source $paths.MeshBinaryDestination -Destination $paths.MeshBinaryBackup)
        [void](Move-IfExists -Source $paths.RuntimeDestination -Destination $paths.RuntimeBackup)
        [void](Move-IfExists -Source $paths.ProductManifestDestination -Destination $paths.ProductManifestBackup)
        [void](Move-IfExists -Source $paths.HostImportsDestination -Destination $paths.HostImportsBackup)

        Move-Item -Path $paths.MeshBinaryStaging -Destination $paths.MeshBinaryDestination -Force
        Move-Item -Path $paths.RuntimeStaging -Destination $paths.RuntimeDestination -Force
        if ((Test-Truthy $env:MESH_LLM_INSTALL_TEST_ALLOW_NONWINDOWS) -and (Test-Truthy $env:MESH_LLM_INSTALL_TEST_FAIL_AFTER_RUNTIME_REPLACE)) {
            throw "test requested failure after runtime replacement"
        }
        Move-Item -Path $paths.ProductManifestStaging -Destination $paths.ProductManifestDestination -Force
        if (Test-Path $paths.HostImportsStaging) {
            Move-Item -Path $paths.HostImportsStaging -Destination $paths.HostImportsDestination -Force
        } else {
            if (Test-Path $paths.HostImportsDestination) {
                Remove-Item $hostImportsDestination -Force
            }
        }
        Remove-InstallBackups -Paths $paths
        Remove-StaleBinaries
    } catch {
        Restore-InstallBackup -Paths $paths
        throw
    } finally {
        Remove-InstallStagingPath -Path $paths.MeshBinaryStaging
        Remove-InstallStagingPath -Path $paths.RuntimeStaging
        Remove-InstallStagingPath -Path $paths.ProductManifestStaging
        Remove-InstallStagingPath -Path $paths.HostImportsStaging
    }
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

param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$OutputDir = "dist",
    [string]$Flavor = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$releaseBinDir = Join-Path $repoRoot "target\release"
$attestationSigningKeyFile = $env:MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE
$attestationPublicKeyFile = $env:MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE

Add-Type -AssemblyName System.IO.Compression.FileSystem

function Normalize-RecipeArgument {
    param(
        [AllowEmptyString()]
        [string]$Value,
        [string[]]$KnownNames = @()
    )

    if ($null -eq $Value) {
        return $Value
    }

    $normalized = $Value.Trim()
    if (-not $normalized) {
        return ""
    }

    if ($normalized -match '^(?<name>[A-Za-z_][A-Za-z0-9_-]*)=(?<value>.*)$') {
        $matchedName = $Matches.name
        $isKnownName = $KnownNames.Count -eq 0
        foreach ($knownName in $KnownNames) {
            if ($matchedName.Equals($knownName, [System.StringComparison]::OrdinalIgnoreCase)) {
                $isKnownName = $true
                break
            }
        }

        if ($isKnownName) {
            $normalized = $Matches.value
        }
    }

    if ($normalized.Length -ge 2) {
        $first = $normalized[0]
        $last = $normalized[$normalized.Length - 1]
        if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
            $normalized = $normalized.Substring(1, $normalized.Length - 2)
        }
    }

    return $normalized.Trim()
}

function Get-ReleaseFlavor {
    param([string]$RequestedFlavor)

    if ($RequestedFlavor) {
        switch ($RequestedFlavor.ToLowerInvariant()) {
            "hip" { return "rocm" }
            default { return $RequestedFlavor.ToLowerInvariant() }
        }
    }

    return "cpu"
}

function Get-BinaryFlavor {
    param([string]$RequestedFlavor)

    # The "release flavor" (outer archive name) and the "binary flavor"
    # (inner executable suffix / runtime BinaryFlavor lookup) are not
    # always the same. hip archives contain -rocm binaries, while
    # cuda-blackwell keeps its distinct runtime flavor suffix.
    if ($RequestedFlavor) {
        switch ($RequestedFlavor.ToLowerInvariant()) {
            "hip" { return "rocm" }
            default { return $RequestedFlavor.ToLowerInvariant() }
        }
    }

    return "cpu"
}

function Get-FlavorSuffix {
    param([string]$BinaryFlavor)

    if (-not $BinaryFlavor -or $BinaryFlavor -in @("cpu", "metal")) {
        return ""
    }

    return "-$BinaryFlavor"
}

function New-ReleaseAssetName {
    param(
        [string]$Prefix,
        [string]$TargetTriple,
        [string]$ArchiveExt,
        [string]$BinaryFlavor
    )

    return "$Prefix-$TargetTriple$(Get-FlavorSuffix $BinaryFlavor).$ArchiveExt"
}

function Get-BundleBinaryName {
    param(
        [string]$BaseName,
        [string]$BinaryFlavor
    )

    if ($BaseName -eq "mesh-llm") {
        return "$BaseName.exe"
    }

    if ($BinaryFlavor) {
        return "$BaseName-$BinaryFlavor.exe"
    }

    return "$BaseName.exe"
}

function New-ZipArchive {
    param(
        [string]$SourceDir,
        [string]$ArchivePath
    )

    if (Test-Path $ArchivePath) {
        Remove-Item $ArchivePath -Force
    }

    $parent = Split-Path -Parent $ArchivePath
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $SourceDir,
        $ArchivePath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $true
    )
}

function Require-File {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Required file not found: $Path"
    }
}

function Resolve-VulkanRuntimeDll {
    $candidates = @()

    if ($env:VULKAN_SDK) {
        $candidates += (Join-Path $env:VULKAN_SDK "Bin\vulkan-1.dll")
    }

    $vulkanSdkRoot = "C:\VulkanSDK"
    if (Test-Path $vulkanSdkRoot) {
        $candidates += Get-ChildItem -Path $vulkanSdkRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "Bin\vulkan-1.dll" }
    }

    $candidates += (Join-Path $env:WINDIR "System32\vulkan-1.dll")

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    throw "Vulkan runtime DLL not found. Install the Vulkan SDK/runtime so vulkan-1.dll is available before packaging."
}

function Resolve-CudaBinDir {
    $candidates = @()

    if ($env:CUDA_PATH) {
        $candidates += (Join-Path $env:CUDA_PATH "bin")
    }

    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        $candidates += Get-ChildItem -Path $cudaRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "bin" }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    throw "CUDA toolkit bin directory not found. Install the CUDA toolkit before packaging a CUDA release."
}

function Copy-CudaRuntimeDependencies {
    param([string]$BundleDir)

    $cudaBin = Resolve-CudaBinDir
    $requiredPatterns = @(
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll"
    )
    $optionalPatterns = @(
        "nvJitLink_*.dll",
        "nvrtc64_*.dll",
        "nvrtc-builtins64_*.dll"
    )
    $copied = @()

    foreach ($pattern in $requiredPatterns) {
        $matches = @(Get-ChildItem -Path $cudaBin -Filter $pattern -File -ErrorAction SilentlyContinue | Sort-Object Name)
        if ($matches.Count -eq 0) {
            throw "CUDA runtime DLL not found: $pattern under $cudaBin"
        }

        foreach ($dll in $matches) {
            Copy-Item $dll.FullName -Destination (Join-Path $BundleDir $dll.Name) -Force
            $copied += $dll.FullName
        }
    }

    foreach ($pattern in $optionalPatterns) {
        $matches = @(Get-ChildItem -Path $cudaBin -Filter $pattern -File -ErrorAction SilentlyContinue | Sort-Object Name)
        foreach ($dll in $matches) {
            Copy-Item $dll.FullName -Destination (Join-Path $BundleDir $dll.Name) -Force
            $copied += $dll.FullName
        }
    }

    foreach ($source in ($copied | Select-Object -Unique)) {
        Write-Host "Bundled CUDA runtime dependency: $source"
    }
}

function Copy-RuntimeDependencies {
    param(
        [string]$BundleDir,
        [string]$BinaryFlavor
    )

    switch ($BinaryFlavor) {
        "vulkan" {
            $vulkanDll = Resolve-VulkanRuntimeDll
            Copy-Item $vulkanDll -Destination (Join-Path $BundleDir "vulkan-1.dll") -Force
            Write-Host "Bundled Vulkan runtime dependency: $vulkanDll"
            return
        }
        { $_ -in @("cuda", "cuda-blackwell") } {
            Copy-CudaRuntimeDependencies -BundleDir $BundleDir
            return
        }
    }
}

function Test-BinaryContainsAsciiText {
    param(
        [string]$Path,
        [string]$Text
    )

    $binaryText = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($Path))
    return $binaryText.Contains($Text)
}

function Assert-MeshBinaryVersion {
    param(
        [string]$Path,
        [string]$ExpectedVersion,
        [string]$BinaryFlavor
    )

    $expected = $ExpectedVersion.TrimStart("v")
    $output = & $Path --version
    if ($LASTEXITCODE -ne 0) {
        if ($BinaryFlavor -in @("cuda", "cuda-blackwell") -and $LASTEXITCODE -eq -1073741515) {
            if (Test-BinaryContainsAsciiText -Path $Path -Text $expected) {
                Write-Warning "CUDA release binary could not start on this driverless Windows runner; verified embedded version string $expected instead."
                return
            }
        }

        throw "Release binary failed --version with exit code ${LASTEXITCODE}: $Path"
    }

    $parts = "$output".Trim() -split '\s+'
    $actual = if ($parts.Count -gt 0) { $parts[$parts.Count - 1] } else { "" }
    if ($actual -ne $expected) {
        throw "Release binary version mismatch: expected $expected, got ${actual}. Binary: $Path. Output: $output"
    }
}

function Test-HasValue {
    param([string]$Value)

    return -not [string]::IsNullOrWhiteSpace($Value)
}

function Assert-AttestationConfig {
    if ((Test-HasValue $attestationSigningKeyFile) -and -not (Test-HasValue $attestationPublicKeyFile)) {
        throw "MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE is required when MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE is set"
    }

    if (-not (Test-HasValue $attestationSigningKeyFile) -and (Test-HasValue $attestationPublicKeyFile)) {
        throw "MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE is required when MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE is set"
    }
}

function Invoke-ReleaseAttestationStamp {
    param([string]$BinaryPath)

    $inspectJson = $null

    if (-not (Test-HasValue $attestationSigningKeyFile)) {
        Write-Host "Release attestation: missing (packaged binary left unstamped)"
        return
    }

    Push-Location $repoRoot
    try {
        & cargo run -q -p xtask -- release-attestation stamp `
            --binary $BinaryPath `
            --signing-key-file $attestationSigningKeyFile | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "release-attestation stamp failed for $BinaryPath"
        }

        $inspectJson = & cargo run -q -p xtask -- release-attestation inspect `
            --binary $BinaryPath `
            --public-key-file $attestationPublicKeyFile `
            --json
        if ($LASTEXITCODE -ne 0) {
            throw "release-attestation inspect failed for $BinaryPath"
        }
        Write-Host $inspectJson
        $inspectStatus = ($inspectJson | ConvertFrom-Json).status
        if ($inspectStatus -ne "valid") {
            throw "release-attestation inspect reported status '$inspectStatus' for $BinaryPath"
        }
    } finally {
        Pop-Location
    }
}

$Version = Normalize-RecipeArgument $Version @("version")
$OutputDir = Normalize-RecipeArgument $OutputDir @("output", "output_dir", "outputdir")
$Flavor = Normalize-RecipeArgument $Flavor @("flavor", "backend")

Assert-AttestationConfig

$releaseFlavor = Get-ReleaseFlavor $Flavor
$binaryFlavor = Get-BinaryFlavor $Flavor
$targetTriple = "x86_64-pc-windows-msvc"
$archiveExt = "zip"
# Outer archive names use the release flavor (e.g. cuda-blackwell); inner
# binary names use the binary flavor (e.g. cuda) so the runtime finds them.
$stableAsset = New-ReleaseAssetName -Prefix "mesh-llm" -TargetTriple $targetTriple -ArchiveExt $archiveExt -BinaryFlavor $releaseFlavor
$versionedAsset = New-ReleaseAssetName -Prefix "mesh-llm-$Version" -TargetTriple $targetTriple -ArchiveExt $archiveExt -BinaryFlavor $releaseFlavor

$meshBinary = Join-Path $releaseBinDir "mesh-llm.exe"

Require-File $meshBinary

$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}
New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null

$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("mesh-llm-release-" + [System.Guid]::NewGuid().ToString("N"))
$bundleDir = Join-Path $stagingRoot "mesh-bundle"
New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

try {
    $bundleBinary = Join-Path $bundleDir (Get-BundleBinaryName "mesh-llm" $binaryFlavor)
    Copy-Item $meshBinary -Destination $bundleBinary -Force
    Copy-RuntimeDependencies -BundleDir $bundleDir -BinaryFlavor $binaryFlavor
    Assert-MeshBinaryVersion -Path $bundleBinary -ExpectedVersion $Version -BinaryFlavor $binaryFlavor

    Invoke-ReleaseAttestationStamp -BinaryPath $bundleBinary
    $versionedPath = Join-Path $resolvedOutputDir $versionedAsset
    $stablePath = Join-Path $resolvedOutputDir $stableAsset

    New-ZipArchive -SourceDir $bundleDir -ArchivePath $versionedPath
    New-ZipArchive -SourceDir $bundleDir -ArchivePath $stablePath

    Write-Host "Created release archives:"
    Get-ChildItem -Path $resolvedOutputDir -File | Sort-Object Name | ForEach-Object {
        Write-Host $_.FullName
    }
} finally {
    if (Test-Path $stagingRoot) {
        Remove-Item $stagingRoot -Recurse -Force
    }
}

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
$nativeRuntimeRoot = if ($env:MESH_LLM_NATIVE_RUNTIME_ROOT) {
    $env:MESH_LLM_NATIVE_RUNTIME_ROOT
} else {
    Join-Path $repoRoot "dist\native-runtimes"
}
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
    # always the same. hip archives contain -rocm binaries.
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

function Get-Sha256Hex {
    param([string]$Path)

    # Use the .NET SHA-256 API directly instead of Get-FileHash. Under
    # `powershell -NoProfile` on the CI runners, Microsoft.PowerShell.Utility
    # module autoloading does not always resolve Get-FileHash, which caused
    # release bundling to fail with CommandNotFoundException. The .NET API is
    # always available regardless of module autoloading.
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            $bytes = $sha256.ComputeHash($stream)
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha256.Dispose()
    }
    return [System.BitConverter]::ToString($bytes).Replace("-", "").ToLowerInvariant()
}

function New-ChecksumSidecar {
    param([string]$Path)

    $hash = Get-Sha256Hex $Path
    $name = Split-Path -Leaf $Path
    Set-Content -Path "$Path.sha256" -Value "$hash  $name" -NoNewline
}

function Require-File {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Required file not found: $Path"
    }
}

function Get-PythonCommand {
    foreach ($name in @("python3", "python")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }
    throw "python3 or python is required for release packaging"
}

function Assert-MeshBinaryVersion {
    param(
        [string]$Path,
        [string]$ExpectedVersion
    )

    $expected = $ExpectedVersion.TrimStart("v")
    $output = & $Path --version
    if ($LASTEXITCODE -ne 0) {
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

    if (-not (Test-Path $attestationSigningKeyFile) -or (Get-Item $attestationSigningKeyFile).Length -eq 0) {
        Write-Host "Release attestation: signing key file is empty or missing ($attestationSigningKeyFile); leaving binary unstamped"
        return
    }

    if (-not (Test-Path $attestationPublicKeyFile) -or (Get-Item $attestationPublicKeyFile).Length -eq 0) {
        Write-Host "Release attestation: public key file is empty or missing ($attestationPublicKeyFile); leaving binary unstamped"
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
# Outer archive names use the release flavor; inner
# binary names use the binary flavor so the runtime finds them.
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
    Assert-MeshBinaryVersion -Path $bundleBinary -ExpectedVersion $Version

    Invoke-ReleaseAttestationStamp -BinaryPath $bundleBinary
    $python = Get-PythonCommand
    $hostReport = Join-Path $bundleDir "host-imports.json"
    & $python (Join-Path $scriptDir "verify-host-dependencies.py") $bundleBinary --report $hostReport
    if ($LASTEXITCODE -ne 0) {
        throw "backend-neutral host dependency verification failed"
    }

    $cudaMajor = if ($env:MESH_LLM_CUDA_TOOLKIT_MAJOR) {
        $env:MESH_LLM_CUDA_TOOLKIT_MAJOR
    } elseif ($env:MESH_CUDA_VERSION) {
        ($env:MESH_CUDA_VERSION -split '\.')[0]
    } else {
        ""
    }
    $selectorArgs = @(
        (Join-Path $scriptDir "select-native-runtime.py")
        "--root"
        $nativeRuntimeRoot
        "--os"
        "windows"
        "--arch"
        "x86_64"
        "--backend"
        $releaseFlavor
    )
    if (Test-HasValue $cudaMajor) {
        $selectorArgs += @("--cuda-major", $cudaMajor)
    }
    $selectorOutput = & $python @selectorArgs
    $selectorExitCode = $LASTEXITCODE
    if ($selectorExitCode -ne 0) {
        throw "failed to select the packaged Windows native runtime"
    }
    $runtimeDir = $selectorOutput | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -Last 1
    if (-not $runtimeDir) {
        throw "failed to select the packaged Windows native runtime"
    }
    $runtimeDestinationRoot = Join-Path $bundleDir "native-runtimes"
    $runtimeDestination = Join-Path $runtimeDestinationRoot (Split-Path -Leaf $runtimeDir)
    New-Item -ItemType Directory -Path $runtimeDestinationRoot -Force | Out-Null
    Copy-Item $runtimeDir -Destination $runtimeDestination -Recurse -Force

    & $python (Join-Path $scriptDir "compose-product-bundle.py") `
        --bundle $bundleDir `
        --host $bundleBinary `
        --runtime $runtimeDestination `
        --version $Version `
        --backend $releaseFlavor
    if ($LASTEXITCODE -ne 0) {
        throw "failed to write the product-v2 bundle manifest"
    }
    $versionedPath = Join-Path $resolvedOutputDir $versionedAsset
    $stablePath = Join-Path $resolvedOutputDir $stableAsset

    New-ZipArchive -SourceDir $bundleDir -ArchivePath $versionedPath
    New-ChecksumSidecar -Path $versionedPath
    New-ZipArchive -SourceDir $bundleDir -ArchivePath $stablePath
    New-ChecksumSidecar -Path $stablePath

    Write-Host "Created release archives:"
    Get-ChildItem -Path $resolvedOutputDir -File | Sort-Object Name | ForEach-Object {
        Write-Host $_.FullName
    }
} finally {
    if (Test-Path $stagingRoot) {
        Remove-Item $stagingRoot -Recurse -Force
    }
}

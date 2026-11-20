# Package the Windows Mesh host into the release zip archives.
#
# The Skippy native runtime is not built here. MESH_LLM_NATIVE_RUNTIME_DIR must
# name one already-downloaded runtime directory (with its schema 2
# manifest.json); it is bundled under native-runtimes\ and described in
# product-manifest.json. Signing follows scripts/package-release.sh.
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$OutputDir = "dist",
    [string]$Flavor = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../../scripts"))
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$releaseBinDir = if ($env:MESH_LLM_RELEASE_BIN_DIR) { $env:MESH_LLM_RELEASE_BIN_DIR } else { Join-Path $repoRoot "target\release" }
$nativeRuntimeDir = $env:MESH_LLM_NATIVE_RUNTIME_DIR
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

    # When MESH_CUDA_VERSION is set (release matrix), include the CUDA major,
    # as package-release.sh does: -cuda-12, -cuda-13.
    if ($BinaryFlavor -eq "cuda" -and $env:MESH_CUDA_VERSION) {
        $major = ("$env:MESH_CUDA_VERSION".Trim() -split '\.', 2)[0]
        return "-$BinaryFlavor-$major"
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

function Invoke-ReleaseAttestation {
    Push-Location $repoRoot
    try {
        $output = & cargo run --locked --quiet -p mesh-llm-release-attestation -- @args
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw "mesh-llm-release-attestation $($args[0]) failed with exit code $LASTEXITCODE"
    }
    return $output
}

function Invoke-ReleaseAttestationStamp {
    param([string]$BinaryPath)

    $hasSigning = Test-HasValue $attestationSigningKeyFile
    $hasPublic = Test-HasValue $attestationPublicKeyFile
    if (-not $hasSigning -and -not $hasPublic) {
        Write-Host "Release attestation: missing (packaged binary left unstamped)"
        return
    }
    if ($hasSigning -ne $hasPublic) {
        throw "MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE and MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE must be set together"
    }
    foreach ($keyFile in @($attestationSigningKeyFile, $attestationPublicKeyFile)) {
        if (-not (Test-Path $keyFile) -or (Get-Item $keyFile).Length -eq 0) {
            Write-Host "Release attestation: key file empty or missing; packaged binary left unstamped"
            return
        }
    }
    # A signed stamp must name the commit it was built from. MESH_RELEASE_COMMIT
    # overrides the checkout's HEAD; an unknown commit is refused, not invented.
    $stampArgs = @("--binary", $BinaryPath, "--signing-key-file", $attestationSigningKeyFile, "--require-source-commit", "--target-triple", $targetTriple)
    if (Test-HasValue $env:MESH_RELEASE_COMMIT) { $stampArgs += @("--commit", $env:MESH_RELEASE_COMMIT) }
    Invoke-ReleaseAttestation stamp @stampArgs | Out-Null
    $inspectJson = Invoke-ReleaseAttestation inspect --binary $BinaryPath --public-key-file $attestationPublicKeyFile --json
    $status = (($inspectJson -join "`n") | ConvertFrom-Json).status
    if ($status -ne "valid") {
        throw "release attestation for $BinaryPath is '$status', expected valid"
    }
    Write-Host "Release attestation: stamped and verified"
}

# Same digest as mesh-llm-system's product_runtime_tree_sha256: over files sorted
# ordinally by relative path, u64 BE path length, path bytes, file SHA-256.
function Get-TreeSha256 {
    param([string]$Root)

    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd("\") + "\"
    $entries = Get-ChildItem -Path $Root -Recurse -File | ForEach-Object {
        [pscustomobject]@{ Relative = $_.FullName.Substring($rootFull.Length).Replace("\", "/"); Path = $_.FullName }
    }
    $sorted = [System.Collections.Generic.List[object]]::new()
    $entries | ForEach-Object { $sorted.Add($_) }
    $sorted.Sort([System.Comparison[object]] { param($a, $b) [string]::CompareOrdinal($a.Relative, $b.Relative) })
    $tree = [System.Security.Cryptography.SHA256]::Create()
    try {
        foreach ($entry in $sorted) {
            $pathBytes = [System.Text.Encoding]::UTF8.GetBytes($entry.Relative)
            $length = [System.BitConverter]::GetBytes([uint64]$pathBytes.Length)
            if ([System.BitConverter]::IsLittleEndian) { [System.Array]::Reverse($length) }
            $fileHash = [System.Security.Cryptography.SHA256]::Create()
            try {
                $stream = [System.IO.File]::OpenRead($entry.Path)
                try { $digest = $fileHash.ComputeHash($stream) } finally { $stream.Dispose() }
            } finally {
                $fileHash.Dispose()
            }
            foreach ($chunk in @($length, $pathBytes, $digest)) {
                [void]$tree.TransformBlock($chunk, 0, $chunk.Length, $null, 0)
            }
        }
        [void]$tree.TransformFinalBlock([byte[]]::new(0), 0, 0)
        return [System.BitConverter]::ToString($tree.Hash).Replace("-", "").ToLowerInvariant()
    } finally {
        $tree.Dispose()
    }
}

# Write product-manifest.json (contract mesh-llm-product-v2), refusing a runtime
# whose backend or ABI the host can't use.
function Write-ProductManifest {
    param(
        [string]$BundleDir,
        [string]$HostPath,
        [string]$RuntimeDir,
        [string]$ProductVersion,
        [string]$Backend
    )

    $expectedKind = switch ($Backend) {
        { $_ -in @("cuda", "cuda-blackwell") } { "cuda" }
        { $_ -in @("rocm", "hip") } { "rocm" }
        default { $Backend }
    }
    $contractText = (& $HostPath --log-format json --print-build-contract | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0) { throw "host --print-build-contract failed: $HostPath" }
    $contract = $contractText | ConvertFrom-Json
    if ($contract.product_version -ne $ProductVersion) {
        throw "host build contract product version $($contract.product_version) does not match $ProductVersion"
    }
    $runtimeManifestPath = Join-Path $RuntimeDir "manifest.json"
    $runtimeManifest = Get-Content -Path $runtimeManifestPath -Raw | ConvertFrom-Json
    if ($runtimeManifest.schema_version -ne 2) {
        throw "native runtime manifest requires schema_version 2: $runtimeManifestPath"
    }
    $runtime = $runtimeManifest.runtime
    if ($runtime.backend.kind -ne $expectedKind) {
        throw "native runtime backend is $($runtime.backend.kind), expected $expectedKind for $Backend"
    }
    if ($runtime.skippy_abi -ne $contract.skippy_abi) {
        throw "native runtime ABI $($runtime.skippy_abi) does not match host-required $($contract.skippy_abi)"
    }
    $bundleFull = [System.IO.Path]::GetFullPath($BundleDir).TrimEnd("\") + "\"
    $product = [ordered]@{
        schema_version = 2
        contract = "mesh-llm-product-v2"
        mesh_version = $ProductVersion
        backend = $Backend
        host = [ordered]@{
            path = [System.IO.Path]::GetFullPath($HostPath).Substring($bundleFull.Length).Replace("\", "/")
            sha256 = Get-Sha256Hex $HostPath
            required_skippy_abi = $contract.skippy_abi
        }
        runtime = [ordered]@{
            id = $runtime.id
            release_version = $runtime.release_version
            skippy_abi = $runtime.skippy_abi
            path = [System.IO.Path]::GetFullPath($RuntimeDir).Substring($bundleFull.Length).Replace("\", "/")
            sha256 = Get-TreeSha256 $RuntimeDir
            manifest_sha256 = Get-Sha256Hex $runtimeManifestPath
        }
    }
    $json = $product | ConvertTo-Json -Depth 5 -Compress
    [System.IO.File]::WriteAllText((Join-Path $BundleDir "product-manifest.json"), "$json`n", [System.Text.UTF8Encoding]::new($false))
}

$Version = Normalize-RecipeArgument $Version @("version")
$OutputDir = Normalize-RecipeArgument $OutputDir @("output", "output_dir", "outputdir")
$Flavor = Normalize-RecipeArgument $Flavor @("flavor", "backend")

$releaseFlavor = Get-ReleaseFlavor $Flavor
$binaryFlavor = Get-BinaryFlavor $Flavor
$targetTriple = "x86_64-pc-windows-msvc"
$archiveExt = "zip"
$stableAsset = New-ReleaseAssetName -Prefix "mesh-llm" -TargetTriple $targetTriple -ArchiveExt $archiveExt -BinaryFlavor $releaseFlavor
$versionedAsset = New-ReleaseAssetName -Prefix "mesh-llm-$Version" -TargetTriple $targetTriple -ArchiveExt $archiveExt -BinaryFlavor $releaseFlavor

$meshBinary = Join-Path $releaseBinDir "mesh-llm.exe"
if (-not (Test-Path $meshBinary)) { throw "Required file not found: $meshBinary" }
if (-not (Test-HasValue $nativeRuntimeDir) -or -not (Test-Path (Join-Path $nativeRuntimeDir "manifest.json"))) {
    throw "MESH_LLM_NATIVE_RUNTIME_DIR must name a native runtime directory with manifest.json (got '$nativeRuntimeDir')"
}

$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path (Get-Location) $OutputDir }
New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null

$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("mesh-llm-release-" + [System.Guid]::NewGuid().ToString("N"))
$bundleDir = Join-Path $stagingRoot "mesh-bundle"
New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

try {
    $bundleBinary = Join-Path $bundleDir (Get-BundleBinaryName "mesh-llm" $binaryFlavor)
    Copy-Item $meshBinary -Destination $bundleBinary -Force
    Invoke-ReleaseAttestationStamp -BinaryPath $bundleBinary
    Assert-MeshBinaryVersion -Path $bundleBinary -ExpectedVersion $Version

    $runtimeRoot = Join-Path $bundleDir "native-runtimes"
    $bundledRuntime = Join-Path $runtimeRoot (Split-Path -Leaf ([System.IO.Path]::GetFullPath($nativeRuntimeDir).TrimEnd("\")))
    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    Copy-Item $nativeRuntimeDir -Destination $bundledRuntime -Recurse -Force
    Write-ProductManifest -BundleDir $bundleDir -HostPath $bundleBinary -RuntimeDir $bundledRuntime -ProductVersion $Version.TrimStart("v") -Backend $releaseFlavor

    $versionedPath = Join-Path $resolvedOutputDir $versionedAsset
    $stablePath = Join-Path $resolvedOutputDir $stableAsset
    New-ZipArchive -SourceDir $bundleDir -ArchivePath $versionedPath
    New-ChecksumSidecar -Path $versionedPath
    New-ZipArchive -SourceDir $bundleDir -ArchivePath $stablePath
    New-ChecksumSidecar -Path $stablePath

    Write-Host "Created release archives:"
    Get-ChildItem -Path $resolvedOutputDir -File | Sort-Object Name | ForEach-Object { Write-Host $_.FullName }
} finally {
    if (Test-Path $stagingRoot) {
        Remove-Item $stagingRoot -Recurse -Force
    }
}

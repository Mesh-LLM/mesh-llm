param([string]$Root, [string]$Source, [string]$SourceSha256, [string]$Options)
$configuration = Get-Content -LiteralPath $Options -Raw | ConvertFrom-Json
$Interactive = [bool]$configuration.interactive
$SkipSetup = [bool]$configuration.skip_setup
$LegacyFlavor = [string]$configuration.flavor
$FailRuntime = [bool]$configuration.fail_runtime
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
if ((Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $SourceSha256) { throw 'installer source changed before helper extraction' }
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw 'production installer AST parse failed' }
$names = @('Test-Truthy','Require-WindowsX64','Get-ChecksumUrl','Read-ExpectedSha256','Test-MissingChecksumResponse','Assert-DownloadedFileChecksum','Write-FlavorCompatibilityWarning','Get-StaleBinaryNames','Remove-StaleBinaries','Convert-HexToBytes','Convert-UInt64ToBigEndianBytes','Update-Sha256Bytes','Complete-Sha256','Get-DeterministicTreeSha256','Assert-JsonProperty','Assert-StringField','Assert-Sha256Field','Assert-SafeRelativePath','Assert-ProductBundle','Remove-InstallStagingPath','Move-IfExists','Restore-InstallBackup','Remove-InstallBackups','Stage-IncomingBundle','Install-MeshBinary','Test-InteractiveSession','Format-SetupCommand','Invoke-SetupOrPrint')
$definitions = @($ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false))
foreach ($name in $names) {
    $matches = @($definitions | Where-Object { $_.Name -eq $name })
    if ($matches.Count -ne 1) { throw "missing or duplicate production helper $name" }
    Invoke-Expression $matches[0].Extent.Text
}
$InstallDir = Join-Path $Root 'bin'
$NoSetup = $SkipSetup
$Flavor = $LegacyFlavor
$RequireChecksum = $true
$ComposedProductMinVersion = [System.Version]::Parse('0.75.0')
$env:MESH_LLM_INSTALL_INTERACTIVE = if ($Interactive) { '1' } else { '0' }
Require-WindowsX64
Write-FlavorCompatibilityWarning
# Select the actual production banner statement, not a copied output oracle.
$banners = @($ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.CommandAst] -and $node.GetCommandName() -eq 'Write-Host' -and $node.CommandElements.Count -eq 2 -and $node.CommandElements[1].Extent.Text -eq '"Installing Windows x64 MeshLLM product bundle"' }, $true))
if ($banners.Count -ne 1) { throw 'missing production installation banner' }
Invoke-Expression $banners[0].Extent.Text
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = Join-Path $Root 'mesh-llm-x86_64-pc-windows-msvc.zip'
[System.IO.Compression.ZipFile]::CreateFromDirectory((Join-Path $Root 'prepared'), $archive)
$digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
$sidecar = Join-Path $Root 'expected-sidecar.txt'
[System.IO.File]::WriteAllText($sidecar, "$digest  mesh-llm-x86_64-pc-windows-msvc.zip`n", [System.Text.UTF8Encoding]::new($false))
# Only the checksum HTTP transport is replaced. Any other URI is refused;
# the real installer checksum verification function still executes.
function Invoke-WebRequest {
    param([string]$Uri, [string]$OutFile)
    if ($Uri -ne 'https://offline.invalid/fixture.zip.sha256') { throw 'fixture refuses network request' }
    Copy-Item -LiteralPath $sidecar -Destination $OutFile
}
Assert-DownloadedFileChecksum -Path $archive -Url 'https://offline.invalid/fixture.zip' -RequireSidecar $true
$unpacked = Join-Path $Root 'unpacked'
Expand-Archive -LiteralPath $archive -DestinationPath $unpacked
# The real native admission above ran before enabling this existing fault hook.
if ($FailRuntime) {
    $env:MESH_LLM_INSTALL_TEST_ALLOW_NONWINDOWS = '1'
    $env:MESH_LLM_INSTALL_TEST_FAIL_AFTER_RUNTIME_REPLACE = '1'
}
Install-MeshBinary -BundleDir (Join-Path $unpacked 'mesh-bundle')
$meshBinary = Join-Path $InstallDir 'mesh-llm.exe'
# Require native application dispatch, never Windows file association fallback.
$nativeHost = Get-Command -Name $meshBinary -CommandType Application -ErrorAction Stop
if ($nativeHost.Path -ne $meshBinary) { throw 'fixture native host command path mismatch' }
& $meshBinary --version
if ($LASTEXITCODE -ne 0) { throw "fixture host version failed $LASTEXITCODE" }
Invoke-SetupOrPrint -MeshBinary $meshBinary

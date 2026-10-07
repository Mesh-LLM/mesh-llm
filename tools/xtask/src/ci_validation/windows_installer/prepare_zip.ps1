param([string]$Root, [string]$Source, [string]$SourceSha256)
$ErrorActionPreference = 'Stop'
$fixturePhaseClock = [System.Diagnostics.Stopwatch]::StartNew()
$expectedCore = [System.IO.Path]::Combine($PSHOME, 'Modules')
if (![string]::Equals([System.IO.Path]::GetFullPath($env:MESH_WINDOWS_FIXTURE_CORE_MODULE_PATH), $expectedCore, [StringComparison]::OrdinalIgnoreCase)) { throw 'fixture admitted core module path mismatch' }
# Windows PowerShell inserts AllUsers at startup; restore the closed child path
# before any cmdlet can trigger module autoload. Never accept ambient modules.
$env:PSModulePath = $expectedCore
if (![string]::Equals($env:PSModulePath, $expectedCore, [StringComparison]::OrdinalIgnoreCase)) { throw 'fixture core module path mismatch' }
foreach ($binding in @(@('Get-FileHash', 'Microsoft.PowerShell.Utility'), @('Test-Path', 'Microsoft.PowerShell.Management'))) {
    $command = Get-Command -Name $binding[0] -CommandType Cmdlet
    if ($command.ModuleName -ne $binding[1] -or ![string]::Equals($command.Module.ModuleBase, [System.IO.Path]::Combine($expectedCore, $binding[1]), [StringComparison]::OrdinalIgnoreCase)) { throw "fixture core cmdlet mismatch: $($binding[0])" }
}
[Console]::Error.WriteLine('windows-fixture prepare-core-modules-admitted ms=' + $fixturePhaseClock.ElapsedMilliseconds)
[Console]::Error.WriteLine('windows-fixture prepare-hash-begin ms=' + $fixturePhaseClock.ElapsedMilliseconds)
if ((Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $SourceSha256) { throw 'installer snapshot digest mismatch' }
[Console]::Error.WriteLine('windows-fixture prepare-hash-complete ms=' + $fixturePhaseClock.ElapsedMilliseconds)
# Execute actual native admission before the one rollback case enables the
# existing installer fault hook. No interpreter or architecture emulation.
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw 'installer snapshot parse failure' }
[Console]::Error.WriteLine('windows-fixture prepare-parse-complete ms=' + $fixturePhaseClock.ElapsedMilliseconds)
foreach ($name in @('Test-Truthy', 'Require-WindowsX64')) {
    $found = @($ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false) | Where-Object { $_.Name -eq $name })
    if ($found.Count -ne 1) { throw "unique native admission helper required: $name" }
    Invoke-Expression $found[0].Extent.Text
}
[Console]::Error.WriteLine('windows-fixture prepare-helpers-complete ms=' + $fixturePhaseClock.ElapsedMilliseconds)
Require-WindowsX64
[Console]::Error.WriteLine('windows-fixture prepare-admission-complete ms=' + $fixturePhaseClock.ElapsedMilliseconds)
Add-Type -AssemblyName System.IO.Compression.FileSystem
[Console]::Error.WriteLine('windows-fixture prepare-assembly-complete ms=' + $fixturePhaseClock.ElapsedMilliseconds)
[System.IO.Compression.ZipFile]::CreateFromDirectory((Join-Path $Root 'prepared'), (Join-Path $Root 'fixture.zip'))
[Console]::Error.WriteLine('windows-fixture prepare-zip-complete ms=' + $fixturePhaseClock.ElapsedMilliseconds)

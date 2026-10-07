param([string]$Root, [string]$Source, [string]$SourceSha256)
$ErrorActionPreference = 'Stop'
$fixturePhaseClock = [System.Diagnostics.Stopwatch]::StartNew()
$expectedCore = [System.IO.Path]::Combine($PSHOME, 'Modules')
if (![string]::Equals([System.IO.Path]::GetFullPath($env:MESH_WINDOWS_FIXTURE_CORE_MODULE_PATH), $expectedCore, [StringComparison]::OrdinalIgnoreCase)) { throw 'fixture admitted core module path mismatch' }
# Windows PowerShell inserts AllUsers at startup; restore the closed child path
# before any cmdlet can trigger module autoload. Never accept ambient modules.
$env:PSModulePath = $expectedCore
if (![string]::Equals($env:PSModulePath, $expectedCore, [StringComparison]::OrdinalIgnoreCase)) { throw 'fixture core module path mismatch' }
# Windows PowerShell 5.1 exports Get-FileHash as a Utility script function.
# Binary Management cmdlets are implemented by the DLL in PSHOME; their
# ModuleBase need not be the manifest directory under PSHOME/Modules.
$hashCommand = Get-Command -Name Get-FileHash -CommandType Function -ErrorAction Stop
[Console]::Error.WriteLine('windows-fixture hash-command kind=' + $hashCommand.CommandType + ' module=' + $hashCommand.ModuleName + ' base=' + $hashCommand.Module.ModuleBase)
if ($hashCommand.ModuleName -ne 'Microsoft.PowerShell.Utility' -or ![string]::Equals($hashCommand.Module.ModuleBase, [System.IO.Path]::Combine($expectedCore, 'Microsoft.PowerShell.Utility'), [StringComparison]::OrdinalIgnoreCase)) { throw 'fixture Get-FileHash native function mismatch' }
$pathCommand = Get-Command -Name Test-Path -CommandType Cmdlet -ErrorAction Stop
$managementAssembly = [System.IO.Path]::Combine($PSHOME, 'Microsoft.PowerShell.Commands.Management.dll')
[Console]::Error.WriteLine('windows-fixture path-command kind=' + $pathCommand.CommandType + ' module=' + $pathCommand.ModuleName + ' assembly=' + $pathCommand.ImplementingType.Assembly.Location)
if ($pathCommand.ModuleName -ne 'Microsoft.PowerShell.Management' -or ![string]::Equals($pathCommand.ImplementingType.Assembly.Location, $managementAssembly, [StringComparison]::OrdinalIgnoreCase)) { throw 'fixture Test-Path native assembly mismatch' }
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

param([string]$Root, [string]$Source, [string]$SourceSha256)
$ErrorActionPreference = 'Stop'
$fixturePhaseClock = [System.Diagnostics.Stopwatch]::StartNew()
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

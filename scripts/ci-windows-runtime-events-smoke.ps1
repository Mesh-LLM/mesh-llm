#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Windows composed-product smoke for the runtime-event boundary
    (`GET /api/runtime/events/v1`, `capabilities.runtime_events` on both
    status routes, SSE connection-shape framing, and clean shutdown).

.DESCRIPTION
    Launches the composed `mesh-llm.exe` product in hermetic, noninteractive
    client mode (no model download -- mirrors the existing
    scripts/ci-client-readiness-smoke.sh contract, but with the management
    console ENABLED since the routes under test live there), then:

      1. Asserts `capabilities.runtime_events` is advertised identically on
         both `/api/status` (nested under `.runtime`) and `/api/runtime`.
      2. Opens a raw SSE connection with no cursor and asserts the frozen
         `no_cursor` initial frame order (`runtime_state`, `runtime_health`).
      3. Opens a raw SSE connection with a cursor for a foreign process
         instance and asserts the frozen `gap` (`stale_instance`) initial
         frame order (`runtime_replay_gap`, `runtime_state`,
         `runtime_health`).
      4. Opens a raw SSE connection with a malformed cursor and asserts an
         HTTP 400 response with no SSE headers ever written (the frozen
         `rejected` shape).
      5. Validates every captured SSE frame against the shared
         `runtime_events_v1` fixture contract (required envelope keys,
         per-event required keys, and the `id: <cursor>\nevent: <name>\n
         data: <json>\n\n` frame grammar).
      6. Sends a graceful CTRL_BREAK shutdown and proves the process exits
         and stays exited.

    Every step's outcome is recorded into a JUnit report plus the launch
    JSON log, captured HTTP headers, captured raw SSE frames, product/runtime
    manifest and checksum copies (best-effort -- their absence is recorded,
    not fatal, since compose-product-input already verifies them upstream),
    a residual-risk statement, and cleanup proof, all under -EvidenceDir.

    Requires no model download. $ErrorActionPreference is 'Stop' and any
    failed assertion produces a non-zero exit so the owning CI job can FAIL.

.PARAMETER ProductDir
    Root of the extracted composed Windows product (contains mesh-llm.exe
    and its own bundled native-runtimes tree; see compose-product-input).

.PARAMETER RuntimeDir
    Root of the extracted raw Windows CPU native-runtime artifact
    (ci-runtime-windows-<architecture>-<backend>). Used only for manifest and
    checksum evidence provenance -- the composed product under -ProductDir
    is self-contained and is what actually gets launched.

.PARAMETER FixtureDir
    crates/mesh-llm-runtime-event-contracts/fixtures/runtime_events_v1 --
    the shared Rust/TypeScript wire-contract fixtures (frames.json,
    cursors.json, recovery.json).

.PARAMETER EvidenceDir
    Directory that receives every artifact this script produces.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProductDir,
    [Parameter(Mandatory = $true)][string]$RuntimeDir,
    [Parameter(Mandatory = $true)][string]$FixtureDir,
    [Parameter(Mandatory = $true)][string]$EvidenceDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# ── Evidence + result plumbing ──────────────────────────────────────────

$script:TestResults = [System.Collections.Generic.List[object]]::new()

function Add-TestResult {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$Passed,
        [string]$Detail = ''
    )
    $script:TestResults.Add([PSCustomObject]@{
        Name    = $Name
        Passed  = $Passed
        Detail  = $Detail
    })
    $status = if ($Passed) { 'PASS' } else { 'FAIL' }
    Write-Host "[$status] $Name"
    if (-not [string]::IsNullOrEmpty($Detail)) {
        Write-Host "        $Detail"
    }
}

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$Condition,
        [string]$Detail = ''
    )
    Add-TestResult -Name $Name -Passed $Condition -Detail $Detail
    if (-not $Condition) {
        throw "assertion failed: $Name ($Detail)"
    }
}

function Write-JUnitReport {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )
    $total = $script:TestResults.Count
    $failed = ($script:TestResults | Where-Object { -not $_.Passed }).Count
    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine('<?xml version="1.0" encoding="UTF-8"?>')
    [void]$sb.AppendLine("<testsuite name=`"ci-windows-runtime-events-smoke`" tests=`"$total`" failures=`"$failed`">")
    foreach ($result in $script:TestResults) {
        $escapedName = [System.Security.SecurityElement]::Escape($result.Name)
        if ($result.Passed) {
            [void]$sb.AppendLine("  <testcase classname=`"runtime_events_smoke`" name=`"$escapedName`" />")
        }
        else {
            $escapedDetail = [System.Security.SecurityElement]::Escape($result.Detail)
            [void]$sb.AppendLine("  <testcase classname=`"runtime_events_smoke`" name=`"$escapedName`">")
            [void]$sb.AppendLine("    <failure message=`"$escapedDetail`" />")
            [void]$sb.AppendLine('  </testcase>')
        }
    }
    [void]$sb.AppendLine('</testsuite>')
    Set-Content -Path $Path -Value $sb.ToString() -Encoding UTF8
}

# ── Setup ────────────────────────────────────────────────────────────────

foreach ($dir in @($ProductDir, $RuntimeDir, $FixtureDir)) {
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) {
        throw "required directory does not exist: $dir"
    }
}
New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null
$manifestsDir = Join-Path $EvidenceDir 'manifests'
$framesDir = Join-Path $EvidenceDir 'sse-frames'
$headersDir = Join-Path $EvidenceDir 'http-headers'
New-Item -ItemType Directory -Force -Path $manifestsDir | Out-Null
New-Item -ItemType Directory -Force -Path $framesDir | Out-Null
New-Item -ItemType Directory -Force -Path $headersDir | Out-Null

$meshLlmExe = Get-ChildItem -LiteralPath $ProductDir -Recurse -Filter 'mesh-llm.exe' -File |
    Select-Object -First 1
if ($null -eq $meshLlmExe) {
    throw "mesh-llm.exe not found under $ProductDir"
}
Write-Host "Using product executable: $($meshLlmExe.FullName)"

# Best-effort manifest/checksum provenance copy. Absence is recorded, not
# fatal -- compose-product-input and the runtime producer already verify
# these upstream; this is supplementary evidence for THIS run.
function Copy-Provenance {
    param(
        [Parameter(Mandatory = $true)][string]$SourceDir,
        [Parameter(Mandatory = $true)][string]$DestSubdir,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $dest = Join-Path $manifestsDir $DestSubdir
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $found = Get-ChildItem -LiteralPath $SourceDir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '(?i)manifest.*\.json$|\.sha256$' }
    if ($null -eq $found -or @($found).Count -eq 0) {
        Set-Content -Path (Join-Path $dest 'NOT_FOUND.txt') `
            -Value "No manifest/checksum files matched under $SourceDir for $Label."
        return
    }
    foreach ($file in $found) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $dest $file.Name) -Force
    }
}
Copy-Provenance -SourceDir $ProductDir -DestSubdir 'product' -Label 'composed product'
Copy-Provenance -SourceDir $RuntimeDir -DestSubdir 'runtime' -Label 'raw Windows CPU runtime'

# ── Fixture contract ─────────────────────────────────────────────────────

$framesFixture = Get-Content -LiteralPath (Join-Path $FixtureDir 'frames.json') -Raw | ConvertFrom-Json
$cursorsFixture = Get-Content -LiteralPath (Join-Path $FixtureDir 'cursors.json') -Raw | ConvertFrom-Json
$recoveryFixture = Get-Content -LiteralPath (Join-Path $FixtureDir 'recovery.json') -Raw | ConvertFrom-Json

Assert-True -Name 'fixtures.frames.version_is_1' -Condition ($framesFixture.version -eq 1) `
    -Detail "frames.json version was $($framesFixture.version)"

$noCursorShape = $recoveryFixture.connection_shapes | Where-Object { $_.shape -eq 'no_cursor' }
$gapShape = $recoveryFixture.connection_shapes | Where-Object { $_.shape -eq 'gap' }
$rejectedShape = $recoveryFixture.connection_shapes | Where-Object { $_.shape -eq 'rejected' }
Assert-True -Name 'fixtures.recovery.has_required_shapes' `
    -Condition ($null -ne $noCursorShape -and $null -ne $gapShape -and $null -ne $rejectedShape) `
    -Detail 'recovery.json is missing no_cursor, gap, or rejected shapes'

$invalidCursor = $cursorsFixture.invalid | Where-Object { $_ -eq 'rt1:not-a-uuid:1' }
Assert-True -Name 'fixtures.cursors.has_expected_invalid_sample' -Condition ($null -ne $invalidCursor) `
    -Detail "cursors.json invalid list did not contain the expected malformed sample"

# A cursor grammar-valid UUID that will not match the freshly-started
# engine's randomly generated process instance -- exercises stale_instance.
$foreignCursor = 'rt1:0195f000-0000-7000-8000-000000000001:0'

# ── Launch the composed product (no model download) ────────────────────

function Get-FreeTcpPort {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        return $listener.LocalEndpoint.Port
    }
    finally {
        $listener.Stop()
    }
}

$apiPort = Get-FreeTcpPort
$consolePort = Get-FreeTcpPort
while ($consolePort -eq $apiPort) { $consolePort = Get-FreeTcpPort }

$stateDir = Join-Path ([System.IO.Path]::GetTempPath()) ("mlre-state-" + [Guid]::NewGuid().ToString('N'))
foreach ($sub in @('home', 'cache', 'config', 'runtime', 'runtime-cache')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $stateDir $sub) | Out-Null
}
$launchLog = Join-Path $EvidenceDir 'launch.jsonl.log'
$pidFile = Join-Path $stateDir 'native-client.pid'
$processHelper = Join-Path $PSScriptRoot 'ci-client-readiness-process.py'
if (-not (Test-Path -LiteralPath $processHelper -PathType Leaf)) {
    throw "missing Windows process-group helper: $processHelper"
}

$env:HOME = Join-Path $stateDir 'home'
$env:XDG_CACHE_HOME = Join-Path $stateDir 'cache'
$env:XDG_CONFIG_HOME = Join-Path $stateDir 'config'
$env:MESH_LLM_CONFIG = Join-Path $stateDir 'config.toml'
$env:MESH_LLM_RUNTIME_ROOT = Join-Path $stateDir 'runtime'
$env:MESH_LLM_NATIVE_RUNTIME_CACHE_DIR = Join-Path $stateDir 'runtime-cache'

$cleanupOk = $true
$nativePid = $null
try {
    # ── 0. launch + readiness wait (inside try so a startup failure still
    #      produces JUnit/evidence output instead of a bare crash) ──────

    $launchArgs = @(
        $processHelper, 'run',
        '--pid-file', $pidFile,
        '--log', $launchLog,
        '--',
        $meshLlmExe.FullName,
        '--log-format', 'json',
        '--port', "$apiPort",
        '--console', "$consolePort",
        'client', '--mesh-discovery-mode', 'mdns'
    )
    Write-Host "Launching: python3 $($launchArgs -join ' ')"
    $launcherProcess = Start-Process -FilePath 'python3' -ArgumentList $launchArgs -PassThru -WindowStyle Hidden

    $maxWaitSeconds = 60
    $ready = $false
    for ($i = 0; $i -lt $maxWaitSeconds; $i++) {
        if ($launcherProcess.HasExited -and -not (Test-Path -LiteralPath $pidFile)) {
            throw "launcher exited before writing a pid file (exit code $($launcherProcess.ExitCode))"
        }
        if ((Test-Path -LiteralPath $pidFile) -and (Test-Path -LiteralPath $launchLog)) {
            $nativePid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
            $logLines = Get-Content -LiteralPath $launchLog -ErrorAction SilentlyContinue
            foreach ($line in $logLines) {
                try {
                    $event = $line | ConvertFrom-Json -ErrorAction Stop
                }
                catch {
                    continue
                }
                $message = [string]$event.message
                $structuredReady = ($event.event -eq 'passive_mode') -and
                    ($event.status -eq 'ready') -and
                    ($event.role -eq 'client')
                if ($structuredReady -or ($message -match '(?i)client ready')) {
                    $ready = $true
                    break
                }
            }
        }
        if ($ready) { break }
        Start-Sleep -Seconds 1
    }

    Add-TestResult -Name 'launch.client_ready' -Passed $ready `
        -Detail "pid=$nativePid api_port=$apiPort console_port=$consolePort"
    if (-not $ready) {
        if (Test-Path -LiteralPath $launchLog) { Get-Content -LiteralPath $launchLog | Write-Host }
        throw "timed out waiting for hermetic structured client readiness on console port $consolePort"
    }
    Write-Host "Client readiness observed (pid=$nativePid, api port=$apiPort, console port=$consolePort)"

    # ── 1. capabilities.runtime_events on both status routes ──────────

    $expectedCapability = [PSCustomObject]@{
        version  = 1
        endpoint = '/api/runtime/events/v1'
        cursor   = 'rt1'
    }

    function Test-CapabilityObject {
        param($Capability, [string]$RouteName)
        return ($null -ne $Capability) -and
            ($Capability.version -eq $expectedCapability.version) -and
            ($Capability.endpoint -eq $expectedCapability.endpoint) -and
            ($Capability.cursor -eq $expectedCapability.cursor)
    }

    $statusResponse = Invoke-WebRequest -Uri "http://127.0.0.1:$consolePort/api/status" -TimeoutSec 15 -UseBasicParsing
    $statusBody = $statusResponse.Content | ConvertFrom-Json
    Set-Content -Path (Join-Path $headersDir 'status.headers.txt') -Value ($statusResponse.RawContent -split "`r`n`r`n")[0]
    Assert-True -Name 'status_route.capabilities.runtime_events' `
        -Condition (Test-CapabilityObject -Capability $statusBody.runtime.capabilities.runtime_events -RouteName '/api/status') `
        -Detail "got $(($statusBody.runtime.capabilities.runtime_events) | ConvertTo-Json -Compress)"

    $runtimeResponse = Invoke-WebRequest -Uri "http://127.0.0.1:$consolePort/api/runtime" -TimeoutSec 15 -UseBasicParsing
    $runtimeBody = $runtimeResponse.Content | ConvertFrom-Json
    Set-Content -Path (Join-Path $headersDir 'runtime.headers.txt') -Value ($runtimeResponse.RawContent -split "`r`n`r`n")[0]
    Assert-True -Name 'runtime_route.capabilities.runtime_events' `
        -Condition (Test-CapabilityObject -Capability $runtimeBody.capabilities.runtime_events -RouteName '/api/runtime') `
        -Detail "got $(($runtimeBody.capabilities.runtime_events) | ConvertTo-Json -Compress)"

    # ── SSE raw-socket helper ──────────────────────────────────────────

    function Read-RuntimeEventsSse {
        param(
            [Parameter(Mandatory = $true)][int]$Port,
            [hashtable]$ExtraHeaders = @{},
            [int]$ExpectedFrameCount = 0,
            [int]$ReadTimeoutMs = 10000
        )
        $client = [System.Net.Sockets.TcpClient]::new()
        $client.Connect('127.0.0.1', $Port)
        $stream = $client.GetStream()
        $stream.ReadTimeout = $ReadTimeoutMs

        $headerText = "Host: localhost`r`nAccept: text/event-stream`r`n"
        foreach ($key in $ExtraHeaders.Keys) {
            $headerText += "${key}: $($ExtraHeaders[$key])`r`n"
        }
        $request = "GET /api/runtime/events/v1 HTTP/1.1`r`n$headerText`r`n"
        $requestBytes = [System.Text.Encoding]::ASCII.GetBytes($request)
        $stream.Write($requestBytes, 0, $requestBytes.Length)

        $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::ASCII)
        $statusLine = $reader.ReadLine()
        $headers = [ordered]@{}
        while ($true) {
            $line = $reader.ReadLine()
            if ([string]::IsNullOrEmpty($line)) { break }
            $splitIndex = $line.IndexOf(':')
            if ($splitIndex -gt 0) {
                $headers[$line.Substring(0, $splitIndex).Trim()] = $line.Substring($splitIndex + 1).Trim()
            }
        }

        $frames = [System.Collections.Generic.List[object]]::new()
        $body = ''
        if ($statusLine -match ' 200 ') {
            for ($i = 0; $i -lt $ExpectedFrameCount; $i++) {
                $frameLines = [System.Collections.Generic.List[string]]::new()
                while ($true) {
                    $line = $reader.ReadLine()
                    if ($null -eq $line) { break }
                    if ($line -eq '') { break }
                    $frameLines.Add($line)
                }
                if ($frameLines.Count -eq 0) { break }
                $frames.Add([string[]]$frameLines.ToArray())
            }
        }
        elseif ($headers.Contains('Content-Length')) {
            $bodyLen = [int]$headers['Content-Length']
            $buffer = New-Object char[] $bodyLen
            $totalRead = 0
            while ($totalRead -lt $bodyLen) {
                $read = $reader.Read($buffer, $totalRead, $bodyLen - $totalRead)
                if ($read -le 0) { break }
                $totalRead += $read
            }
            $body = -join $buffer
        }

        $reader.Dispose()
        $stream.Dispose()
        $client.Dispose()

        return [PSCustomObject]@{
            StatusLine = $statusLine
            Headers    = $headers
            Frames     = $frames
            Body       = $body
        }
    }

    function Get-FrameEventName {
        param([string[]]$FrameLines)
        foreach ($line in $FrameLines) {
            if ($line.StartsWith('event:')) {
                return $line.Substring(6).Trim()
            }
        }
        return $null
    }

    function Get-FrameData {
        param([string[]]$FrameLines)
        foreach ($line in $FrameLines) {
            if ($line.StartsWith('data:')) {
                return $line.Substring(5).Trim() | ConvertFrom-Json
            }
        }
        return $null
    }

    function Save-Frames {
        param([string]$Name, $Result)
        $path = Join-Path $framesDir "$Name.txt"
        $lines = @("STATUS: $($Result.StatusLine)")
        foreach ($h in $Result.Headers.Keys) { $lines += "HEADER: ${h}: $($Result.Headers[$h])" }
        if (-not [string]::IsNullOrEmpty($Result.Body)) { $lines += "BODY: $($Result.Body)" }
        $frameIndex = 0
        foreach ($frame in $Result.Frames) {
            $lines += "--- frame $frameIndex ---"
            $lines += $frame
            $frameIndex++
        }
        Set-Content -Path $path -Value $lines
    }

    function Test-HasProperty {
        param($Object, [string]$PropertyName)
        if ($null -eq $Object) { return $false }
        return $null -ne (Get-Member -InputObject $Object -Name $PropertyName -ErrorAction SilentlyContinue)
    }

    function Assert-EnvelopeKeys {
        param([string]$FrameLabel, $Data, [string]$EventName)
        Assert-True -Name "$FrameLabel.data_line_present" -Condition ($null -ne $Data) `
            -Detail "$EventName frame had no parseable data: line"
        $required = @($framesFixture.required_envelope_keys)
        foreach ($key in $required) {
            Assert-True -Name "$FrameLabel.envelope_has_$key" -Condition (Test-HasProperty -Object $Data -PropertyName $key) `
                -Detail "missing required envelope key '$key' on $EventName frame"
        }
        $perEvent = $framesFixture.per_event_required_keys.$EventName
        if ($null -ne $perEvent) {
            foreach ($key in @($perEvent)) {
                Assert-True -Name "$FrameLabel.$EventName.has_$key" -Condition (Test-HasProperty -Object $Data -PropertyName $key) `
                    -Detail "missing per-event required key '$key' on $EventName frame"
            }
        }
    }

    # ── 2. no_cursor initial frame order ────────────────────────────────

    $noCursorExpected = @($noCursorShape.frame_order)
    $noCursorResult = Read-RuntimeEventsSse -Port $apiPort -ExpectedFrameCount $noCursorExpected.Count
    Save-Frames -Name 'no_cursor' -Result $noCursorResult
    Assert-True -Name 'no_cursor.status_is_200' -Condition ($noCursorResult.StatusLine -match ' 200 ') `
        -Detail "status line was: $($noCursorResult.StatusLine)"
    Assert-True -Name 'no_cursor.content_type_is_event_stream' `
        -Condition ($noCursorResult.Headers['Content-Type'] -eq 'text/event-stream') `
        -Detail "Content-Type was: $($noCursorResult.Headers['Content-Type'])"
    $noCursorNames = @($noCursorResult.Frames | ForEach-Object { Get-FrameEventName -FrameLines $_ })
    Assert-True -Name 'no_cursor.frame_order' -Condition (($noCursorNames -join ',') -eq ($noCursorExpected -join ',')) `
        -Detail "expected [$($noCursorExpected -join ', ')] got [$($noCursorNames -join ', ')]"
    for ($i = 0; $i -lt $noCursorResult.Frames.Count; $i++) {
        $name = $noCursorNames[$i]
        $data = Get-FrameData -FrameLines $noCursorResult.Frames[$i]
        Assert-EnvelopeKeys -FrameLabel "no_cursor.frame$i" -Data $data -EventName $name
    }

    # ── 3. gap (stale_instance) initial frame order ─────────────────────

    $gapExpectedRaw = @($gapShape.frame_order)
    # recovery.json spells the replay wildcard as "runtime_event*"; the
    # stale_instance gap never replays events (there is nothing to replay
    # for a process instance the engine has never seen), so this shape's
    # concrete order is exactly runtime_replay_gap, runtime_state,
    # runtime_health with zero runtime_event frames.
    $gapExpected = $gapExpectedRaw | Where-Object { $_ -ne 'runtime_event*' }
    $gapResult = Read-RuntimeEventsSse -Port $apiPort `
        -ExtraHeaders @{ 'Last-Event-ID' = $foreignCursor } `
        -ExpectedFrameCount $gapExpected.Count
    Save-Frames -Name 'gap_stale_instance' -Result $gapResult
    Assert-True -Name 'gap.status_is_200' -Condition ($gapResult.StatusLine -match ' 200 ') `
        -Detail "status line was: $($gapResult.StatusLine)"
    $gapNames = @($gapResult.Frames | ForEach-Object { Get-FrameEventName -FrameLines $_ })
    Assert-True -Name 'gap.frame_order' -Condition (($gapNames -join ',') -eq ($gapExpected -join ',')) `
        -Detail "expected [$($gapExpected -join ', ')] got [$($gapNames -join ', ')]"
    if ($gapNames.Count -gt 0 -and $gapNames[0] -eq 'runtime_replay_gap') {
        $gapData = Get-FrameData -FrameLines $gapResult.Frames[0]
        Assert-EnvelopeKeys -FrameLabel 'gap.frame0' -Data $gapData -EventName 'runtime_replay_gap'
        Assert-True -Name 'gap.reason_is_stale_instance' -Condition ($gapData.reason -eq 'stale_instance') `
            -Detail "runtime_replay_gap.reason was: $($gapData.reason)"
    }

    # ── 4. rejected (malformed cursor) ──────────────────────────────────

    $rejectedResult = Read-RuntimeEventsSse -Port $apiPort `
        -ExtraHeaders @{ 'Last-Event-ID' = 'rt1:not-a-uuid:1' } `
        -ExpectedFrameCount 0
    Save-Frames -Name 'rejected_malformed_cursor' -Result $rejectedResult
    Assert-True -Name 'rejected.status_is_400' -Condition ($rejectedResult.StatusLine -match ' 400 ') `
        -Detail "status line was: $($rejectedResult.StatusLine)"
    Assert-True -Name 'rejected.no_sse_headers_written' `
        -Condition ($rejectedResult.Headers['Content-Type'] -ne 'text/event-stream') `
        -Detail "Content-Type was: $($rejectedResult.Headers['Content-Type'])"

    # ── 5. keepalive/frame grammar note (documented, not waited on) ────

    Assert-True -Name 'fixtures.frame_grammar_documented' `
        -Condition (-not [string]::IsNullOrEmpty($framesFixture.frame_grammar)) `
        -Detail 'frames.json is missing frame_grammar'
}
catch {
    $cleanupOk = $false
    Add-TestResult -Name 'unhandled_exception' -Passed $false -Detail $_.Exception.Message
    Write-Host "SMOKE FAILED: $($_.Exception.Message)"
}
finally {
    # ── 6. shutdown + cleanup proof ─────────────────────────────────────

    function Test-NativeProcessRunning {
        param([Parameter(Mandatory = $true)][string]$TargetPid)
        & python3 $processHelper is-running --pid $TargetPid 2>&1 | Out-Null
        return $LASTEXITCODE -eq 0
    }

    $shutdownLines = [System.Collections.Generic.List[string]]::new()
    if ($null -ne $nativePid -and $nativePid -ne '') {
        $shutdownLines.Add("Sending CTRL_BREAK to pid $nativePid")
        & python3 $processHelper ctrl-break --pid $nativePid 2>&1 | Out-Null

        $stopped = $false
        for ($i = 0; $i -lt 15; $i++) {
            if (-not (Test-NativeProcessRunning -TargetPid $nativePid)) { $stopped = $true; break }
            Start-Sleep -Seconds 1
        }
        if (-not $stopped) {
            $shutdownLines.Add("did not exit within 15s after CTRL_BREAK; force-stopping")
            & python3 $processHelper force-stop --pid $nativePid 2>&1 | Out-Null
            Start-Sleep -Seconds 1
        }
        $stillRunning = Test-NativeProcessRunning -TargetPid $nativePid
        $shutdownLines.Add("post-shutdown is-running=$stillRunning")
        Assert-True -Name 'shutdown.process_exited' -Condition (-not $stillRunning) `
            -Detail "pid $nativePid was still running after shutdown + force-stop"
    }
    else {
        $shutdownLines.Add('no native pid was recorded; nothing to shut down')
    }
    Set-Content -Path (Join-Path $EvidenceDir 'cleanup-proof.txt') -Value $shutdownLines

    Remove-Item -LiteralPath $stateDir -Recurse -Force -ErrorAction SilentlyContinue

    $residualRisk = @(
        '# Residual risk',
        '',
        'No real native model-open callback executes on Windows in this smoke:',
        'the composed product is launched in client mode with no model download,',
        'so this coverage proves the runtime-event HTTP/SSE wire boundary',
        '(route, capability discovery, connection-shape framing, shutdown)',
        'against a real Windows binary, not a real native callback firing from',
        'an actively loaded model.',
        '',
        'The static non-unix build path has no runtime events by construction:',
        'only the dynamic-native-runtime path exercised by this composed',
        'product carries the runtime-event engine wiring, so this smoke does',
        'not (and cannot) prove behavior for a statically linked Windows build.'
    )
    Set-Content -Path (Join-Path $EvidenceDir 'residual-risk.md') -Value $residualRisk

    Write-JUnitReport -Path (Join-Path $EvidenceDir 'junit.xml')

    $summaryLines = $script:TestResults | ForEach-Object {
        $status = if ($_.Passed) { 'PASS' } else { 'FAIL' }
        "$status`t$($_.Name)`t$($_.Detail)"
    }
    Set-Content -Path (Join-Path $EvidenceDir 'test-summary.txt') -Value $summaryLines

    if (Test-Path -LiteralPath $launchLog) {
        Copy-Item -LiteralPath $launchLog -Destination (Join-Path $EvidenceDir 'launch.jsonl.log') -Force -ErrorAction SilentlyContinue
    }
}

$failureCount = ($script:TestResults | Where-Object { -not $_.Passed }).Count
if ($failureCount -gt 0 -or -not $cleanupOk) {
    Write-Host "FAILED: $failureCount assertion(s) failed"
    exit 1
}
Write-Host "PASSED: $($script:TestResults.Count) assertion(s)"
exit 0

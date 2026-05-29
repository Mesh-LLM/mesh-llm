param(
    [string[]]$ConsoleUrls = @("http://127.0.0.1:3131"),
    [string[]]$ApiUrls = @("http://127.0.0.1:9337/v1"),
    [string]$Model = "auto",
    [string]$OutputDir = $env:TEMP,
    [string]$MeshLlm = "",
    [switch]$RunProbe,
    [switch]$SkipHttp,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
CollectSplitDiagnostics.ps1 - capture split-readiness diagnostics for maintainers.

Usage:
  .\contrib\windows\CollectSplitDiagnostics.ps1 -Model meshllm/Qwen3-8B-Q4_K_M-layers

Options:
  -ConsoleUrls  Management API URLs, default http://127.0.0.1:3131
  -ApiUrls      OpenAI API base URLs, default http://127.0.0.1:9337/v1
  -Model        Model id/ref for split doctor and optional chat probe
  -OutputDir    Parent output directory, default TEMP
  -MeshLlm      mesh-llm.exe path, default target\release\mesh-llm.exe or PATH
  -RunProbe     Run a tiny /chat/completions probe through each API URL
  -SkipHttp     Capture local system/process facts only
  -Help         Print this help
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

function Resolve-MeshBinary {
    param([string]$Preferred)

    if ($Preferred -and (Test-Path -LiteralPath $Preferred)) {
        return (Resolve-Path -LiteralPath $Preferred).Path
    }

    $ReleaseBinary = Join-Path (Get-Location) "target\release\mesh-llm.exe"
    if (Test-Path -LiteralPath $ReleaseBinary) {
        return (Resolve-Path -LiteralPath $ReleaseBinary).Path
    }

    $Command = Get-Command "mesh-llm" -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }

    return $null
}

function New-DiagnosticDirectory {
    param([string]$Parent)

    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Base = Join-Path $Parent "mesh-split-diagnostics-$Stamp"
    New-Item -ItemType Directory -Force -Path $Base | Out-Null
    return $Base
}

function Write-TextFile {
    param(
        [string]$Path,
        [string]$Content
    )

    Set-Content -LiteralPath $Path -Value $Content -Encoding UTF8
}

function Invoke-CaptureCommand {
    param(
        [string]$Path,
        [string]$FileName,
        [string]$Command,
        [string[]]$Arguments
    )

    $Target = Join-Path $Path $FileName
    try {
        $Output = & $Command @Arguments 2>&1 | Out-String
        Write-TextFile -Path $Target -Content $Output
    } catch {
        Write-TextFile -Path $Target -Content "command failed: $($_.Exception.Message)"
    }
}

function Redact-Text {
    param([string]$Text)

    if (-not $Text) {
        return $Text
    }

    $Redacted = $Text
    $Redacted = [regex]::Replace($Redacted, '("token"\s*:\s*")[^"]+(")', '$1<redacted>$2')
    $Redacted = [regex]::Replace($Redacted, '(Authorization:\s*Bearer\s+)[^\s"]+', '$1<redacted>', 'IgnoreCase')
    $Redacted = [regex]::Replace($Redacted, '([A-Za-z0-9_]*(?:TOKEN|KEY)\s*=\s*)[^\s"]+', '$1<redacted>', 'IgnoreCase')
    $Redacted = [regex]::Replace($Redacted, '("--join"\s*,\s*")[^"]+(")', '$1<redacted>$2')
    return $Redacted
}

function Invoke-HttpCapture {
    param(
        [string]$Path,
        [string]$Name,
        [string]$Url
    )

    $RedactedPath = Join-Path $Path "$Name.json"
    try {
        $Response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 10 -Uri $Url
        $Content = [string]$Response.Content
        Write-TextFile -Path $RedactedPath -Content (Redact-Text $Content)
    } catch {
        Write-TextFile -Path $RedactedPath -Content "request failed: $($_.Exception.Message)"
    }
}

function Invoke-ChatProbe {
    param(
        [string]$Path,
        [string]$Name,
        [string]$ApiBase,
        [string]$Model
    )

    $ProbePath = Join-Path $Path "$Name.chat-probe.json"
    $Body = @{
        model = $Model
        messages = @(@{ role = "user"; content = "Reply with mesh split diagnostic probe." })
        max_tokens = 16
        stream = $false
    } | ConvertTo-Json -Depth 8

    try {
        $Url = ($ApiBase.TrimEnd('/')) + "/chat/completions"
        $Response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 30 -Method Post -Uri $Url -ContentType "application/json" -Body $Body
        Write-TextFile -Path $ProbePath -Content (Redact-Text ([string]$Response.Content))
    } catch {
        Write-TextFile -Path $ProbePath -Content "probe failed: $($_.Exception.Message)"
    }
}

function Copy-RuntimeLogTails {
    param([string]$Path)

    $RuntimeRoot = Join-Path $HOME ".mesh-llm\runtime"
    if (-not (Test-Path -LiteralPath $RuntimeRoot)) {
        return
    }

    $LogsDir = Join-Path $Path "logs"
    New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
    Get-ChildItem -LiteralPath $RuntimeRoot -Recurse -Filter "skippy-native.log" -ErrorAction SilentlyContinue |
        ForEach-Object {
            $SafeName = ($_.FullName -replace '[\\/:*?"<>| ]', '_')
            $Target = Join-Path $LogsDir $SafeName
            try {
                $Tail = Get-Content -LiteralPath $_.FullName -Tail 400 -ErrorAction Stop | Out-String
                Write-TextFile -Path $Target -Content $Tail
            } catch {
                Write-TextFile -Path $Target -Content "log read failed: $($_.Exception.Message)"
            }
        }
}

$MeshBinary = Resolve-MeshBinary -Preferred $MeshLlm
$CaptureDir = New-DiagnosticDirectory -Parent $OutputDir

$Manifest = [ordered]@{
    created_at = (Get-Date).ToString("o")
    model = $Model
    console_urls = $ConsoleUrls
    api_urls = $ApiUrls
    mesh_llm = $MeshBinary
    run_probe = [bool]$RunProbe
    skip_http = [bool]$SkipHttp
    os = [System.Environment]::OSVersion.VersionString
    powershell = $PSVersionTable.PSVersion.ToString()
}
Write-TextFile -Path (Join-Path $CaptureDir "manifest.json") -Content ($Manifest | ConvertTo-Json -Depth 6)

if ($MeshBinary) {
    Invoke-CaptureCommand -Path $CaptureDir -FileName "mesh-llm.version.txt" -Command $MeshBinary -Arguments @("--version")
    Invoke-CaptureCommand -Path $CaptureDir -FileName "mesh-llm.gpus.json" -Command $MeshBinary -Arguments @("gpus", "--json")
} else {
    Write-TextFile -Path (Join-Path $CaptureDir "mesh-llm.version.txt") -Content "mesh-llm binary not found"
}

try {
    Get-CimInstance Win32_OperatingSystem | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $CaptureDir "windows.os.json") -Encoding UTF8
    Get-CimInstance Win32_VideoController | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $CaptureDir "windows.gpus.json") -Encoding UTF8
} catch {
    Write-TextFile -Path (Join-Path $CaptureDir "windows.cim-error.txt") -Content $_.Exception.Message
}

Get-Process | Select-Object Id, ProcessName, Path | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $CaptureDir "processes.json") -Encoding UTF8

Invoke-CaptureCommand -Path $CaptureDir -FileName "netstat.txt" -Command "netstat" -Arguments @("-ano")

foreach ($OptionalCommand in @("nvidia-smi", "vulkaninfo", "rocminfo")) {
    $Command = Get-Command $OptionalCommand -ErrorAction SilentlyContinue
    if ($Command) {
        Invoke-CaptureCommand -Path $CaptureDir -FileName "$OptionalCommand.txt" -Command $Command.Source -Arguments @()
    }
}

if (-not $SkipHttp) {
    $Index = 0
    foreach ($ConsoleUrl in $ConsoleUrls) {
        $Name = "console-$Index"
        $Base = $ConsoleUrl.TrimEnd('/')
        Invoke-HttpCapture -Path $CaptureDir -Name "$Name.status" -Url "$Base/api/status"
        Invoke-HttpCapture -Path $CaptureDir -Name "$Name.runtime-stages" -Url "$Base/api/runtime/stages"
        if ($Model -and $Model -ne "auto") {
            $Encoded = [System.Uri]::EscapeDataString($Model)
            Invoke-HttpCapture -Path $CaptureDir -Name "$Name.split-readiness" -Url "$Base/api/diagnostics/split-readiness?model_ref=$Encoded"
        }
        $Index += 1
    }

    $Index = 0
    foreach ($ApiUrl in $ApiUrls) {
        $Name = "api-$Index"
        $Base = $ApiUrl.TrimEnd('/')
        Invoke-HttpCapture -Path $CaptureDir -Name "$Name.models" -Url "$Base/models"
        if ($RunProbe) {
            Invoke-ChatProbe -Path $CaptureDir -Name $Name -ApiBase $Base -Model $Model
        }
        $Index += 1
    }
}

Copy-RuntimeLogTails -Path $CaptureDir

$ZipPath = "$CaptureDir.zip"
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -Path (Join-Path $CaptureDir "*") -DestinationPath $ZipPath -Force

Write-Host "Split diagnostics written to:"
Write-Host "  $CaptureDir"
Write-Host "  $ZipPath"

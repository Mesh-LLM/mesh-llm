[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][ValidateSet('start','stop','monitor','status')][string]$Action,
  [Parameter(Mandatory=$true)][string]$NodeName,
  [string]$Token,
  [string]$BindIp,
  [int]$ConsolePort,
  [int]$ApiPort,
  [int]$QuicPort,
  [UInt64]$TestCapacityCapBytes = 0,
  [string]$Model = '',
  [UInt64]$ForceVramBytes = 8589934592,
  [int]$ForceGpuCount = 1,
  [string]$ForceGpuName = 'win3 cpu split',
  [string[]]$ExtraEnv = @(),
  [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = 'Stop'
$root = "C:\MeshLLM\fleet\$NodeName"
$bin = Join-Path $root 'mesh-llm.exe'
$state = Join-Path $root 'state'
$profile = Join-Path $state 'profile'
$capacity = Join-Path $state 'cpu-stage-capacity.bytes'
$capacityLog = Join-Path $state 'capacity.log'
$stdout = Join-Path $state 'stdout.log'
$stderr = Join-Path $state 'stderr.log'
$pidFile = Join-Path $state 'mesh-llm.pid'
$monitorPidFile = Join-Path $state 'capacity-monitor.pid'

function Stop-PidFile([string]$path) {
  if (Test-Path $path) {
    $processId = [int](Get-Content $path -Raw)
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Remove-Item $path -Force -ErrorAction SilentlyContinue
  }
}

if ($Action -eq 'stop') {
  Stop-PidFile $pidFile
  Stop-PidFile $monitorPidFile
  return
}

if ($Action -eq 'status') {
  $processId = if (Test-Path $pidFile) { [int](Get-Content $pidFile -Raw) } else { 0 }
  [pscustomobject]@{
    node = $NodeName
    pid = $processId
    running = [bool](Get-Process -Id $processId -ErrorAction SilentlyContinue)
    capacity_bytes = if (Test-Path $capacity) { [UInt64](Get-Content $capacity -Raw) } else { 0 }
    api = try { Invoke-RestMethod "http://127.0.0.1:$ConsolePort/api/status" -TimeoutSec 2 } catch { $null }
  } | ConvertTo-Json -Depth 6 -Compress
  return
}

New-Item -ItemType Directory -Force -Path $profile,(Join-Path $profile 'AppData\Local') | Out-Null

if ($Action -eq 'monitor') {
  $reserveFloor = [UInt64](768MB)
  while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $available = [UInt64]$os.FreePhysicalMemory * 1024
    $remaining = if ($available -gt $reserveFloor) { $available - $reserveFloor } else { 0 }
    $percentageReserve = [UInt64][Math]::Floor($remaining * 0.10)
    $reserved = [UInt64]($reserveFloor + $percentageReserve)
    $safe = if ($available -gt $reserved) { [UInt64][Math]::Floor(($available - $reserved) * 0.80) } else { 0 }
    if ($TestCapacityCapBytes -gt 0 -and $safe -gt $TestCapacityCapBytes) { $safe = $TestCapacityCapBytes }
    $tmp = "$capacity.tmp"
    [IO.File]::WriteAllText($tmp, [string]$safe, [Text.Encoding]::ASCII)
    Move-Item -Force $tmp $capacity
    Add-Content -Path $capacityLog -Value ("{0:o} worker={1} available_ram={2} reserved_ram={3} advertised_cpu_stage_bytes={4} source=runtime_capacity_file" -f (Get-Date),$NodeName,$available,$reserved,$safe)
    Start-Sleep -Seconds 5
  }
}

if (-not (Test-Path $bin)) { throw "missing binary: $bin" }
if (-not $Token) { throw 'join token required' }
Stop-PidFile $pidFile
Stop-PidFile $monitorPidFile
Remove-Item $capacity -Force -ErrorAction SilentlyContinue
$script = $MyInvocation.MyCommand.Path
$monitorArgs = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$script,'-Action','monitor','-NodeName',$NodeName)
if ($TestCapacityCapBytes -gt 0) { $monitorArgs += @('-TestCapacityCapBytes',[string]$TestCapacityCapBytes) }
$monitor = Start-Process powershell.exe -ArgumentList $monitorArgs -WindowStyle Hidden -PassThru
Set-Content -Path $monitorPidFile -Value $monitor.Id -Encoding ASCII
$deadline = (Get-Date).AddSeconds(10)
while (-not (Test-Path $capacity) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
if (-not (Test-Path $capacity)) { throw 'capacity monitor did not publish' }

$env:HOME = $profile
$env:USERPROFILE = $profile
$env:LOCALAPPDATA = Join-Path $profile 'AppData\Local'
$env:MESH_LLM_RUNTIME_ROOT = Join-Path $state 'runtime'
$env:MESH_LLM_CPU_STAGE_CAPACITY_FILE = $capacity
$env:MESH_LLM_ARTIFACT_TRANSFER = 'open'
$env:MESH_LLM_LOCAL_PACKAGE_FALLBACK = Join-Path $state 'package-bootstrap'
if ($ForceVramBytes -gt 0) { $env:MESH_LLM_FORCE_VRAM_BYTES = "$ForceVramBytes" }
if ($ForceGpuCount -gt 0) { $env:MESH_LLM_FORCE_GPU_COUNT = "$ForceGpuCount" }
if (-not [string]::IsNullOrWhiteSpace($ForceGpuName)) { $env:MESH_LLM_FORCE_GPU_NAME = $ForceGpuName }

function Set-EnvAssignments([string[]]$Assignments) {
  foreach ($assignment in $Assignments) {
    if ([string]::IsNullOrWhiteSpace($assignment)) { continue }
    $idx = $assignment.IndexOf('=')
    if ($idx -lt 1) { throw "Invalid env override '$assignment'. Use NAME=VALUE." }
    $name = $assignment.Substring(0, $idx)
    $value = $assignment.Substring($idx + 1)
    Set-Item -Path "Env:$name" -Value $value
  }
}

Set-EnvAssignments $ExtraEnv
$offload = Join-Path $state 'offload.toml'
@"
[defaults.model_fit]
kv_cache_policy = "saver"
kv_offload = "auto"
cache_type_k = "q4_0"
cache_type_v = "q4_0"
ctx_size = 4096

[defaults.hardware]
gpu_layers = 0
device = "cpu"
mmap = true
"@ | Set-Content -Path $offload -Encoding UTF8
$meshArgs = @('--config',$offload,'serve','--join',$Token,'--mesh-name','llama-local-split','--split','--name',"$NodeName-llama-stage",'--console',[string]$ConsolePort,'--port',[string]$ApiPort,'--bind-ip',$BindIp,'--bind-port',[string]$QuicPort,'--mesh-discovery-mode','mdns','--ctx-size','4096','--device','cpu','--llama-flavor','cpu','--headless')
if (-not [string]::IsNullOrWhiteSpace($Model)) { $meshArgs += @('--model',$Model) }
$meshArgs += $ExtraArgs
$process = Start-Process -FilePath $bin -ArgumentList $meshArgs -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content -Path $pidFile -Value $process.Id -Encoding ASCII
[pscustomobject]@{node=$NodeName;pid=$process.Id;monitor_pid=$monitor.Id;capacity_bytes=[UInt64](Get-Content $capacity -Raw);forced_vram_bytes=$ForceVramBytes;forced_gpu_count=$ForceGpuCount;forced_gpu_name=$ForceGpuName;extra_env=$ExtraEnv;extra_args=$ExtraArgs} | ConvertTo-Json -Compress

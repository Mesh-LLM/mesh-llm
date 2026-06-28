[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][ValidateSet('start','status','stop','logs','preflight')][string]$Action,
  [Parameter(Mandatory=$true)][string]$ConfigBase64,
  [switch]$WaitReady
)
$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'
# Future launches inherit HF_TOKEN from the user environment. This fallback
# makes direct Hugging Face model/package resolution work without writing the
# credential into the fleet JSON or logs.
if (-not $env:HF_TOKEN) { $env:HF_TOKEN = [Environment]::GetEnvironmentVariable('HF_TOKEN', 'User') }
if ($env:HF_TOKEN) { $env:HUGGING_FACE_HUB_TOKEN = $env:HF_TOKEN }
function Decode([string]$v){ ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($v)) | ConvertFrom-Json) }
function Have($v){ $null -ne $v -and [string]$v -ne '' }
function Q([string]$v){ if($v -notmatch '[\s"]'){return $v}; '"'+(($v -replace '(\\*)"','$1$1\"') -replace '(\\*)$','$1$1')+'"' }
function State($c){ Join-Path (Join-Path (if(Have $c.state_root){$c.state_root}else{Join-Path $env:LOCALAPPDATA 'MeshLLM\fleet'}) $c.mesh_name) $c.node_name }
function Status([int]$p){try{Invoke-RestMethod "http://127.0.0.1:$p/api/status" -TimeoutSec 1}catch{$null}}
function WriteOffload($c,$path){
  $l=@('[defaults.model_fit]')
  if(Have $c.kv_cache_policy){$l+="kv_cache_policy = `"$($c.kv_cache_policy)`""}
  if(Have $c.kv_offload){$l+="kv_offload = `"$($c.kv_offload)`""}
  if(Have $c.cache_type_k){$l+="cache_type_k = `"$($c.cache_type_k)`""}
  if(Have $c.cache_type_v){$l+="cache_type_v = `"$($c.cache_type_v)`""}
  if(Have $c.ctx_size){$l+="ctx_size = $($c.ctx_size)"}
  $l+='';$l+='[defaults.hardware]'
  if(Have $c.gpu_layers){if("$($c.gpu_layers)" -match '^-?\d+$'){$l+="gpu_layers = $($c.gpu_layers)"}else{$l+="gpu_layers = `"$($c.gpu_layers)`""}}
  if(Have $c.device){$l+="device = `"$($c.device)`""}
  if(Have $c.mmap){$l+="mmap = $($c.mmap)"}
  Set-Content -LiteralPath $path -Value ($l -join "`r`n") -Encoding UTF8
}
function AddArg($a,[string]$n,$v){if(Have $v){[void]$a.Add($n);[void]$a.Add([string]$v)}}
$c=Decode $ConfigBase64; $s=State $c; New-Item -ItemType Directory -Force -Path $s|Out-Null
$pidFile=Join-Path $s 'mesh-llm.pid'; $out=Join-Path $s 'stdout.log'; $err=Join-Path $s 'stderr.log'
switch($Action){
 'preflight'{ $x=Test-Path $c.bin; $v=if($x){(& $c.bin --version 2>&1|Select-Object -First 1).ToString().Trim()}else{$null}; [pscustomobject]@{node=$c.node_name;machine=$env:COMPUTERNAME;binary_exists=$x;version=$v;expected=$c.expected_version}|ConvertTo-Json -Compress; break }
 'status'{ $p=if(Test-Path $pidFile){[int](Get-Content $pidFile -Raw)}else{$null}; [pscustomobject]@{node=$c.node_name;machine=$env:COMPUTERNAME;pid=$p;running=if($p){[bool](Get-Process -Id $p -ErrorAction SilentlyContinue)}else{$false};status=(Status ([int]$c.console_port))}|ConvertTo-Json -Depth 5 -Compress; break }
 'stop'{if(Test-Path $pidFile){$p=[int](Get-Content $pidFile -Raw);Stop-Process -Id $p -Force -ErrorAction SilentlyContinue;Remove-Item $pidFile -Force -ErrorAction SilentlyContinue;[pscustomobject]@{node=$c.node_name;stopped=$p}|ConvertTo-Json -Compress}else{'{"stopped":null}'};break}
 'logs'{Get-Content $err -Tail 100 -ErrorAction SilentlyContinue;Get-Content $out -Tail 100 -ErrorAction SilentlyContinue;break}
}
if(-not(Test-Path $c.bin)){throw "mesh-llm.exe not found: $($c.bin)"}
$ver=(& $c.bin --version 2>&1|Select-Object -First 1).ToString().Trim()
if(Have $c.expected_version -and $ver -ne "mesh-llm $($c.expected_version)"){throw "version mismatch: $ver (expected mesh-llm $($c.expected_version))"}
if(Test-Path $pidFile){$old=[int](Get-Content $pidFile -Raw);if(Get-Process -Id $old -ErrorAction SilentlyContinue){throw "already running pid $old"}}
$toml=Join-Path $s 'offload.toml';WriteOffload $c $toml
$a=[Collections.Generic.List[string]]::new();if($c.role -eq 'client'){[void]$a.Add('client')}else{[void]$a.Add('serve')}
AddArg $a '--mesh-name' $c.mesh_name;AddArg $a '--console' $c.console_port;AddArg $a '--port' $c.api_port
# Workers stay headless by default. Set NODE_<NAME>_WEB_UI=1 to serve the embedded
# operator UI, which contains both normal Mesh-LLM chat and Flushnet tool chat.
if ($c.web_ui -ne $true) { [void]$a.Add('--headless') }
if($c.role -ne 'client'){AddArg $a '--model' $c.model}
if($c.role -ne 'seed'){$token=[Console]::In.ReadToEnd().Trim();if(-not $token){throw 'invite token required on stdin'};AddArg $a '--join' $token}
if($c.lan -eq $true){AddArg $a '--mesh-discovery-mode' 'mdns'}
if($c.split -eq $true -and $c.role -ne 'client'){[void]$a.Add('--split')}
AddArg $a '--bind-ip' $c.bind_ip;AddArg $a '--bind-port' $c.bind_port;AddArg $a '--ctx-size' $c.ctx_size;AddArg $a '--max-vram' $c.max_vram;AddArg $a '--device' $c.device;AddArg $a '--llama-flavor' $c.llama_flavor;AddArg $a '--tensor-split' $c.tensor_split;AddArg $a '--config' $toml
if(Have $c.extra_args){foreach($z in @($c.extra_args -split '\s+')){if($z){[void]$a.Add($z)}}}
$prev=$env:MESH_LLM_ARTIFACT_TRANSFER;if($c.artifact_transfer -eq 'off'){Remove-Item Env:MESH_LLM_ARTIFACT_TRANSFER -ErrorAction SilentlyContinue}elseif(Have $c.artifact_transfer){$env:MESH_LLM_ARTIFACT_TRANSFER=[string]$c.artifact_transfer}
try{$line=($a|ForEach-Object{Q $_}) -join ' ';$p=Start-Process -FilePath $c.bin -ArgumentList $line -WorkingDirectory (Split-Path $c.bin -Parent) -RedirectStandardOutput $out -RedirectStandardError $err -PassThru}finally{if($null -eq $prev){Remove-Item Env:MESH_LLM_ARTIFACT_TRANSFER -ErrorAction SilentlyContinue}else{$env:MESH_LLM_ARTIFACT_TRANSFER=$prev}}
Set-Content $pidFile $p.Id -Encoding ascii
if($WaitReady){$until=(Get-Date).AddSeconds([int]$c.wait_seconds);do{Start-Sleep -Milliseconds 500;$st=Status([int]$c.console_port)}while($null -eq $st -and (Get-Date)-lt $until -and (Get-Process -Id $p.Id -ErrorAction SilentlyContinue));if($null -eq $st){throw "process started pid $($p.Id); API not ready; inspect $err"}}
[pscustomobject]@{node=$c.node_name;machine=$env:COMPUTERNAME;pid=$p.Id;version=$ver;waited=[bool]$WaitReady;console=$c.console_port;api=$c.api_port}|ConvertTo-Json -Compress

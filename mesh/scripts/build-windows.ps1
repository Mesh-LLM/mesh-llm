# Build the MeshLLM host and console on Windows. Skippy is external: the host
# loads a packaged Skippy runtime at run time, so no native backend is built here.
param(
    [string]$BuildProfile = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../../scripts"))
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$meshUiDir = Join-Path $repoRoot "mesh\crates\mesh-llm-ui"
$compilerLauncherArgs = @()
$compilerCacheBin = $null
$buildProfile = if ($BuildProfile) { $BuildProfile.ToLowerInvariant() } elseif ($env:MESH_LLM_BUILD_PROFILE) { $env:MESH_LLM_BUILD_PROFILE.ToLowerInvariant() } else { "debug" }

switch ($buildProfile) {
    "debug" { if (-not $env:VITE_MESH_LLM_DEBUG_UI) { $env:VITE_MESH_LLM_DEBUG_UI = "true" } }
    "dev" { if (-not $env:VITE_MESH_LLM_DEBUG_UI) { $env:VITE_MESH_LLM_DEBUG_UI = "true" } }
    "release" { $env:VITE_MESH_LLM_DEBUG_UI = "false" }
    default { throw "Unsupported MESH_LLM_BUILD_PROFILE/BuildProfile '$buildProfile'. Expected debug, dev, or release." }
}

function Resolve-CommandPath {
    param([string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    return $null
}

function ConvertTo-CmakePath {
    param([string]$Path)

    if (-not $Path) {
        return $null
    }

    return $Path.Replace('\', '/')
}

function Show-LldInstallInstructions {
    throw @"
LLVM lld was not found for the Windows MSVC target.

lld is required for faster Rust builds (measured up to 26% faster locally).

Install one of these, then rerun the just command:
  rustup component add llvm-tools-preview

Or install LLVM lld-link:
  winget install LLVM.LLVM
  choco install llvm

The build requires lld. It looks for rust-lld.exe in the active Rust sysroot first, then falls
back to rust-lld.exe or lld-link.exe on PATH.
"@
}

function Resolve-LldLinker {
    try {
        $sysroot = (& rustc --print sysroot).Trim()
        if ($LASTEXITCODE -eq 0 -and $sysroot) {
            foreach ($target in @("x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc")) {
                $candidate = Join-Path $sysroot "lib\rustlib\$target\bin\rust-lld.exe"
                if (Test-Path $candidate) {
                    return $candidate
                }
            }
        }
    } catch {
    }

    foreach ($name in @("rust-lld.exe", "lld-link.exe")) {
        $command = Resolve-CommandPath $name
        if ($command) {
            return $command
        }
    }

    return $null
}

function Configure-LldLinker {
    $linker = Resolve-LldLinker
    if (-not $linker) {
        Show-LldInstallInstructions
    }

    if ($env:MESH_CI_STABLE_PROC_MACRO_METADATA -eq "1") {
        $linker = Join-Path $scriptDir "cargo-linker.cmd"
    }
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = $linker
    $env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER = $linker
    Write-Host "Using Rust linker: $linker"
}

function Configure-CompilerCache {
    $script:compilerCacheBin = Resolve-CommandPath "sccache"
    if (-not $script:compilerCacheBin) {
        $script:compilerCacheBin = Resolve-CommandPath "ccache"
    }
    if (-not $script:compilerCacheBin) {
        $script:compilerLauncherArgs = @()
        return
    }

    Write-Host "Using compiler cache: $script:compilerCacheBin"
    $cmakeCompilerCacheBin = ConvertTo-CmakePath $script:compilerCacheBin
    $script:compilerLauncherArgs = @(
        "-DCMAKE_C_COMPILER_LAUNCHER=$cmakeCompilerCacheBin",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=$cmakeCompilerCacheBin"
    )
}

function Set-BuildVersionStamp {
    if ($env:MESH_LLM_BUILD_VERSION) {
        Write-Host "Using preset MESH_LLM_BUILD_VERSION: $($env:MESH_LLM_BUILD_VERSION)"
        return
    }

    $pkgid = $null
    try {
        Push-Location $repoRoot
        $pkgid = (& cargo pkgid -p mesh-llm 2>$null).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $pkgid) {
            Write-Warning "Unable to derive build version; cargo pkgid unavailable."
            Remove-Item Env:MESH_LLM_BUILD_VERSION -ErrorAction SilentlyContinue
            return
        }
    } finally {
        Pop-Location
    }

    $releaseVersion = $pkgid.Substring($pkgid.LastIndexOf('#') + 1)
    if (-not $releaseVersion -or $releaseVersion -eq $pkgid) {
        Write-Warning "Unable to derive build version; cargo pkgid output was unexpected."
        Remove-Item Env:MESH_LLM_BUILD_VERSION -ErrorAction SilentlyContinue
        return
    }

    if ($buildProfile -eq "release") {
        $env:MESH_LLM_BUILD_VERSION = $releaseVersion
        Write-Host "Using release MESH_LLM_BUILD_VERSION: $($env:MESH_LLM_BUILD_VERSION)"
        return
    }

    $sha = $null
    try {
        $sha = (& git -C $repoRoot rev-parse --short=6 HEAD 2>$null).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $sha) {
            Write-Warning "Unable to derive build version; git SHA unavailable."
            Remove-Item Env:MESH_LLM_BUILD_VERSION -ErrorAction SilentlyContinue
            return
        }
    } catch {
        Write-Warning "Unable to derive build version; git SHA unavailable."
        Remove-Item Env:MESH_LLM_BUILD_VERSION -ErrorAction SilentlyContinue
        return
    }
    $sha = $sha.ToUpperInvariant()

    $statusOutput = $null
    try {
        $statusOutput = (& git -C $repoRoot status --porcelain --untracked-files=all 2>$null)
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Unable to derive build version; git status unavailable."
            Remove-Item Env:MESH_LLM_BUILD_VERSION -ErrorAction SilentlyContinue
            return
        }
    } catch {
        Write-Warning "Unable to derive build version; git status unavailable."
        Remove-Item Env:MESH_LLM_BUILD_VERSION -ErrorAction SilentlyContinue
        return
    }

    $dirtySuffix = if ($statusOutput) { ".dirty" } else { "" }
    $env:MESH_LLM_BUILD_VERSION = "$releaseVersion+g$sha$dirtySuffix"
    Write-Host "Derived MESH_LLM_BUILD_VERSION: $($env:MESH_LLM_BUILD_VERSION)"
}

function Test-Sccache {
    return $compilerCacheBin -and ((Split-Path -Leaf $compilerCacheBin).ToLowerInvariant() -like "sccache*")
}

function Import-CmdEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandLine
    )

    $output = & cmd.exe /s /c "$CommandLine && set"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize Windows build environment with command: $CommandLine"
    }

    foreach ($line in $output) {
        if ($line -match '^(?<name>[^=]+)=(?<value>.*)$') {
            Set-Item -Path "env:$($Matches.name)" -Value $Matches.value
        }
    }
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @()
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        $argString = if ($Arguments.Count -gt 0) { " " + ($Arguments -join " ") } else { "" }
        throw "Command failed with exit code ${LASTEXITCODE}: $Command$argString"
    }
}

function Get-UiBuildEnvStampContent {
    return @(
        "MESH_LLM_BUILD_PROFILE=$buildProfile",
        "VITE_MESH_LLM_DEBUG_UI=$($env:VITE_MESH_LLM_DEBUG_UI)",
        "VITE_BASE_PATH=$($env:VITE_BASE_PATH)",
        "VITE_ROUTER_BASE_PATH=$($env:VITE_ROUTER_BASE_PATH)",
        "VITE_STORAGE_NAMESPACE=$($env:VITE_STORAGE_NAMESPACE)"
    ) -join "`n"
}

function Test-UiBuildRequired {
    param(
        [Parameter(Mandatory = $true)]
        [string]$UiDirectory
    )

    $distDir = Join-Path $UiDirectory "dist"
    if (-not (Test-Path $distDir)) {
        return $true
    }

    $distFiles = Get-ChildItem -Path $distDir -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $distFiles) {
        return $true
    }

    $buildEnvStamp = Join-Path $distDir ".mesh-llm-ui-build-env"
    if (-not (Test-Path $buildEnvStamp)) {
        return $true
    }

    $expectedBuildEnvStamp = Get-UiBuildEnvStampContent
    $actualBuildEnvStamp = Get-Content -Path $buildEnvStamp -Raw
    if ($actualBuildEnvStamp.TrimEnd() -ne $expectedBuildEnvStamp.TrimEnd()) {
        return $true
    }

    $distTimestampUtc = (Get-Item $distDir).LastWriteTimeUtc
    $uiBuildInputs = @(
        (Join-Path $UiDirectory "package.json"),
        (Join-Path $UiDirectory "pnpm-lock.yaml"),
        (Join-Path $UiDirectory "vite.config.ts"),
        (Join-Path $UiDirectory "tsconfig.json"),
        (Join-Path $UiDirectory "tsconfig.app.json"),
        (Join-Path $UiDirectory "tsconfig.node.json"),
        (Join-Path $UiDirectory "biome.json"),
        (Join-Path $UiDirectory "index.html"),
        (Join-Path $UiDirectory "src"),
        (Join-Path $UiDirectory "public")
    )

    foreach ($path in $uiBuildInputs) {
        if (-not (Test-Path $path)) {
            continue
        }

        $item = Get-Item $path
        if ($item.PSIsContainer) {
            $newerInput = Get-ChildItem -Path $path -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTimeUtc -gt $distTimestampUtc } |
                Select-Object -First 1
            if ($newerInput) {
                return $true
            }
            continue
        }

        if ($item.LastWriteTimeUtc -gt $distTimestampUtc) {
            return $true
        }
    }

    return $false
}

function Test-PnpmInstallRequired {
    param(
        [Parameter(Mandatory = $true)]
        [string]$UiDirectory
    )

    $nodeModulesDir = Join-Path $UiDirectory "node_modules"
    if (-not (Test-Path $nodeModulesDir)) {
        return $true
    }

    $nodeModulesTimestampUtc = (Get-Item $nodeModulesDir).LastWriteTimeUtc
    foreach ($manifestName in @("package.json", "pnpm-lock.yaml")) {
        $manifestPath = Join-Path $UiDirectory $manifestName
        if (-not (Test-Path $manifestPath)) {
            continue
        }

        if ((Get-Item $manifestPath).LastWriteTimeUtc -gt $nodeModulesTimestampUtc) {
            return $true
        }
    }

    return $false
}

function Get-MissingCommands {
    param([string[]]$Commands)

    $missing = @()
    foreach ($command in $Commands) {
        if (-not (Resolve-CommandPath $command)) {
            $missing += $command
        }
    }
    return $missing
}

function Ensure-MsvcToolchain {
    $requiredTools = @("cl", "link", "lib", "rc", "mt")
    $missingTools = @(Get-MissingCommands -Commands $requiredTools)
    if ($missingTools.Count -eq 0) {
        return
    }

    Write-Host "MSVC/Windows SDK tools missing before vcvars64 import: $($missingTools -join ', ')"

    $programFilesX86 = ${env:ProgramFiles(x86)}
    $vswhereCandidates = @()
    if ($programFilesX86) {
        $vswhereCandidates += (Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    if ($env:ProgramFiles) {
        $vswhereCandidates += (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    $vswhereFromPath = Resolve-CommandPath "vswhere"
    if ($vswhereFromPath) {
        $vswhereCandidates += $vswhereFromPath
    }

    $vcvars64 = $null
    $vswhere = $vswhereCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique -First 1
    if ($vswhere) {
        $installationPathOutput = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
        $installationPath = if ($installationPathOutput) { $installationPathOutput.Trim() } else { "" }
        if ($installationPath) {
            $candidate = Join-Path $installationPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $candidate) {
                $vcvars64 = $candidate
            }
        }
    }

    if (-not $vcvars64) {
        $searchRoots = @($programFilesX86, $env:ProgramFiles) | Where-Object { $_ } | Select-Object -Unique
        foreach ($searchRoot in $searchRoots) {
            $candidate = Get-ChildItem -Path $searchRoot -Filter vcvars64.bat -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -like '*Microsoft Visual Studio*VC\Auxiliary\Build\vcvars64.bat' } |
                Select-Object -First 1
            if ($candidate) {
                $vcvars64 = $candidate.FullName
                break
            }
        }
    }

    if (-not (Test-Path $vcvars64)) {
        throw "Visual Studio Build Tools with vcvars64.bat were not found on this Windows runner."
    }

    Import-CmdEnvironment "`"$vcvars64`" >nul"

    $missingTools = @(Get-MissingCommands -Commands $requiredTools)
    if ($missingTools.Count -gt 0) {
        throw "MSVC toolchain initialization completed, but required Visual Studio/Windows SDK tools are still unavailable in PATH: $($missingTools -join ', '). Install the Windows SDK and Visual Studio Build Tools on this runner."
    }
}

function Invoke-InRepo {
    param(
        [scriptblock]$Script
    )

    Push-Location $repoRoot
    try {
        & $Script
    } finally {
        Pop-Location
    }
}

Ensure-MsvcToolchain
Configure-LldLinker
Configure-CompilerCache
if ((Test-Sccache) -and -not $env:RUSTC_WRAPPER) {
    $env:RUSTC_WRAPPER = $script:compilerCacheBin
    Write-Host "Using sccache for Rust compilation: $env:RUSTC_WRAPPER"
}

Invoke-InRepo {
    if ($env:MESH_LLM_SKIP_UI -eq "1") {
        Write-Host "Skipping mesh-llm UI build because MESH_LLM_SKIP_UI=1."
    } elseif ((Test-Path $meshUiDir) -and (Test-UiBuildRequired -UiDirectory $meshUiDir)) {
        Write-Host "Building mesh-llm UI (profile: $buildProfile, debug UI: $($env:VITE_MESH_LLM_DEBUG_UI))..."
        Push-Location $meshUiDir
        try {
            if (Test-PnpmInstallRequired -UiDirectory $meshUiDir) {
                Invoke-NativeCommand "pnpm" @("install", "--frozen-lockfile")
            }
            Invoke-NativeCommand "pnpm" @("run", "build")
            Set-Content -Path (Join-Path (Join-Path $meshUiDir "dist") ".mesh-llm-ui-build-env") -Value (Get-UiBuildEnvStampContent)
        } finally {
            Pop-Location
        }
    }

    Set-BuildVersionStamp
    $hostArgs = @("build", "--locked", "-p", "mesh-llm", "--bin", "mesh-llm", "--no-default-features", "--features", "web-ui,dynamic-native-runtime")
    $outputProfile = "debug"
    if ($buildProfile -eq "release") {
        $hostArgs += "--release"
        $outputProfile = "release"
    }
    Invoke-NativeCommand "cargo" $hostArgs
    Write-Host "Mesh host: target\$outputProfile\mesh-llm.exe"
}

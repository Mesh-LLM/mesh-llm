@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "MESH_LLD="
for /f "usebackq delims=" %%S in (`rustc --print sysroot 2^>nul`) do set "MESH_RUST_SYSROOT=%%S"
if defined MESH_RUST_SYSROOT (
  for %%T in (x86_64-pc-windows-msvc aarch64-pc-windows-msvc) do (
    if not defined MESH_LLD if exist "!MESH_RUST_SYSROOT!\lib\rustlib\%%T\bin\rust-lld.exe" set "MESH_LLD=!MESH_RUST_SYSROOT!\lib\rustlib\%%T\bin\rust-lld.exe"
  )
)
if not defined MESH_LLD for %%L in (rust-lld.exe lld-link.exe) do (
  if not defined MESH_LLD for /f "delims=" %%P in ('where %%L 2^>nul') do if not defined MESH_LLD set "MESH_LLD=%%P"
)
if not defined MESH_LLD (
  >&2 echo Error: LLVM lld-link is required for Windows Rust links.
  >&2 echo Run "rustup component add llvm-tools-preview" or install LLVM.LLVM with winget.
  exit /b 1
)

set "MESH_LLD_FLAVOR="
for %%F in ("!MESH_LLD!") do if /I "%%~nxF"=="rust-lld.exe" set "MESH_LLD_FLAVOR=-flavor link"

if "%~1"=="--mesh-probe" (
  echo %MESH_LLD%
  exit /b 0
)

setlocal DisableDelayedExpansion
if defined MESH_CI_STABLE_PROC_MACRO_METADATA goto stable_proc_macro_metadata
"%MESH_LLD%" %MESH_LLD_FLAVOR% %*
exit /b %ERRORLEVEL%

:stable_proc_macro_metadata
if not defined MESH_LLM_AUTOMATION_BIN (
  >&2 echo Error: stable proc-macro linking requires restored product-source automation.
  exit /b 1
)
"%MESH_LLM_AUTOMATION_BIN%" --repo-root "%~dp0.." native windows-linker --lld "%MESH_LLD%" -- %MESH_LLD_FLAVOR% %*
exit /b %ERRORLEVEL%

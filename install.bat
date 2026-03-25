@echo off
REM thermal-viewer installer for Windows
REM Run from the repo root in a Command Prompt or PowerShell window.

setlocal enabledelayedexpansion

set REPO=ibmua/thermal-viewer
set INSTALLED_TC=0

echo === thermal-viewer installer ===
echo.

REM ── 1. Python deps ─────────────────────────────────────────────────────────
echo Installing Python package and dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install . --quiet
if errorlevel 1 (
    echo ERROR: package install failed. Is Python 3.9+ installed and on PATH?
    exit /b 1
)
echo   thermal-viewer + Python dependencies OK

REM ── 2. Detect Python tag ───────────────────────────────────────────────────
for /f "delims=" %%i in ('python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"') do set PY_TAG=%%i

REM ── 3a. Try local pre-built wheel ──────────────────────────────────────────
echo Looking for pre-built wheel (thermal_core\wheels\) ...
for %%W in (thermal_core\wheels\thermal_core-*-%PY_TAG%-*-win_amd64.whl) do (
    echo Installing %%W ...
    python -m pip install "%%W" --force-reinstall --quiet
    set INSTALLED_TC=1
    echo   thermal_core ^(Rust^) OK -- 60fps fast path enabled
    goto :after_local
)
:after_local

REM ── 3b. Try GitHub Releases download ───────────────────────────────────────
if %INSTALLED_TC%==0 (
    set WHEEL_NAME=thermal_core-0.1.0-%PY_TAG%-%PY_TAG%-win_amd64.whl
    set WHEEL_URL=https://github.com/%REPO%/releases/latest/download/!WHEEL_NAME!
    echo Trying to download from GitHub Releases...
    powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '!WHEEL_URL!' -OutFile '%TEMP%\!WHEEL_NAME!' -ErrorAction Stop; Write-Host 'Downloaded.' } catch { Write-Host 'Not found.' }"
    if exist "%TEMP%\!WHEEL_NAME!" (
        python -m pip install "%TEMP%\!WHEEL_NAME!" --force-reinstall --quiet
        copy /y "%TEMP%\!WHEEL_NAME!" "thermal_core\wheels\" >nul 2>&1
        set INSTALLED_TC=1
        echo   thermal_core ^(Rust^) OK -- 60fps fast path enabled
    )
)

REM ── 3c. Build from source ──────────────────────────────────────────────────
if %INSTALLED_TC%==0 (
    where cargo >nul 2>&1
    if not errorlevel 1 (
        python -m pip install maturin --quiet
        echo Building thermal_core from source ^(~30 seconds^)...
        cd thermal_core
        python -m maturin build --release --quiet
        for %%W in (target\wheels\thermal_core-*.whl) do (
            python -m pip install "%%W" --force-reinstall --quiet
            set INSTALLED_TC=1
            echo   thermal_core ^(Rust^) OK -- 60fps fast path enabled
        )
        cd ..
    )
)

if %INSTALLED_TC%==0 (
    echo.
    echo   WARNING: thermal_core not installed -- viewer runs in pure-Python mode ^(~55fps^)
    echo   To unlock 60fps:
    echo     1. Install Rust: https://rustup.rs
    echo     2. Re-run this script.
)

echo.
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo   WARNING: ffmpeg not found -- video recording works, but microphone audio
    echo            cannot be finalized into the saved MP4 until ffmpeg is installed.
) else (
    echo   ffmpeg OK -- microphone recording can be muxed into MP4
)

echo.
echo === Install complete ===
echo Run with: python thermal_viewer.py
where thermal-viewer >nul 2>&1
if errorlevel 1 (
    for /f "delims=" %%i in ('python -c "import site; print(site.USER_BASE + r'\\Scripts\\thermal-viewer.exe')"' ) do set TV_SCRIPT=%%i
    echo Console script installed at: !TV_SCRIPT!
) else (
    echo      or: thermal-viewer
)
endlocal

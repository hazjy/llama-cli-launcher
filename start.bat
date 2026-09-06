@echo off
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8

set "SCRIPT_DIR=%~dp0"
set "LLAMA_DIR="
set "LAUNCHER_DIR="

REM === Find llama_launcher module ===
if exist "%SCRIPT_DIR%llama_launcher\__init__.py" set "LAUNCHER_DIR=%SCRIPT_DIR%"
if not defined LAUNCHER_DIR if exist "%SCRIPT_DIR%..\llama_launcher\__init__.py" set "LAUNCHER_DIR=%SCRIPT_DIR%..\"

if defined LAUNCHER_DIR (
    pushd "%LAUNCHER_DIR%"
    set "LAUNCHER_DIR=!CD!"
    popd
    echo [OK] Launcher:  !LAUNCHER_DIR!
) else (
    echo [ERROR] llama_launcher module not found
    pause
    exit /b 1
)

REM === Find llama-server.exe ===
REM Check own dir
if exist "%SCRIPT_DIR%llama-server.exe" set "LLAMA_DIR=%SCRIPT_DIR%"
REM Check parent
if not defined LLAMA_DIR if exist "%SCRIPT_DIR%..\llama-server.exe" set "LLAMA_DIR=%SCRIPT_DIR%..\"
REM Check grandparent
if not defined LLAMA_DIR if exist "%SCRIPT_DIR%..\..\llama-server.exe" set "LLAMA_DIR=%SCRIPT_DIR%..\..\"
REM Check PATH
if not defined LLAMA_DIR (
    for /f "delims=" %%i in ('where llama-server.exe 2^>nul') do (
        set "LLAMA_DIR=%%~dpi"
        goto :found_llama
    )
)
REM Brute force: search D: drive root
if not defined LLAMA_DIR if exist "D:\llama-server.exe" set "LLAMA_DIR=D:\"
if not defined LLAMA_DIR (
    for /f "delims=" %%d in ('dir /b /ad "D:\" 2^>nul') do (
        if exist "D:\%%d\llama-server.exe" set "LLAMA_DIR=D:\%%d\"
    )
)

:found_llama
if defined LLAMA_DIR (
    pushd "%LLAMA_DIR%"
    set "LLAMA_DIR=!CD!"
    popd
)

if not defined LLAMA_DIR (
    echo [ERROR] llama-server.exe not found
    pause
    exit /b 1
)
echo [OK] llama.cpp: !LLAMA_DIR!

REM === Find models directory ===
set "MODELS_DIR="
if exist "!LLAMA_DIR!\models" set "MODELS_DIR=!LLAMA_DIR!\models"

if not defined MODELS_DIR (
    echo [ERROR] Models dir not found in !LLAMA_DIR!
    pause
    exit /b 1
)
pushd "!MODELS_DIR!"
set "MODELS_DIR=!CD!"
popd
echo [OK] Models:    !MODELS_DIR!
echo.

cd /d "!LAUNCHER_DIR!"

REM === Find a Python interpreter that has the dependencies ===
REM Machine may have multiple Pythons; prefer py -3.14 (deps installed),
REM fall back to plain python, then report a friendly error.
set "PY_CMD=python"
where py >nul 2>nul
if not errorlevel 1 (
    py -3.14 -c "import click,rich,gguf,psutil" >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py -3.14"
)
%PY_CMD% -c "import click,rich,gguf,psutil" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 未找到可用的 Python（缺少 click/rich/gguf/psutil 依赖）
    echo   安装依赖: py -3.14 -m pip install click rich gguf psutil
    pause
    exit /b 1
)

set "LLAMA_CPP_DIR=!LLAMA_DIR!"

%PY_CMD% -m llama_launcher.cli run -d "!MODELS_DIR!"
pause

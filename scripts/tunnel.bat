@echo off
REM Usage:
REM   tunnel.bat <remote_host>

if "%~1"=="" (
    echo Usage: %0 ^<remote_host^>
    exit /b 1
)

set REMOTE_HOST=%~1

ssh -N ^
  -R 21812:localhost:21812 ^
  -L 6379:localhost:6379 ^
  %REMOTE_HOST%
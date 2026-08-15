@echo off
chcp 65001 >nul
title 知教 GOAI Demo
cd /d %~dp0

set "PYTHON_CMD="
where python >nul 2>nul
if not errorlevel 1 (
  python -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
  where py >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
  set PYTHON_CMD="%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

if not defined PYTHON_CMD (
  echo [知教] 未检测到可用的 Python。请安装 Python 3.8+：https://www.python.org/downloads/
  pause
  exit /b 1
)

echo [知教] 正在启动服务……
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"
%PYTHON_CMD% server.py
pause

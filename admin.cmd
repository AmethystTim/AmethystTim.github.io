@echo off
rem Double-click to open the local blog admin console (with live preview).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv || goto :error
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)
".venv\Scripts\python.exe" tools\admin.py %*
goto :eof
:error
echo Failed to prepare the Python environment.
pause

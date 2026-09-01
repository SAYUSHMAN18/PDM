@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_and_run.bat first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
set "PYTHONPATH=%CD%\src"
python -m predictive_maintenance.cli analyze --sos data\current\SosFluidSample.xlsx --telemetry data\current\TelematicDataSample.xlsx --output outputs\current
pause
endlocal


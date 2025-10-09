@echo off
REM Run this INSIDE admin\ folder
set PORT=%PORT%
if "%PORT%"=="" set PORT=8080
python -m uvicorn server_quali:app --host 0.0.0.0 --port %PORT%

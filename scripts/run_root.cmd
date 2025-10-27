@echo off
REM Run this at project ROOT (folder that contains admin\)
set PORT=%PORT%
if "%PORT%"=="" set PORT=8080
python -m uvicorn admin.server_quali:app --host 0.0.0.0 --port %PORT%

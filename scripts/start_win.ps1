$ErrorActionPreference = "Stop"
if (-not $env:PORT) { $env:PORT = 8080 }
$here = Get-Location
if (Test-Path "$here\admin\server_quali.py") {
  $target = "admin.server_quali:app"
} elseif (Test-Path "$here\server_quali.py") {
  $target = "server_quali:app"
} else {
  Write-Host "[FATAL] server_quali.py not found. Go to project root or admin\ folder." -ForegroundColor Red
  exit 1
}
python -m uvicorn $target --host 0.0.0.0 --port $env:PORT

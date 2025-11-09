param(
  [int]$Port = 8080,
  [string]$AdminToken = "devtoken",
  [string]$ApiToken   = "devtoken"
)

# 콘솔 실행/예약 실행 모두 안전: PSScriptRoot 우선, 없으면 고정 경로
$ADMIN = if ($PSScriptRoot) { $PSScriptRoot } else { "C:\Users\user\Desktop\퀄리저널\admin" }
$VENV  = Join-Path $ADMIN ".venv"
$UVI   = Join-Path $VENV  "Scripts\uvicorn.exe"

# 토큰 환경변수
$env:ADMIN_TOKEN = $AdminToken
$env:API_TOKEN   = $ApiToken

Set-Location $ADMIN

# 8080 점유 시 8010으로 자동 변경
try {
  $inUse = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop
  if ($inUse) { $Port = 8010 }
} catch { }

# uvicorn 기동
Start-Process -FilePath $UVI -ArgumentList @("server_quali:app","--host","0.0.0.0","--port",$Port) -NoNewWindow

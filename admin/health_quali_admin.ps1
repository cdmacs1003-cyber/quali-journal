# 간단 헬스체크: 8080/8010 둘 다 시도, 실패 시 기동 스크립트 호출
\ = \False
foreach (\ in 8080,8010) {
  try {
    \ = Invoke-WebRequest "http://127.0.0.1:\/api/status" -Headers @{ "X-Admin-Token"="devtoken" } -TimeoutSec 5 -UseBasicParsing
    if (\.StatusCode -eq 200) { \ = \True; break }
  } catch {}
}
if (-not \) {
  powershell -ExecutionPolicy Bypass -File "C:\Users\user\Desktop\퀄리저널\admin\start_admin_task.ps1"
}

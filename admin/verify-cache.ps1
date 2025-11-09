# verify-cache.ps1 — QualiJournal Admin 캐시 헤더/서비스워커 점검
# 사용: PowerShell에서 이 파일이 있는 폴더로 이동 → .\verify-cache.ps1

param(
  [string]$Origin = "https://admin.standardai.co.kr"
)

function Head($url) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Method Head -Uri $url -Headers @{"Cache-Control"="no-cache"}
    return $r
  } catch {
    Write-Host "HEAD 실패: $url" -ForegroundColor Red
    throw
  }
}

Write-Host "== 1) index.html 헤더 확인 ==" -ForegroundColor Cyan
$r1 = Head("$Origin/")
$r1.Headers["Cache-Control"] | ForEach-Object { "Cache-Control: $_" }
if ($r1.Headers["Cache-Control"] -notmatch "no-store") { Write-Warning "index.html에 no-store 누락 가능" }

Write-Host "`n== 2) service-worker.js 헤더 확인 ==" -ForegroundColor Cyan
$r2 = Head("$Origin/service-worker.js")
$r2.Headers["Cache-Control"] | ForEach-Object { "Cache-Control: $_" }
if ($r2.Headers["Cache-Control"] -notmatch "no-store") { Write-Warning "service-worker.js에 no-store 누락 가능" }

Write-Host "`n== 3) 해시 자산 찾기 & 헤더 확인 ==" -ForegroundColor Cyan
$index = (Invoke-WebRequest -UseBasicParsing -Uri "$Origin/index.html" -Headers @{"Cache-Control"="no-cache"}).Content
$match = [regex]::Match($index, "assets/[^""']+\.[a-f0-9]{8,}\.(js|css)")
if (-not $match.Success) { Write-Error "index.html에서 해시 자산을 찾지 못했습니다."; exit 1 }
$assetPath = $match.Value
Write-Host "찾은 자산: $assetPath"
$r3 = Head("$Origin/$assetPath")
$r3.Headers["Cache-Control"] | ForEach-Object { "Cache-Control: $_" }
if ($r3.Headers["Cache-Control"] -notmatch "immutable") { Write-Warning "정적 자산에 immutable 헤더가 보이지 않습니다." }

Write-Host "`n== 4) 끝 ==" -ForegroundColor Green

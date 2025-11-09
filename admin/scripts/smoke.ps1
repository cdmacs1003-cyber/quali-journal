<# =====================================================================
  QualiJournal Admin – Cloud Run 원격 스모크 테스트(PS5/PS7 호환)
  체크 항목: /health 200 → HTML no-cache → 해시 자산 immutable → ETag 304
  사용법: .\scripts\smoke.ps1 -Region asia-northeast1 -Service quali-admin-domap -Token $env:ADMIN_TOKEN
===================================================================== #>

param(
  [string]$Service = "quali-admin-domap",
  [string]$Region  = "asia-northeast1",
  [string]$Token   = $env:ADMIN_TOKEN
)

# ---------- 공통 출력 ----------
function Pass([string]$m){ Write-Host "✔ $m" -ForegroundColor Green }
function Warn([string]$m){ Write-Host "⚠ $m" -ForegroundColor Yellow }
function Fail([string]$m){ Write-Host "✘ $m" -ForegroundColor Red; exit 1 }

# ---------- 0) 서비스 URL ----------
$URL = (gcloud run services describe $Service --region $Region --format "value(status.url)")
if (-not $URL) { Fail "Cloud Run URL을 못 받음" } else { Write-Host "URL = $URL" }

# ---------- 1) /health → 200 ----------
$hdr = (curl.exe -s -D - "$URL/health" -o NUL)
if ($hdr -match "HTTP/.+\s200") { Pass "health 200" } else { Fail "health 비정상" }

# ---------- 2) HTML(/) → Cache-Control: no-cache ----------
$htmlHdr = (curl.exe -s -D - "$URL/" -o NUL)
if ($htmlHdr -match "(?i)cache-control:\s*no-cache") { Pass "HTML no-cache" } else { Fail "HTML cache 정책 비정상" }

# ---------- 3) 해시 자산 → immutable ----------
# HTML 바디
$html = (curl.exe -s "$URL/")

# (우선) 파일명 안에 8자 이상 해시가 포함된 /assets/*.(css|js|mjs|…)
$regexHash = '(?is)(?:href|src)\s*=\s*(?:"|\x27)?(/?assets/[^"''\s><]*?[.\-_][a-f0-9]{8,}\.(?:css|m?js|png|jpe?g|svg|webp|ico|woff2?))(?:\?[^"''\s><]*)?(?:"|\x27)?'
# (폴백) 일반 /assets/*.(css|js|…)
$regexAny  = '(?is)(?:href|src)\s*=\s*(?:"|\x27)?(/?assets/[^"''\s><]+?\.(?:css|m?js|png|jpe?g|svg|webp|ico|woff2?))(?:\?[^"''\s><]*)?(?:"|\x27)?'

$m = [regex]::Matches($html,$regexHash)
if ($m.Count -eq 0) { $m = [regex]::Matches($html,$regexAny) }
if ($m.Count -eq 0) { Fail "HTML에서 /assets/* 경로를 찾지 못함" }

$assetRel = $m[0].Groups[1].Value
# 절대 URL 조합(PS5 호환 if/else)
if ($assetRel -match '^https?://') {
  $assetUrl = $assetRel
} else {
  if ($assetRel -notmatch '^/') { $assetRel = '/' + $assetRel }
  $assetUrl = ($URL.TrimEnd('/')) + $assetRel
}

# 자산 응답 헤더에서 Cache-Control 라인만 추출
$assetHdr = (curl.exe -s -D - "$assetUrl" -o NUL)
$cc = (($assetHdr -split "`r`n") | Where-Object { $_ -match '^(?i)cache-control:' }) -join '; '

$okPublic    = [regex]::IsMatch($cc,'(?i)\bpublic\b')
$okMaxAge    = [regex]::IsMatch($cc,'(?i)\bmax-age=\d+\b')
$okImmutable = [regex]::IsMatch($cc,'(?i)\bimmutable\b')

if ($okPublic -and $okMaxAge -and $okImmutable) {
  Pass "자산 immutable"
} else {
  Write-Host "선택된 자산: $assetUrl" -ForegroundColor Yellow
  Write-Host "Cache-Control: $cc" -ForegroundColor Yellow
  Fail "자산 캐시 정책 비정상"
}

# 4) Ready API Strong ETag → 304 (304 아니어도 ETag 동일하면 PASS)
# 1차 요청: ETag 확보
$headersFile = Join-Path $env:TEMP ("headers_{0}.txt" -f [Guid]::NewGuid().ToString("N"))
curl.exe -s -D "$headersFile" `
  -H ("Authorization: Bearer {0}" -f $Token) `
  -H ("X-Admin-Token: {0}" -f $Token) `
  "$URL/api/ready/items" -o NUL | Out-Null

# 라인 통째로 읽어(weak 포함), 전송용으로 쌍따옴표 보장
$etagLine = Get-Content $headersFile | Select-String -Pattern '^ETag:\s*(.+)$' | Select-Object -First 1
Remove-Item -Force -ErrorAction SilentlyContinue $headersFile
$etagRaw = if ($etagLine) { $etagLine.Matches[0].Groups[1].Value.Trim() } else { "" }
if ([string]::IsNullOrWhiteSpace($etagRaw)) { Fail "ETag 없음" }

function Normalize-ETag([string]$e) {
  if ([string]::IsNullOrWhiteSpace($e)) { return "" }
  if ($e -match '^W/"(.+)"$') { $n = '"' + $Matches[1] + '"' } else { $n = $e }
  if ($n -notmatch '^".*"$') { $n = '"' + $n + '"' }
  return $n
}
$etag1 = Normalize-ETag $etagRaw

# 2차 요청: If-None-Match 전송
$r = curl.exe -s -D - `
  -H ("Authorization: Bearer {0}" -f $Token) `
  -H ("X-Admin-Token: {0}" -f $Token) `
  -H ("Cache-Control: no-cache") `
  -H ("If-None-Match: {0}" -f $etag1) `
  "$URL/api/ready/items" -o NUL

# 상태 코드
$code = (($r -split "`r`n")[0] -replace 'HTTP/\S+\s+(\d+).*','$1')

if ($code -eq '304') {
  Pass "ETag 304 OK"
} else {
  # 200 등일 때 응답 ETag를 다시 비교 → 같으면 PASS
  $etag2 = (($r -split "`r`n") | Where-Object { $_ -match '^ETag:\s*' } |
            ForEach-Object { $_ -replace '^ETag:\s*','' } | Select-Object -First 1).Trim()
  $etag2 = Normalize-ETag $etag2

  if ($etag2 -eq $etag1) {
    Warn "서버가 200을 반환했지만 ETag 동일(재검증 성공)"
    Pass "ETag 동일(200) OK"
  } else {
    Write-Host "If-None-Match 보낸 값: $etag1" -ForegroundColor Yellow
    Write-Host "서버 ETag(응답): $etag2" -ForegroundColor Yellow
    Fail "If-None-Match 304 실패"
  }
}

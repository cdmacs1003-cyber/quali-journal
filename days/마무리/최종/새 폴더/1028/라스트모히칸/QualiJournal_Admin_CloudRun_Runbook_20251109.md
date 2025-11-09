# 퀄리저널(Admin) Cloud Run 배포·검증 런북
_작성시각: 2025-11-09 12:43 KST_

## 0) 전제
- GCP 프로젝트: `quali-journal-prod`
- 지역(도메인용 domap): `asia-northeast1` / 서비스: `quali-admin-domap`
- 지역(백오피스 admin): `asia-northeast3` / 서비스: `quali-journal-admin`
- Artifact Registry 리포지토리: `qualijournal`
- GitHub Actions 자동배포 파일: `.github/workflows/deploy-admin.yml` (push/main·수동 실행 지원)
- 로컬 Docker/WSL 없이 **Cloud Build**만으로 빌드 가능

## 1) 루트에 Cloud Build 설정 파일 준비(이미 있으면 생략)
`cloudbuild.admin.yaml` (루트에 저장)

```yaml
steps:
- name: gcr.io/cloud-builders/docker
  args: ["build","-t","$_IMAGE","-f","admin/Dockerfile","."]
images:
- "$_IMAGE"
```

## 2) PowerShell — 원격 빌드 → 배포 → 최신화
```powershell
# 레포 루트
cd "C:\Users\user\Desktop\퀄리저널"

# 변수
$PROJECT="quali-journal-prod"
$REG="asia-northeast1"
$REPO="qualijournal"
$IMG="asia-northeast1-docker.pkg.dev/$PROJECT/$REPO/quali-admin:$(Get-Date -Format yyyyMMddHHmmss)"

# 빌드/푸시
gcloud config set project $PROJECT
gcloud builds submit --config cloudbuild.admin.yaml --substitutions _IMAGE=$IMG .

# 배포(도메인용 domap)
gcloud run deploy quali-admin-domap `
  --region $REG `
  --image  $IMG `
  --port   8080 `
  --set-secrets "ADMIN_TOKEN=ADMIN_TOKEN:latest" `
  --allow-unauthenticated

# 최신 리비전에 100%
gcloud run services update-traffic quali-admin-domap --to-latest --region $REG
```

## 3) 원격 스모크(health / HTML no-cache / 자산 immutable / ETag)
```powershell
$URL   = gcloud run services describe quali-admin-domap --region asia-northeast1 --format="value(status.url)"
$TOKEN = $env:ADMIN_TOKEN

# 3-1) /health = 200
curl.exe -s -D - "$URL/health" -o NUL | findstr /R "^HTTP/.* 200"

# 3-2) HTML no-cache
curl.exe -s -D - "$URL/" -o NUL | findstr /I "Cache-Control"

# 3-3) 해시 자산 immutable
$html = (curl.exe -s "$URL/")
$css  = ($html | Select-String -Pattern 'href="([^"]*assets/[^"]+\.css)"' -AllMatches).Matches[0].Groups[1].Value
# 절대 URL 만들기(PS5 호환 if/else)
if ($css -match '^https?://') { $assetUrl = $css } else { $assetUrl = ($URL.TrimEnd('/')) + '/' + ($css.TrimStart('/')) }
curl.exe -s -D - $assetUrl -o NUL | findstr /I "Cache-Control"

# 3-4) ETag 재검증(304 또는 동일 ETag 200)
$headersFile = Join-Path $env:TEMP ("headers_{0}.txt" -f [Guid]::NewGuid().ToString("N"))
curl.exe -s -D "$headersFile" -H ("Authorization: Bearer {0}" -f $TOKEN) -H ("X-Admin-Token: {0}" -f $TOKEN) "$URL/api/ready/items" -o NUL | Out-Null
$etagLine = Get-Content $headersFile | Select-String -Pattern '^ETag:\s*(.+)$' | Select-Object -First 1
Remove-Item -Force -ErrorAction SilentlyContinue $headersFile
$etagRaw = if ($etagLine) { $etagLine.Matches[0].Groups[1].Value.Trim() } else { "" }

function Normalize-ETag([string]$e) {
  if ([string]::IsNullOrWhiteSpace($e)) { return "" }
  if ($e -match '^W/\"(.+)\"$') { $n = '"' + $Matches[1] + '"' } else { $n = $e }
  if ($n -notmatch '^\".*\"$') { $n = '"' + $n + '"' }
  return $n
}

$etag1 = Normalize-ETag $etagRaw

$r = curl.exe -s -D - `
  -H ("Authorization: Bearer {0}" -f $TOKEN) `
  -H ("X-Admin-Token: {0}" -f $TOKEN) `
  -H ("If-None-Match: {0}" -f $etag1) `
  "$URL/api/ready/items" -o NUL

$code = (($r -split "`r`n")[0] -replace 'HTTP/\S+\s+(\d+).*','$1')
if ($code -eq '304') {
  "ETag 304 OK"
} else {
  $etag2 = (($r -split "`r`n") | Where-Object { $_ -match '^ETag:\s*' } | ForEach-Object { $_ -replace '^ETag:\s*','' } | Select-Object -First 1).Trim()
  $etag2 = Normalize-ETag $etag2
  if ($etag2 -eq $etag1) { "ETag 동일(200) OK" } else { "ETag 재검증 실패" }
}
```

## 4) 비용 보호(선택)
```powershell
gcloud run services update quali-admin-domap `
  --region asia-northeast1 `
  --min-instances 0 `
  --max-instances 2 `
  --concurrency 80 `
  --cpu-throttling
```

## 5) 트러블슈팅 핵심
- 8080 포트 리슨 실패 → Dockerfile의 CMD를 `python -m uvicorn server_quali:app --host 0.0.0.0 --port ${PORT:-8080}` 형태로 고정.
- 캐시 불일치 → `tools/build_assets_hash.py --clean`가 dist 생성(해시 파일명) 후 서버가 dist를 우선 서빙.
- /api/* 401 → `Authorization: Bearer <ADMIN_TOKEN>` 또는 `X-Admin-Token: <ADMIN_TOKEN>` 헤더 필요.
- 304 대신 200 → ETag 동일성 검사로 PASS 처리.

---

### 부록 A) 원클릭: deploy_and_smoke.ps1
```powershell
param(
  [string]$Project = "quali-journal-prod",
  [string]$Region  = "asia-northeast1",
  [string]$Repo    = "qualijournal",
  [string]$Service = "quali-admin-domap",
  [string]$Token   = $env:ADMIN_TOKEN
)
$ErrorActionPreference="Stop"
$IMG="asia-northeast1-docker.pkg.dev/$Project/$Repo/quali-admin:$(Get-Date -Format yyyyMMddHHmmss)"
gcloud config set project $Project | Out-Null
gcloud builds submit --config cloudbuild.admin.yaml --substitutions _IMAGE=$IMG .
gcloud run deploy $Service --region $Region --image $IMG --port 8080 `
  --set-secrets "ADMIN_TOKEN=ADMIN_TOKEN:latest" --allow-unauthenticated
gcloud run services update-traffic $Service --to-latest --region $Region
.\scripts\smoke.ps1 -Region $Region -Service $Service -Token $Token
```

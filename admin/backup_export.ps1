# backup_export.ps1  (stable / unicode-safe / idempotent)
# 목적: Cloud Run Admin 서비스에서 MD/CSV Export 받아
#       <프로젝트루트>\backup\YYYYMMDD_HHmmss\ 에 저장 (+선택: GCS 업로드)

[CmdletBinding()]
param(
  [string]$Service = 'qualijournal-admin',
  [string]$Region  = 'asia-northeast3',
  # 우선순위: -Token 파라미터 > $env:ADMIN_TOKEN > admin_token.txt
  [string]$Token   = $null
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# 1) 서비스 URL
$U = & gcloud run services describe $Service --region $Region --format "value(status.url)"
if (-not $U) { throw "Service URL not found. Check gcloud project/region/service." }

# 2) 토큰
if (-not $Token) {
  if ($env:ADMIN_TOKEN) { $Token = $env:ADMIN_TOKEN }
  elseif (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'admin_token.txt')) {
    $Token = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'admin_token.txt') -Raw).Trim()
  }
}
if (-not $Token) { throw "ADMIN_TOKEN missing. Use -Token or set ENV ADMIN_TOKEN or admin_token.txt." }

# 3) 백업 폴더 (스크립트 위치 기준: <프로젝트루트>\backup\YYYYMMDD_HHmmss)
$ProjectRoot = Split-Path -Path $PSScriptRoot -Parent            # ...\퀄리저널
$BackupRoot  = Join-Path $ProjectRoot 'backup'
$Stamp       = Get-Date -Format 'yyyyMMdd_HHmmss'
$OutDir      = Join-Path $BackupRoot $Stamp
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null     # 공백/한글 경로 OK

# 4) 출력 파일
$OutMd  = Join-Path $OutDir ("export_{0}.md"  -f $Stamp)
$OutCsv = Join-Path $OutDir ("export_{0}.csv" -f $Stamp)

# 5) 다운로드 (curl.exe를 인자 배열로 호출: 따옴표/공백 안전)
$Auth = "Authorization: Bearer $Token"
& 'curl.exe' -fSL -H $Auth "$U/api/export/md"  -o "$OutMd"
if ($LASTEXITCODE -ne 0) { throw "curl md failed ($LASTEXITCODE)" }

& 'curl.exe' -fSL -H $Auth "$U/api/export/csv" -o "$OutCsv"
if ($LASTEXITCODE -ne 0) { throw "curl csv failed ($LASTEXITCODE)" }

Write-Host "Backup OK"
Write-Host "dir : $OutDir"
Write-Host "md  : $OutMd"
Write-Host "csv : $OutCsv"

# 6) (선택) GCS 업로드: 환경변수 BACKUP_BUCKET이 있으면 업로드
if ($env:BACKUP_BUCKET) {
  & 'gsutil' -m cp "$OutMd" "$OutCsv" $env:BACKUP_BUCKET
  if ($LASTEXITCODE -ne 0) { Write-Warning "GCS upload failed ($LASTEXITCODE)" }
  else { Write-Host "Uploaded to $($env:BACKUP_BUCKET)" }
}

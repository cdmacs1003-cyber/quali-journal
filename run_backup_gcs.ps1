# run_backup_gcs.ps1
#requires -Version 5.1
$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12

function Write-Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }

# === 로깅 유틸 ===
$Global:LOG_DIR  = Join-Path $HOME "Desktop\퀄리저널\logs"
$Global:LOG_FILE = Join-Path $Global:LOG_DIR "backup.log"

function Write-Log {
  param([Parameter(Mandatory)][string]$Level, [Parameter(Mandatory)][string]$Msg)
  if (-not (Test-Path $Global:LOG_DIR)) { New-Item -ItemType Directory -Path $Global:LOG_DIR -Force | Out-Null }
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  Add-Content -Encoding UTF8 -Path $Global:LOG_FILE -Value ("$stamp [$Level] $Msg")
}

function Rotate-Log {
  if (Test-Path $Global:LOG_FILE) {
    $sizeMB = [math]::Round((Get-Item $Global:LOG_FILE).Length/1MB,2)
    if ($sizeMB -ge 5) {
      $bak = Join-Path $Global:LOG_DIR ("backup.log." + (Get-Date -Format "yyyyMMdd"))
      Move-Item -Force $Global:LOG_FILE $bak
    }
  }
}
Rotate-Log
# === /로깅 유틸 ===

# === 공통 재시도 유틸 (최대 3회, 지수 백오프) ===
function Invoke-WithRetry {
  param(
    [Parameter(Mandatory)][ScriptBlock]$Script,
    [int]$MaxAttempts = 3,
    [int]$BaseDelayMs = 500
  )
  $attempt = 1
  while ($true) {
    try { return & $Script }
    catch {
      if ($attempt -ge $MaxAttempts) { throw }
      Start-Sleep -Milliseconds ($BaseDelayMs * [math]::Pow(2, $attempt-1))
      $attempt++
    }
  }
}

function Curl-HttpCode {
  param(
    [Parameter(Mandatory)][string]$Url,
    [string[]]$Headers = @(),
    [string]$OutFile = $null
  )
  $args = @("-s","-w","%{http_code}")
  foreach($h in $Headers){ $args += @("-H",$h) }
  if ($OutFile) { $args += @("-o",$OutFile) }
  $args += $Url
  $code = & curl.exe @args
  return $code
}
# === /공통 유틸 ===

# --- 이메일 전송 함수(환경변수 필요) ---
function Send-Mail {
  param([Parameter(Mandatory)][string]$Subject, [Parameter(Mandatory)][string]$Body)
  try{
    if(!$env:SMTP_HOST -or !$env:SMTP_PORT -or !$env:SMTP_USER -or !$env:SMTP_PASS -or !$env:SMTP_FROM -or !$env:SMTP_TO){
      return
    }
    $smtp             = New-Object System.Net.Mail.SmtpClient($env:SMTP_HOST,[int]$env:SMTP_PORT)
    $smtp.EnableSsl   = $true
    $smtp.Credentials = New-Object System.Net.NetworkCredential($env:SMTP_USER,$env:SMTP_PASS)
    $mail             = New-Object System.Net.Mail.MailMessage($env:SMTP_FROM,$env:SMTP_TO,$Subject,$Body)
    $smtp.Send($mail)
  } catch {
    Write-Log "ERROR" ("MAIL ERROR: " + $_.Exception.Message)
  }
}

# --- GCP 프로젝트/서비스 URL ---
$PROJECT = (gcloud config get-value project)
if(-not $PROJECT){ throw "GCP project not configured." }

$URL = (gcloud run services describe qualijournal-admin --region asia-northeast3 --format="value(status.url)")
if(-not $URL){ throw "Cloud Run service URL not found." }

# --- ADMIN_TOKEN 확보(파일 우선 → 환경변수 폴백) ---
$TokenFile = "C:\Users\user\Desktop\퀄리저널\admin\admin_token.txt"
$TOKEN = ""

if (Test-Path $TokenFile) {
  $content = Get-Content -Path $TokenFile -Encoding ascii -Raw -ErrorAction SilentlyContinue
  if (-not [string]::IsNullOrWhiteSpace($content)) { $TOKEN = $content.Trim() }
}
if ([string]::IsNullOrWhiteSpace($TOKEN)) { $TOKEN = ($env:TOKEN       | ForEach-Object { $_ }) }
if ([string]::IsNullOrWhiteSpace($TOKEN)) { $TOKEN = ($env:ADMIN_TOKEN | ForEach-Object { $_ }) }
if ([string]::IsNullOrWhiteSpace($TOKEN)) { throw "ADMIN_TOKEN is empty. (admin_token.txt or env TOKEN/ADMIN_TOKEN)" }

# --- 로컬 백업 폴더 ---
$base = Join-Path $HOME "Desktop\퀄리저널\backup"
$dts  = Get-Date -Format "yyyyMMdd_HHmmss"
$dir  = Join-Path $base $dts
New-Item -ItemType Directory -Path $dir -Force | Out-Null

# --- Export 파일 경로 ---
$mdPath  = Join-Path $dir ("export_{0}.md"  -f $dts)
$csvPath = Join-Path $dir ("export_{0}.csv" -f $dts)

# --- Export 호출 ---
$h1 = "Authorization: Bearer $TOKEN"
$h2 = "X-Geo-Country: KR"

$code_md  = Invoke-WithRetry { Curl-HttpCode -Url "$URL/api/export/md"  -Headers @($h1,$h2) -OutFile $mdPath }
$code_csv = Invoke-WithRetry { Curl-HttpCode -Url "$URL/api/export/csv" -Headers @($h1,$h2) -OutFile $csvPath }

Write-Info "EXPORT md=$code_md, csv=$code_csv"
Write-Log  "INFO"  ("EXPORT md={0}, csv={1}" -f $code_md,$code_csv)

$exportOK = ($code_md -eq "200" -and $code_csv -eq "200")

# --- GCS 업로드(성공 시, 재시도 포함) ---
$BUCKET = "gs://qualijournal-backup-$PROJECT"
$y=(Get-Date -Format "yyyy"); $m=(Get-Date -Format "MM"); $d=(Get-Date -Format "dd")
$dst = "$BUCKET/y=$y/m=$m/d=$d/"

if ($exportOK) {
  Invoke-WithRetry { gcloud storage cp "$mdPath"  "$dst" | Out-Null } | Out-Null
  Invoke-WithRetry { gcloud storage cp "$csvPath" "$dst" | Out-Null } | Out-Null
}

# --- 서버 통보(/api/backup/notify) + 메일 알림 ---
if ($exportOK) {
  # 성공 메일
  Send-Mail "✅ QualiJournal 백업 성공" (
    "OK: {0}, {1}`nGCS: {2}" -f (Split-Path -Leaf $mdPath), (Split-Path -Leaf $csvPath), $dst
  )
  Write-Log "INFO" ("SUCCESS: {0}, {1} → {2}" -f (Split-Path -Leaf $mdPath),(Split-Path -Leaf $csvPath),$dst)

  # 서버에 성공 통보 (PS 5.1 안전)
  try {
    $unix = [int]([DateTimeOffset](Get-Date)).ToUnixTimeSeconds()

    $size_md  = 0
    if (Test-Path $mdPath)  { $size_md  = (Get-Item $mdPath).Length }

    $size_csv = 0
    if (Test-Path $csvPath) { $size_csv = (Get-Item $csvPath).Length }

    $payload = @{
      ok       = $true
      ts       = $unix
      size_md  = $size_md
      size_csv = $size_csv
    } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri "$URL/api/backup/notify" `
      -Headers @{ Authorization = "Bearer $TOKEN"; "Content-Type" = "application/json" } `
      -Body $payload | Out-Null
  } catch {
    Write-Log "ERROR" ("NOTIFY ERROR: " + $_.Exception.Message)
  }
}
else {
  # 실패 메일 + 로그 + 종료
  Write-Log "ERROR" ("EXPORT FAILED: md={0}, csv={1}  URL={2}" -f $code_md,$code_csv,$URL)
  Send-Mail "❌ QualiJournal 백업 실패" ("md={0}, csv={1}`nURL={2}" -f $code_md,$code_csv,$URL)
  throw "Export failed: md=$code_md, csv=$code_csv"
}

# --- 완료 메시지 ---
$finalMsg = ("완료: 로컬={0}  GCS={1}  (md={2}, csv={3})" -f $dir, $dst, $code_md, $code_csv)
Write-Host $finalMsg -ForegroundColor Green
Write-Log  "INFO"  $finalMsg
exit 0

# =====================================================================
# run_quali_today.ps1 — QualiJournal Daily Run (UTF-8 / Non-Breaking)
# - 커뮤니티: 수집 → 발행(HTML/MD/JSON)
# - 키워드: (여러 개) 선택 → 자동 승인(상위 N) → 발행
# - 종합 발행(공식+커뮤)
# - 결과 알림(Slack/Email) *값 비어있으면 자동 건너뜀
# - 로그 7일 경과분 압축 보관
# =====================================================================

#region 0) 공통 준비(경로/인코딩/함수)

# 스크립트 위치를 루트로 고정
$ROOT = if ($PSScriptRoot) { $PSScriptRoot } else { Join-Path $env:USERPROFILE 'Desktop\퀄리저널' }
Set-Location $ROOT

# UTF-8 고정(콘솔/파이썬)
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$env:PYTHONUTF8      = "1"
$env:PYTHONIOENCODING= "utf-8"
$env:PYTHONPATH      = (Get-Location).Path

# 공통 함수: 단계 실행(에러 발생해도 이어감, 메시지만 표시)
function Invoke-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][scriptblock]$Do
    )
    Write-Host ("`n>>> " + $Name) -ForegroundColor Cyan
    try {
        & $Do
        if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
            Write-Host ("[warn] {0}: exit {1}" -f $Name, $LASTEXITCODE) -ForegroundColor Yellow
        } else {
            Write-Host ("[ok]   {0}" -f $Name) -ForegroundColor Green
        }
    } catch {
        Write-Host ("[err]  {0}: {1}" -f $Name,$_.Exception.Message) -ForegroundColor Red
    }
}

#endregion

#region 1) 설정(필요 시 값만 수정) — 비어있으면 자동 건너뜀

# 여러 키워드: 파일 우선(data\keywords.txt 줄단위), 없으면 기본 배열 사용
$KW_FILE = Join-Path $ROOT 'data\keywords.txt'
$KWS = @()
if (Test-Path $KW_FILE) {
    $KWS = Get-Content $KW_FILE | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith('#') }
}
if (-not $KWS -or $KWS.Count -eq 0) {
    $KWS = @("IPC-A-610","SMT","J-STD-001")    # ← 원하는 키워드 추가
}

# 자동 승인 갯수(권장 15)
$AUTO_APPROVE_TOP = 15

# Slack Incoming Webhook (없으면 "")
$SLACK_WEBHOOK = ""   # 예: https://hooks.slack.com/services/...

# Email (없으면 "")
$SMTP_HOST = ""       # 예: smtp.gmail.com
$SMTP_PORT = 587
$MAIL_FROM = ""       # 예: bot@yourdomain.com
$MAIL_TO   = ""       # 예: you@yourdomain.com
$MAIL_USER = ""       # 인증 계정(옵션)
$MAIL_PASS = ""       # 앱 비밀번호/토큰(옵션)
$MAIL_SUBJ = "[퀄리저널] 일일 결과"

#endregion

#region 2) 커뮤니티: 수집 → 발행(HTML/MD/JSON)

Invoke-Step "Community: Collect"        { py .\orchestrator.py --collect-community }
Invoke-Step "Community: Publish(all)"   { py .\orchestrator.py --publish-community --format all }

#endregion

#region 3) 키워드: 다중 선택 → 자동 승인 → 발행

foreach ($KW in $KWS) {
    Invoke-Step ("Keyword: Collect — " + $KW) { py .\orchestrator.py --collect-keyword "$KW" }
    Invoke-Step ("Keyword: ApproveTop " + $AUTO_APPROVE_TOP + " — " + $KW) {
        py .\orchestrator.py --approve-keyword "$KW" --approve-keyword-top $AUTO_APPROVE_TOP
    }
    Invoke-Step ("Keyword: Publish — " + $KW) { py .\orchestrator.py --publish-keyword "$KW" }
}

#endregion

#region 4) 종합 발행(공식+커뮤)

Invoke-Step "Daily: Publish(all)" { py .\orchestrator.py --publish --format all }

#endregion

#region 5) 결과 확인 & 요약 메시지

$today = Get-Date -Format 'yyyy-MM-dd'
$ok_comm  = Test-Path (Join-Path $ROOT ("archive\community_{0}.json" -f $today))
$ok_daily = Test-Path (Join-Path $ROOT ("archive\daily_{0}.json" -f $today))

# 키워드는 대표로 첫 번째 기준 True/False, 전체 목록은 상세에 포함
$firstKW  = $KWS[0]
$ok_kw    = Test-Path (Join-Path $ROOT ("archive\{0}_{1}.json" -f $today,$firstKW))

"community  : $ok_comm"
"keyword[$firstKW] : $ok_kw"
"daily      : $ok_daily"

# 상세(옵션): 각 키워드별 존재 여부
$kwChecks = @()
foreach ($K in $KWS) {
    $kwChecks += ("- {0} : {1}" -f $K,(Test-Path (Join-Path $ROOT ("archive\{0}_{1}.json" -f $today,$K))))
}
$msg = @"
[퀄리저널] 일일 결과 ($today)
community  : $ok_comm
keyword[$firstKW] : $ok_kw
daily      : $ok_daily

[키워드별 상태]
$($kwChecks -join "`n")
"@

#endregion

#region 6) 알림(Slack / Email) — 값이 비어있으면 자동 Skip

if ($SLACK_WEBHOOK) {
    Invoke-Step "Notify: Slack" {
        $payload = @{ text = $msg }
        Invoke-RestMethod -Method POST -Uri $SLACK_WEBHOOK -Body (ConvertTo-Json $payload) `
            -ContentType 'application/json; charset=utf-8'
    }
}

if ($SMTP_HOST -and $MAIL_FROM -and $MAIL_TO) {
    Invoke-Step "Notify: Email" {
        $sec  = if ($MAIL_PASS) { (ConvertTo-SecureString $MAIL_PASS -AsPlainText -Force) } else { $null }
        $cred = if ($MAIL_USER -and $sec) { New-Object System.Management.Automation.PSCredential($MAIL_USER,$sec) } else { $null }
        Send-MailMessage -SmtpServer $SMTP_HOST -Port $SMTP_PORT -UseSsl `
            -From $MAIL_FROM -To $MAIL_TO -Subject ($MAIL_SUBJ + " - " + $today) `
            -Body $msg -Encoding UTF8 -Credential $cred
    }
}

#endregion

#region 7) 로그 7일 경과분 압축 보관

$LOG_DIR  = Join-Path $ROOT 'logs'
$LOG_ARCH = Join-Path $LOG_DIR 'archive'
if (Test-Path $LOG_DIR) {
    New-Item -ItemType Directory -Force $LOG_ARCH | Out-Null
    $old = Get-ChildItem $LOG_DIR -File -Recurse | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) }
    if ($old -and $old.Count -gt 0) {
        $zip = Join-Path $LOG_ARCH ("logs_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".zip")
        try {
            Compress-Archive -Path ($old.FullName) -DestinationPath $zip -Force
            $old | Remove-Item -Force
            Write-Host ("[logs] archived => {0}" -f $zip) -ForegroundColor DarkCyan
        } catch {
            Write-Host ("[logs] archive FAIL: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
        }
    }
}

#endregion

# 끝

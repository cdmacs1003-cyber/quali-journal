# run_log_report_weekly.ps1
#requires -Version 5.1
$ErrorActionPreference="Stop"
[Console]::OutputEncoding=[Text.Encoding]::UTF8

$LOG = Join-Path $HOME "Desktop\퀄리저널\logs\backup.log"
if(!(Test-Path $LOG)){ throw "로그 파일이 없습니다: $LOG" }

$today   = Get-Date
$from    = $today.AddDays(-6).Date  # 지난 7일
$lines   = Get-Content $LOG -Encoding UTF8

# 로그 파서 (yyyy-MM-dd HH:mm:ss [LEVEL] message)
$re='^(?<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?<lvl>[A-Z]+)\] (?<msg>.*)$'
$parsed = foreach($ln in $lines){
  if($ln -match $re){
    $ts=[datetime]::ParseExact($Matches.ts,'yyyy-MM-dd HH:mm:ss',$null)
    if($ts -ge $from){ [PSCustomObject]@{ ts=$ts; lvl=$Matches.lvl; msg=$Matches.msg } }
  }
}

$byDay   = $parsed | Group-Object { $_.ts.ToString('yyyy-MM-dd') } | Sort-Object Name
$success = ($parsed | Where-Object {$_.msg -like 'SUCCESS*'}).Count
$failed  = ($parsed | Where-Object {$_.lvl -eq 'ERROR'}).Count

# Markdown 생성
$md  = New-Object System.Collections.Generic.List[string]
$md.Add("# Weekly Backup Log Report")
$md.Add(("* 기간: {0} ~ {1}" -f $from.ToString('yyyy-MM-dd'), $today.ToString('yyyy-MM-dd')))
$md.Add(("* 총 성공: {0}, 총 실패: {1}" -f $success,$failed))
$md.Add("")
foreach($g in $byDay){
  $ok = ($g.Group | Where-Object {$_.msg -like 'SUCCESS*'}).Count
  $er = ($g.Group | Where-Object {$_.lvl -eq 'ERROR'}).Count
  $md.Add("## " + $g.Name + ("  —  성공 {0}, 실패 {1}" -f $ok,$er))
  foreach($r in ($g.Group | Sort-Object ts)){
    $md.Add(("- {0} [{1}] {2}" -f $r.ts.ToString('HH:mm:ss'),$r.lvl,$r.msg))
  }
  $md.Add("")
}

# 저장 + (옵션) GCS 업로드
$REPORT_DIR = Join-Path $HOME "Desktop\퀄리저널\reports"
New-Item -ItemType Directory -Path $REPORT_DIR -Force | Out-Null
$fn = "weekly_log_report_{0}.md" -f (Get-Date -Format "yyyyMMdd_HHmm")
$fp = Join-Path $REPORT_DIR $fn
$md | Set-Content -Encoding UTF8 $fp
Write-Host "주간 리포트 생성: $fp" -ForegroundColor Green

# GCS 업로드(선택) — 프로젝트가 설정돼 있고 버킷 이름 규칙을 사용할 때만
try{
  $PROJECT = (gcloud config get-value project)
  if($PROJECT){
    $BUCKET = "gs://qualijournal-backup-$PROJECT/reports/weekly/"
    gcloud storage cp "$fp" "$BUCKET" | Out-Null
    Write-Host "GCS 업로드: $BUCKET$fn" -ForegroundColor Green
  }
}catch{}

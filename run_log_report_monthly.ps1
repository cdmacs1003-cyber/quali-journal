# run_log_report_monthly.ps1
#requires -Version 5.1
$ErrorActionPreference="Stop"
[Console]::OutputEncoding=[Text.Encoding]::UTF8

$LOG = Join-Path $HOME "Desktop\퀄리저널\logs\backup.log"
if(!(Test-Path $LOG)){ throw "로그 파일이 없습니다: $LOG" }

$now    = Get-Date
$start  = Get-Date -Year $now.Year -Month $now.Month -Day 1
$end    = $start.AddMonths(1).AddSeconds(-1)

$lines  = Get-Content $LOG -Encoding UTF8
$re='^(?<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?<lvl>[A-Z]+)\] (?<msg>.*)$'
$parsed = foreach($ln in $lines){
  if($ln -match $re){
    $ts=[datetime]::ParseExact($Matches.ts,'yyyy-MM-dd HH:mm:ss',$null)
    if($ts -ge $start -and $ts -le $end){ [PSCustomObject]@{ ts=$ts; lvl=$Matches.lvl; msg=$Matches.msg } }
  }
}

$byDay   = $parsed | Group-Object { $_.ts.ToString('yyyy-MM-dd') } | Sort-Object Name
$success = ($parsed | Where-Object {$_.msg -like 'SUCCESS*'}).Count
$failed  = ($parsed | Where-Object {$_.lvl -eq 'ERROR'}).Count

# KPI
$mdOK    = ($parsed | Where-Object {$_.msg -like 'EXPORT md=200*' -or $_.msg -like 'EXPORT md=200,*'}).Count
$csvOK   = ($parsed | Where-Object {$_.msg -like 'EXPORT csv=200*' -or $_.msg -like 'EXPORT *, csv=200'}).Count

# Markdown
$md  = New-Object System.Collections.Generic.List[string]
$md.Add("# Monthly Backup Log Report")
$md.Add(("* 기간: {0} ~ {1}" -f $start.ToString('yyyy-MM-dd'), $end.ToString('yyyy-MM-dd')))
$md.Add(("* 총 성공: {0}, 총 실패: {1}, md 200: {2}, csv 200: {3}" -f $success,$failed,$mdOK,$csvOK))
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

$REPORT_DIR = Join-Path $HOME "Desktop\퀄리저널\reports"
New-Item -ItemType Directory -Path $REPORT_DIR -Force | Out-Null
$fn = "monthly_log_report_{0}.md" -f (Get-Date -Format "yyyyMM")
$fp = Join-Path $REPORT_DIR $fn
$md | Set-Content -Encoding UTF8 $fp
Write-Host "월간 리포트 생성: $fp" -ForegroundColor Green

try{
  $PROJECT = (gcloud config get-value project)
  if($PROJECT){
    $BUCKET = "gs://qualijournal-backup-$PROJECT/reports/monthly/"
    gcloud storage cp "$fp" "$BUCKET" | Out-Null
    Write-Host "GCS 업로드: $BUCKET$fn" -ForegroundColor Green
  }
}catch{}

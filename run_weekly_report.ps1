$ErrorActionPreference="Stop"
$PROJECT=(gcloud config get-value project)
$BUCKET="gs://qualijournal-backup-$PROJECT"
$U=(gcloud run services describe qualijournal-admin --region asia-northeast3 --format="value(status.url)")

# 지난 7일 경로(y/m/d) 스캔  Markdown 생성
$today = Get-Date
$lines = @("# Weekly Backup Report", "`n기간: {0} ~ {1}" -f ($today.AddDays(-6).ToString("yyyy-MM-dd")), $today.ToString("yyyy-MM-dd"), "")

for($i=6;$i -ge 0;$i--){
  $d=$today.AddDays(-$i)
  $y=$d.ToString("yyyy"); $m=$d.ToString("MM"); $dd=$d.ToString("dd")
  $prefix="$BUCKET/y=$y/m=$m/d=$dd/"
  $list=(gcloud storage ls $prefix 2>$null)
  $lines += "## $y-$m-$dd"
  if($list){
    $list.Trim().Split("`n") | ForEach-Object {
      if($_){ $lines += ("- "+$_) }
    }
  }else{
    $lines += "- (파일 없음)"
  }
  $lines += ""
}

# 로컬 저장 + GCS 업로드(weekly/)
$base=Join-Path $HOME "Desktop\퀄리저널\backup"
New-Item -ItemType Directory -Force -Path $base | Out-Null
$fn="weekly_report_{0}.md" -f $today.ToString("yyyyMMdd_HHmm")
$fp=Join-Path $base $fn
$lines | Set-Content $fp -Encoding utf8
gcloud storage cp $fp "$BUCKET/weekly/$fn"

# (선택) 텔레그램 알림
if($env:TG_BOT_TOKEN -and $env:TG_CHAT_ID){
  $u="https://api.telegram.org/bot$env:TG_BOT_TOKEN/sendMessage"
  Invoke-RestMethod -Method Post -Uri $u -Body @{chat_id=$env:TG_CHAT_ID; text="🗓️ 주간 리포트 업로드: $fn"}
}

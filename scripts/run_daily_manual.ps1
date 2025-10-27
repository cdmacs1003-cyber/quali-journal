# run_daily_manual.ps1
$ErrorActionPreference = 'Stop'

# 0) 서비스 URL/토큰
$SERVICE = "qualijournal-admin"
$REGION  = "asia-northeast3"
$U = (gcloud run services describe $SERVICE --region $REGION --format="value(status.url)")
$TOKEN = "0fbc7e8eac0745d3a68434a2d43f8d3e"   # 필요 시 새 토큰으로 교체

function PostJson($path){
  $url = "$U$path"
  Write-Host "POST $url"
  curl.exe -sS -X POST $url `
    -H "Authorization: Bearer $TOKEN" `
    -H "Content-Type: application/json" `
    -d "{}" | Out-Host
}

# 1) 보고서 생성
PostJson "/api/report"

# 2) 요약/번역(전체)
PostJson "/api/enrich/keyword"

# 3) 요약/번역(선정본)
PostJson "/api/enrich/selection"

# 4) Export + 백업(이미 만들어 둔 스크립트 재사용)
& "$PSScriptRoot\backup_export.ps1"

Write-Host "DONE: report → enrich(all) → enrich(selection) → export+backup"

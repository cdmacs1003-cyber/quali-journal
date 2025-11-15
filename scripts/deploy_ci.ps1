Param(
  [string]$Project   = $env:PROJECT_ID,
  [string]$Region    = $env:REGION,
  [string]$Service   = $env:SERVICE
)

if (-not $Project) { throw "PROJECT_ID env not set" }
if (-not $Region)  { $Region = "asia-northeast3" }
if (-not $Service) { $Service = "quali-admin" }

gcloud config set project $Project | Out-Null
gcloud services enable run.googleapis.com cloudbuild.googleapis.com | Out-Null

# 소스 배포 (Dockerfile 사용 시 --source . 그대로 가능)
gcloud run deploy $Service `
  --region $Region `
  --source . `
  --allow-unauthenticated `
  --set-env-vars "QUALI_DB_MODE=local" `
  --set-env-vars "GATE_REQUIRED=15" `
  --set-env-vars "PYTHONUNBUFFERED=1" `
  --traffic latest=100

# 안전망: 최신 리비전에 100% 트래픽 강제
gcloud run services update-traffic $Service --region $Region --to-latest

Write-Host "Deployed $Service ($Region) with 100% traffic to latest."

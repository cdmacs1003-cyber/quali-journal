param(
  [string]$Project = "quali-journal-prod",
  [string]$Region  = "asia-northeast1",
  [string]$Repo    = "qualijournal",
  [string]$Service = "quali-admin-domap",
  [string]$Token   = $env:ADMIN_TOKEN
)

$ErrorActionPreference = "Stop"
$IMG="asia-northeast1-docker.pkg.dev/$Project/$Repo/quali-admin:$(Get-Date -Format yyyyMMddHHmmss)"

Write-Host " Build & Push: $IMG"
gcloud config set project $Project | Out-Null
gcloud builds submit --config cloudbuild.admin.yaml --substitutions _IMAGE=$IMG .

Write-Host " Deploy to Cloud Run ($Service)"
gcloud run deploy $Service --region $Region --image $IMG --port 8080 `
  --set-secrets "ADMIN_TOKEN=ADMIN_TOKEN:latest" --allow-unauthenticated

Write-Host " Traffic  latest"
gcloud run services update-traffic $Service --to-latest --region $Region

Write-Host " Smoke tests"
.\scripts\smoke.ps1 -Region $Region -Service $Service -Token $Token

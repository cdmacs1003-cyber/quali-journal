param(
  [string]$Project      = "quali-journal-prod",
  [string]$Pool         = "github-wif",
  [string]$Provider     = "github-oidc",
  [string]$Repo         = "cdmacs1003-cyber/quali-journal",
  [string]$SA           = "github-deploy@quali-journal-prod.iam.gserviceaccount.com",
  [string]$WorkflowPath = ".github/workflows/deploy-admin-domap.yml",
  [string]$Ref          = ""
)
function Fail($m){Write-Host $m -ForegroundColor Red; exit 1}
# GCP 프로젝트
& gcloud config set project $Project *> $null
$projNum = (& gcloud projects describe $Project --format "value(projectNumber)").Trim(); if (-not $projNum){ Fail "GCP 프로젝트 조회 실패" }
# 풀/프로바이더 보장
$poolName = (& gcloud iam workload-identity-pools describe $Pool --location=global --format "value(name)" 2>$null).Trim()
if (-not $poolName){ & gcloud iam workload-identity-pools create $Pool --location=global --display-name "GitHub WIF" *> $null }
$provName = (& gcloud iam workload-identity-pools providers describe $Provider --location=global --workload-identity-pool=$Pool --format "value(name)" 2>$null).Trim()
if (-not $provName){
  & gcloud iam workload-identity-pools providers create-oidc $Provider `
    --location=global --workload-identity-pool=$Pool `
    --display-name "GitHub OIDC" `
    --issuer-uri=https://token.actions.githubusercontent.com `
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.ref=assertion.ref" *> $null
  $provName = (& gcloud iam workload-identity-pools providers describe $Provider --location=global --workload-identity-pool=$Pool --format "value(name)").Trim()
}
# WIF 사용 권한(레포 한정)
$member = "principalSet://iam.googleapis.com/projects/$projNum/locations/global/workloadIdentityPools/$Pool/attribute.repository/$Repo"
& gcloud iam service-accounts add-iam-policy-binding $SA --role roles/iam.workloadIdentityUser --member $member *> $null
# GitHub 시크릿
& gh secret set GCP_PROJECT                    -R $Repo -b $Project
& gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER -R $Repo -b $provName
& gh secret set GCP_SERVICE_ACCOUNT            -R $Repo -b $SA
# 워크플로 실행 + watch
if ([string]::IsNullOrWhiteSpace($Ref)){ $Ref = (& git rev-parse --abbrev-ref HEAD).Trim() }
$wfName = [System.IO.Path]::GetFileName($WorkflowPath)
& gh workflow run -R $Repo $wfName --ref $Ref *> $null
Start-Sleep -Seconds 6
$json = gh run list -R $Repo --workflow $wfName --branch $Ref --limit 1 --json databaseId,status,conclusion | ConvertFrom-Json
if ($json -and $json.Count -ge 1){
  $id = $json[0].databaseId; & gh run watch -R $Repo $id --exit-status; exit $LASTEXITCODE
} else { Write-Host "디스패치 완료. Actions 탭에서 실행 내역 확인" -ForegroundColor Yellow }

param(
  [string]$Commit      = "fix: admin deploy",
  [string]$Project     = "quali-journal-prod",
  [string]$Pool        = "github-wif",
  [string]$Provider    = "github-oidc",
  [string]$Repo        = "cdmacs1003-cyber/quali-journal",
  [string]$SA          = "github-deploy@quali-journal-prod.iam.gserviceaccount.com",
  [string]$WorkflowPath= ".github/workflows/deploy-admin-domap.yml"
)
Set-ExecutionPolicy -Scope Process Bypass -Force
.\fix-commit-push-v2.ps1 -Commit $Commit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.\setup-wif-and-run-v2.ps1 -Project $Project -Pool $Pool -Provider $Provider -Repo $Repo -SA $SA -WorkflowPath $WorkflowPath

# QualiJournal Admin Deploy Runbook (canonical)
## 1) 커밋/푸시
.\fix-commit-push-v2.ps1 -Commit "fix: 작업 내용"
## 2) 배포(WIF 보정 + 워크플로 트리거)
.\setup-wif-and-run-v2.ps1 `
  -Project "quali-journal-prod" `
  -Pool "github-wif" -Provider "github-oidc" `
  -Repo "cdmacs1003-cyber/quali-journal" `
  -SA "github-deploy@quali-journal-prod.iam.gserviceaccount.com" `
  -WorkflowPath ".github/workflows/deploy-admin-domap.yml"
## 3) 최신 리비전 트래픽 전환
gcloud run services update-traffic quali-admin-domap --to-latest `
  --region asia-northeast1 --project quali-journal-prod
## 4) 운영 3줄 점검
$BASE="https://admin.standardai.co.kr"
curl.exe -sI "$BASE/"                                  | findstr /R "^Cache-Control"
curl.exe -sI "$BASE/service-worker.js?v=$(Get-Random)" | findstr /R "^HTTP\|^Cache-Control"
curl.exe -s  "$BASE/service-worker.js?v=$(Get-Random)" -o sw.js; `
  Select-String -Path .\sw.js -Pattern "BUILD|skipWaiting|clients\.claim"

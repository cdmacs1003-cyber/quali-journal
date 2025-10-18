# QualiJournal — Cloud Run Deploy Success Snapshot
**Date:** 2025-10-16 (KST)

This snapshot captures what worked today so we can reuse it next time.

---

## 1) Secrets & Token (ADMIN_TOKEN)
- Fix type conflict in one shot:
```
gcloud run services update quali-journal-admin --region asia-northeast3   --remove-env-vars ADMIN_TOKEN   --update-secrets ADMIN_TOKEN=ADMIN_TOKEN:latest
```
- Runtime SA must read the secret: grant **roles/secretmanager.secretAccessor** to `quali-run-sa@quali-journal-prod.iam.gserviceaccount.com`.
- Load token without hidden CR/LF:
```
$TOKEN = (gcloud secrets versions access latest --secret=ADMIN_TOKEN --project quali-journal-prod | Out-String).Trim()
```

## 2) Smoke
- `/api/status`: **GET** only (no `-I`).
```
$URL = gcloud run services describe quali-journal-admin --region asia-northeast3 --format "value(status.url)"
curl.exe -fsS -i -H "Authorization: Bearer $TOKEN" "$URL/api/status"
```
- `/api/report`: **POST** with body length.
```
curl.exe -s -X POST -H "X-Admin-Token: $TOKEN" -H "Content-Length: 0" "$URL/api/report"
```

## 3) Cloud Scheduler (06:00 KST daily)
- Use `key=value` for headers (not `key: value`). If update fails, delete & recreate.
```
$JOB="quali-report-daily"
gcloud scheduler jobs delete $JOB --project "quali-journal-prod" --location "asia-northeast3" -q
gcloud scheduler jobs create http $JOB --project "quali-journal-prod" --location "asia-northeast3"   --schedule "0 6 * * *" --time-zone "Asia/Seoul"   --uri "$URL/api/report" --http-method POST   --headers "X-Admin-Token=$TOKEN"   --message-body "{}"
```

## 4) CI/CD — final `deploy.yml`
```yaml
name: Deploy to Cloud Run
on:
  push: { branches: [ "main" ] }
  workflow_dispatch:
permissions: { contents: read, id-token: write }
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Auth to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      - name: Write env.yaml
        run: |
          cat > env.yaml <<'EOF'
          ALLOWED_ORIGINS: "http://localhost:3000,http://127.0.0.1:3000"
          EOF
      - name: Deploy to Cloud Run
        if: github.event_name != 'pull_request'
        run: |
          set -eux
          gcloud config set project "quali-journal-prod"
          gcloud run deploy "quali-journal-admin"             --region "asia-northeast3"             --source "."             --service-account "quali-run-sa@quali-journal-prod.iam.gserviceaccount.com"             --env-vars-file "env.yaml"             --set-secrets "ADMIN_TOKEN=ADMIN_TOKEN:latest"             --allow-unauthenticated             --quiet
```

## 5) IAM quick map
- **GitHub deploy SA** `github-deploy@quali-journal-prod.iam.gserviceaccount.com`:
  - `roles/run.admin`, `roles/iam.serviceAccountUser`, `roles/serviceusage.serviceUsageAdmin`,
  - `roles/secretmanager.secretAccessor`, `roles/storage.admin` (can tighten later to bucket scope).
- **Cloud Build SA** `{PROJECT_NUMBER}@cloudbuild.gserviceaccount.com` ⇒ `roles/artifactregistry.writer`.
- **Runtime SA** `quali-run-sa@...` ⇒ `roles/secretmanager.secretAccessor`.

## 6) APIs enabled
`run, artifactregistry, cloudbuild, secretmanager, iamcredentials, cloudresourcemanager, serviceusage`

## 7) Error → Fix cheatsheet
- 401 Unauthorized → reload & Trim token; header typo check.
- 405 Method Not Allowed → use GET (not HEAD).
- 411 Length Required → add `-d ""` or `Content-Length: 0`.
- Permission denied on secret → grant Secret Accessor to runtime SA.
- `storage.objects.create` denied → give deploy SA Storage Admin (or bucket-level objectAdmin).
- `Unsupported service account` → pass **email**, not `projects/.../serviceAccounts/...`.

---

## 8) Quick rollback / rerun
```
gcloud run revisions list --service quali-journal-admin --region asia-northeast3 --sort-by="~metadata.creationTimestamp" --limit=2
$prev = (gcloud run revisions list --service quali-journal-admin --region asia-northeast3 --sort-by="~metadata.creationTimestamp" --format="value(metadata.name)" | Select-Object -Skip 1 -First 1)
gcloud run services update-traffic quali-journal-admin --region asia-northeast3 --to-revisions "$prev=100"
```


# QualiJournal Admin – Handover (Context Seed)
*Generated:* 2025-10-26 07:38:22 KST

## SSOT (Fixed)
- **Project:** `quali-journal-prod`
- **Source/Test Service (Seoul):** `quali-journal-admin` → Cloud Run @ `asia-northeast3` (run.app URL)
- **Live/Domain Service (Tokyo):** `quali-admin-domap` → Cloud Run @ `asia-northeast1` (`https://admin.standardai.co.kr`)
- **Artifact Registry Repo:** `quali` (region `asia-northeast3`)
- **Secret:** `ADMIN_TOKEN` from Secret Manager (`latest`)

## Current Code/Config Highlights
- **FastAPI** (server_quali.py) with protected routes (Bearer ADMIN_TOKEN)
- Tool APIs:
  - `POST /api/tools/repair` → repair JSON + rebuild selected
  - `POST /api/tools/approve_top?n=20` → auto-approve Top-N + sync
- **Admin UI** (index.html): Quick Tools (Approve/Repair) wired to above APIs
- **Data files**: `selected_keyword_articles.json` (working), `selected_articles.json` (selected)
- **Tools**: `sync_selected_for_publish.py`, `repair_selection_files.py`, `force_approve_top20.py`
- **CI/CD**: GitHub Actions (WIF) → build/push → deploy to **SRC** → smoke (health/status)

## Golden Workflow – Code → Live
1. **Commit to main** (server_quali.py / index.html / tools/*) → Actions must be green.
2. **Get SRC image (@sha256)**  
   ```powershell
   $IMG_SRC = (gcloud run services describe quali-journal-admin --region asia-northeast3 --format "value(spec.template.spec.containers[0].image)")
   ```
3. **Promote same image to LIVE (no source deploy)**  
   ```powershell
   gcloud run deploy quali-admin-domap `
     --image "$IMG_SRC" `
     --region asia-northeast1 --project quali-journal-prod `
     --platform managed --allow-unauthenticated `
     --set-secrets ADMIN_TOKEN=ADMIN_TOKEN:latest `
     --set-env-vars "PYTHONUTF8=1,PYTHONIOENCODING=utf-8"
   ```
4. **Shift traffic to latest & open with cache-bust**
   ```powershell
   gcloud run services update-traffic quali-admin-domap --to-latest --region asia-northeast1 --project quali-journal-prod
   $URL_LIVE = (gcloud run services describe quali-admin-domap --region asia-northeast1 --project quali-journal-prod --format "value(status.url)")
   Start-Process "$URL_LIVE/?v=$(Get-Random)"
   Start-Process "https://admin.standardai.co.kr/?v=$(Get-Random)"
   ```
5. **Live data sync (once per deploy if needed)**
   ```powershell
   $TOKEN=(gcloud secrets versions access latest --secret=ADMIN_TOKEN --project quali-journal-prod | Out-String).Trim()
   curl.exe -s -o NUL -w "%{http_code}`n" -H "Authorization: Bearer $TOKEN" "$URL_LIVE/api/tools/repair"
   curl.exe -s -o NUL -w "%{http_code}`n" -H "Authorization: Bearer $TOKEN" "$URL_LIVE/api/tools/approve_top?n=20"
   ```
6. **Smoke**
   ```powershell
   curl.exe -s -o NUL -w "%{http_code}`n" "$URL_LIVE/health"            # 200
   curl.exe -s -o NUL -w "%{http_code}`n" "$URL_LIVE/api/status"        # 401
   curl.exe -s -o NUL -w "%{http_code}`n" -H "Authorization: Bearer $TOKEN" "$URL_LIVE/api/status"  # 200
   ```

## Common Fixes
- **Container import failed** → grant `roles/artifactregistry.reader` to LIVE runtime SA on `quali` repo (`asia-northeast3`).
- **ADMIN_TOKEN type conflict** → in CI use `--set-secrets ADMIN_TOKEN=ADMIN_TOKEN:latest`; do not set as env var.
- **Domain shows old UI** → `update-traffic --to-latest` + open with `?v=$(Get-Random)`.
- **Ready is empty only on LIVE** → run `/api/tools/repair` + `/api/tools/approve_top` on LIVE once.

## Repo Layout (expected)
```
repo-root/
├─ server_quali.py
├─ index.html
├─ requirements.txt
├─ Dockerfile
├─ config.json
├─ selected_keyword_articles.json
├─ selected_articles.json
└─ tools/
   ├─ sync_selected_for_publish.py
   ├─ repair_selection_files.py
   └─ force_approve_top20.py
```

## Next Backlog
- Approve UI (table + checkbox + note), items API (GET/PATCH), pytest scenario, scheduler, token rotation.

# QualiJournal Admin — Deploy & Verify Playbook

> 목적: **항상 신버전 UI/백엔드가 Cloud Run에 정확히 반영**되도록, _한 번에_ 배포·검증·백업까지 끝내는 플레이북.

---

## 0) SSOT(단일 진실) 경로
- **항상 이 경로를 소스 폴더로 사용**:  
  `C:\Users\user\Desktop\퀄리저널\admin`

> 주의: `userW` 등 다른 계정 경로로 배포하면 **구버전 HTML**이 올라갈 수 있음.

---

## 1) HTML 포함 규칙(.gcloudignore / .dockerignore)
아래 3줄이 **반드시 포함**되어야 합니다(없으면 추가).

```
!index.html
!index_lite_black.html
!**/*.html
```

위 규칙은 HTML이 업로드에서 빠지지 않도록 보장합니다.

---

## 2) Cloud Run(소스) 배포 — Buildpacks + 엔트리포인트 고정

```powershell
$ADMIN="C:\Users\user\Desktop\퀄리저널\admin"
gcloud run deploy qualijournal-admin --source "$ADMIN" --region asia-northeast3 --allow-unauthenticated `
  --set-build-env-vars GOOGLE_ENTRYPOINT="uvicorn server_quali:app --host 0.0.0.0 --port 8080" `
  --set-env-vars "ADMIN_TOKEN=$TOKEN,PYTHONUTF8=1,PYTHONIOENCODING=utf-8"
```

- `ADMIN_TOKEN`은 운영용 32자 토큰(예: `echo $TOKEN` 결과)을 사용
- 위 명령은 **uvicorn server_quali:app**으로 기동을 고정합니다.

---

## 3) 신판 HTML 적용 확인 (캐시 무시 검사)

```powershell
$U=$(gcloud run services describe qualijournal-admin --region asia-northeast3 --format="value(status.url)")
if ((curl.exe -s "$U/?v=$(Get-Random)") -like "*ADMIN_TOKEN*") {"NEW ✅"} else {"OLD ❌"}
```

- `NEW ✅` → 신판 적용 확인
- `OLD ❌` → 0단계 경로/1단계 포함 규칙을 재확인 후 재배포

---

## 4) 백엔드 스모크(필수 API)

```powershell
# 상태
curl.exe -I -H "Authorization: Bearer $TOKEN" "$U/api/status"

# 내보내기
curl.exe -I -H "Authorization: Bearer $TOKEN" "$U/api/export/md"
curl.exe -I -H "Authorization: Bearer $TOKEN" "$U/api/export/csv"
```

- 각 명령이 `HTTP/1.1 200 OK`면 정상

---

## 5) 일일 백업(로컬 저장)

```powershell
$base=Join-Path $HOME "Desktop\퀄리저널\backup"; $d=Get-Date -Format "yyyyMMdd_HHmmss"; $dir=Join-Path $base $d
New-Item -ItemType Directory -Path $dir -Force | Out-Null
curl.exe -fSL -H "Authorization: Bearer $TOKEN" "$U/api/export/md"  -o (Join-Path $dir ("export_"+$d+".md"))
curl.exe -fSL -H "Authorization: Bearer $TOKEN" "$U/api/export/csv" -o (Join-Path $dir ("export_"+$d+".csv"))
Write-Host "백업 완료 → $dir"
```

> 선택: GCS 업로드 자동화는 추후 스크립트 추가

---

## 6) 문제 해결(Plan‑B: 도커 이미지 고정 배포)

```powershell
$ADMIN="C:\Users\user\Desktop\퀄리저널\admin"
$PROJECT=(gcloud config get-value project)
$IMG="asia-northeast3-docker.pkg.dev/$PROJECT/quali/qualijournal-admin:stable"
gcloud artifacts repositories create quali --repository-format=docker --location=asia-northeast3 2>$null

# Dockerfile(HTML 포함, uvicorn 고정)
@'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
CMD ["uvicorn","server_quali:app","--host","0.0.0.0","--port","8080"]
'@ | Set-Content "$ADMIN\Dockerfile" -Encoding utf8

# (안전장치) .dockerignore에 HTML 포함 규칙
@'
.git
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
*.zip
*.7z
*.log
logs/
archive/**
!index.html
!index_lite_black.html
!**/*.html
'@ | Set-Content "$ADMIN\.dockerignore" -Encoding utf8

gcloud builds submit --tag $IMG $ADMIN
gcloud run deploy qualijournal-admin --image $IMG --region asia-northeast3 --allow-unauthenticated `
  --set-env-vars "ADMIN_TOKEN=$TOKEN,PYTHONUTF8=1,PYTHONIOENCODING=utf-8"
```

배포 후 **3) 신판 확인**, **4) 스모크**를 다시 수행합니다.

---

## 7) 운영 루틴(매일 아침 1분)

1. `/api/status` 200 확인  
2. 필요 시 **MD/CSV 내보내기** → 5) 백업 스크립트 실행  
3. UI에서 토큰 칩이 **인증됨(초록)**임을 확인  
4. 에러 발생 시 Cloud Run 로그 확인 → 필요 시 Plan‑B

---

## 부록) 빠른 체크 3줄

```powershell
$U=$(gcloud run services describe qualijournal-admin --region asia-northeast3 --format="value(status.url)")
if ((curl.exe -s "$U/?v=$(Get-Random)") -like "*ADMIN_TOKEN*") {"UI: NEW ✅"} else {"UI: OLD ❌"}
curl.exe -I -H "Authorization: Bearer $TOKEN" "$U/api/export/md"  | Select-String "HTTP/1.1"
curl.exe -I -H "Authorization: Bearer $TOKEN" "$U/api/export/csv" | Select-String "HTTP/1.1"
```

---

### 메모(개발자용)
- `server_quali:app`을 엔트리포인트로 **항상 고정**  
- 필요 시 `/` 응답에 `Cache-Control: no-store` 헤더 유지  
- 토큰: 운영 SECRET과 동일한 32자 GUID(hex) 사용

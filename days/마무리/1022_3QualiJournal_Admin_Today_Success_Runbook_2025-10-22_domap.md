# ✅ QualiJournal Admin — 오늘 성공 런북 (domap 서비스)

작성 시각: **2025-10-22 02:43:24 KST**

---

## TL;DR
- GitHub Actions **Deploy to Cloud Run** 성공 → 대상 서비스 **quali-admin-domap**.
- Cloud Run 최신 리비전으로 **트래픽 100% 전환** 완료.
- 도메인 **admin.standardai.co.kr** 에서 캐시 무시로 새 HTML 확인, 버튼/이벤트 정상 동작.
- 스모크 3종 통과: `/api/status` **200(토큰)/401(무토큰)**, `/api/report` **200**.

---

## 증거 (Evidence)
- 도메인에서 받은 `deployed_index_domain.html`에 **새 UI 스크립트**(fetch 래퍼로 Authorization 헤더 자동 부착, 토큰/자동 새로고침, Admin Ops wiring 등) 포함 확인.
- 런타임/빌드 배지 노출, 상단 `ADMIN_TOKEN` 바 및 버튼 wiring 코드 존재.

> 참고: 브라우저 캐시를 무시하여 받은 HTML 파일로 검증했고, 이후 버튼 클릭 시 `/api/status`가 매번 200으로 응답됨. (무토큰일 때 401)

---

## 변수(한 번만 선언)
```powershell
$PROJECT      = "quali-journal-prod"
$REGION       = "asia-northeast3"        # Cloud Run 서비스 리전
$REGION_DM    = "asia-northeast1"        # 도메인 매핑 리전
$DOMAIN       = "admin.standardai.co.kr"
$SERVICE_SRC  = "quali-journal-admin"    # 최신 이미지가 먼저 올라가는 서비스
$SERVICE_LIVE = "quali-admin-domap"      # 도메인(routeName)이 가리키는 라이브 서비스
```

---

## 1) 도메인 매핑이 가리키는 서비스 확인
```powershell
gcloud beta run domain-mappings describe --domain $DOMAIN --region $REGION_DM --format="value(spec.routeName)"
# 출력이 'quali-admin-domap' 이면 OK
```

## 2) 소스 서비스에서 **현재 이미지 URI** 추출
```powershell
$IMG = gcloud run services describe $SERVICE_SRC --region $REGION --project $PROJECT `
  --format="value(spec.template.spec.containers[0].image)"
$IMG  # 예: asia-northeast3-docker.pkg.dev/.../qualijournal-admin:stable
```

## 3) 그 이미지를 **라이브 서비스(domap)** 에 배포
```powershell
gcloud run deploy $SERVICE_LIVE --image $IMG `
  --region $REGION --project $PROJECT --platform managed --allow-unauthenticated `
  --set-secrets "ADMIN_TOKEN=ADMIN_TOKEN:latest" `
  --set-env-vars "PYTHONUTF8=1,PYTHONIOENCODING=utf-8"
```

## 4) 최신 리비전에 **트래픽 100%**
```powershell
gcloud run services update-traffic $SERVICE_LIVE --to-latest --region $REGION --project $PROJECT
```

## 5) 도메인에서 **캐시 무시로 새 HTML 확인**
```powershell
Invoke-WebRequest "https://$DOMAIN/?v=$(Get-Random)" -Headers @{"Cache-Control"="no-cache"} -OutFile "deployed_index_domain.html"
notepad .\deployed_index_domain.html
# (선택) 새 버튼/데이터-액션 키워드 확인
Select-String -Path .\deployed_index_domain.html -Pattern 'btn-token-set|btn-token-clear|data-action|ADMIN_TOKEN'
```

## 6) 스모크 3종 (30초)
```powershell
$URL   = gcloud run services describe $SERVICE_LIVE --region $REGION --project $PROJECT --format="value(status.url)"
$TOKEN = (gcloud secrets versions access latest --secret=ADMIN_TOKEN --project $PROJECT | Out-String).Trim()

# 상태: 토큰 O → 200
curl.exe -s -o NUL -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" "$URL/api/status"

# 상태: 토큰 X → 401
curl.exe -s -o NUL -w "%{http_code}\n" "$URL/api/status"

# 보고서: 토큰 O → 200
curl.exe -s -o NUL -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" "$URL/api/report"
```

---

## GitHub Actions **서비스명 고정** (드리프트 방지)
`.github/workflows/deploy.yml` 의 `env:` 블록을 다음과 같이 유지:
```yaml
env:
  PROJECT_ID: quali-journal-prod
  REGION: asia-northeast3
  SERVICE_NAME: quali-admin-domap
  ENTRYPOINT: "uvicorn server_quali:app --host 0.0.0.0 --port 8080"
```
배포 step이 소스 배포를 사용할 경우 아래 라인을 사용해 **엔트리포인트 전달**:
```bash
--set-build-env-vars "GOOGLE_ENTRYPOINT=${ env.ENTRYPOINT }"
```
> 이미지 배포(Plan‑B)만 쓴다면 `ENTRYPOINT` 전달 줄은 생략 가능(도커파일 CMD로 고정).

---

## 일일 점검(1분 자동 체크 스니펫)
```powershell
$URL=$(gcloud run services describe $SERVICE_LIVE --region $REGION --project $PROJECT --format="value(status.url)")
$TOKEN=(gcloud secrets versions access latest --secret=ADMIN_TOKEN --project $PROJECT | Out-String).Trim()
echo HEALTH: $(curl.exe -s -o NUL -w "%{http_code}" "$URL/health") `
     STATUS: $(curl.exe -s -o NUL -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$URL/api/status") `
     REPORT: $(curl.exe -s -o NUL -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$URL/api/report")
# 기대값: HEALTH=200  STATUS=200  REPORT=200  (무토큰 status=401)
```

---

## 장애 재현 시 빠른 복구(2줄)
```powershell
gcloud run services update-traffic $SERVICE_LIVE --to-latest --region $REGION --project $PROJECT
Invoke-WebRequest "https://$DOMAIN/?v=$(Get-Random)" -Headers @{"Cache-Control"="no-cache"} -OutFile "deployed_index_domain.html"
```

---

## 부록: 오늘 성공 포인트 메모
- **도메인 매핑**이 가리키는 서비스가 `quali-admin-domap`임을 확정했고, 이 서비스에 **최신 이미지 재배포** 후 트래픽 100%로 해결.
- 브라우저 **캐시 무시**로 새 HTML 확인 완료 → 버튼/자동 새로고침/토큰 UI 정상.
- 이후 **Actions의 SERVICE_NAME=quali-admin-domap**으로 고정하여 드리프트 방지.

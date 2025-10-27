
# ✅ Cloud Run 트래픽 최신화 & PowerShell 따옴표 오류 방지 런북

작성: **2025-10-22 03:13:54 KST**

---

## TL;DR
- **증상**: `gcloud run services update-traffic ...` 실행 시 `argument --project: expected one argument` 등 문법 오류.
- **원인**: PowerShell에서 **홑따옴표(')** 로 `'$SERVICE_NAME'`/`'$PROJECT_ID'` 등을 감싸면 **변수 치환이 안 됨** → 값이 비어 전달됨.
- **정답**: 홑따옴표 제거 또는 **쌍따옴표(")** 사용. 아래 “정답 명령”을 그대로 복붙.

---

## 1) 정답 명령 (PowerShell)

```powershell
# 변수 확인/선언
$SERVICE_NAME = "quali-admin-domap"
$REGION       = "asia-northeast3"
$PROJECT_ID   = "quali-journal-prod"

# 최신 리비전에 트래픽 100% (변수 확장 O)
gcloud run services update-traffic $SERVICE_NAME --to-latest --region $REGION --project $PROJECT_ID
```

값을 직접 넣는 버전:
```powershell
gcloud run services update-traffic "quali-admin-domap" --to-latest --region "asia-northeast3" --project "quali-journal-prod"
```

**주의** (금지 예시):
```powershell
# ❌ 변수 확장 안 됨 → 그대로 문자열로 전달됨
gcloud run services update-traffic '$SERVICE_NAME' --to-latest --region '$REGION' --project '$PROJECT_ID'
```

---

## 2) 적용 확인 (10초)

```powershell
# 트래픽이 최신 리비전에 100%인지
gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format="value(status.traffic)"
# 기대 출력: [{'latestRevision': True, 'percent': 100}]
```

필요 시 최신 리비전 이름도 확인:
```powershell
gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format="value(status.latestReadyRevisionName)"
```

---

## 3) 도메인 매핑과 서비스 일치 점검

도메인이 **어느 서비스(routeName)** 를 가리키는지 먼저 확정:
```powershell
$DOMAIN    = "admin.standardai.co.kr"
$REGION_DM = "asia-northeast1"  # 도메인 매핑 리전
gcloud beta run domain-mappings describe --domain $DOMAIN --region $REGION_DM --format="value(spec.routeName)"
# 출력 예: quali-admin-domap
```

- 도메인이 `quali-admin-domap`을 가리킨다면, **해당 서비스**에 배포/트래픽 최신화를 수행해야 화면이 바뀝니다.

---

## 4) 배포 후 캐시 무시로 최신 UI 확인

```powershell
Invoke-WebRequest "https://$DOMAIN/?v=$(Get-Random)" -Headers @{"Cache-Control"="no-cache"} -OutFile "deployed_index_domain.html"
# 새 버튼/데이터-액션 존재 확인
Select-String -Path .\deployed_index_domain.html -Pattern 'btn-token-set|btn-token-clear|data-action'
```

---

## 5) 스모크 3종 (매일 30초 점검)

```powershell
$URL   = gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format="value(status.url)"
$TOKEN = (gcloud secrets versions access latest --secret=ADMIN_TOKEN --project $PROJECT_ID | Out-String).Trim()

# curl 버전 (PowerShell은 write-out에 홑따옴표 사용)
curl.exe -s -o NUL -w '%{http_code}' "$URL/health"
curl.exe -s -o NUL -w '%{http_code}' -H "Authorization: Bearer $TOKEN" "$URL/api/status"
curl.exe -s -o NUL -w '%{http_code}' -H "Authorization: Bearer $TOKEN" "$URL/api/report"

# PowerShell 순정 버전
(Invoke-WebRequest "$URL/health").StatusCode
(Invoke-WebRequest "$URL/api/status" -Headers @{Authorization="Bearer $TOKEN"}).StatusCode
(Invoke-WebRequest "$URL/api/report" -Headers @{Authorization="Bearer $TOKEN"}).StatusCode
```

**기대값**: HEALTH=200, STATUS=200(토큰)/401(무토큰), REPORT=200

---

## 6) GitHub Actions 드리프트 방지

`.github/workflows/deploy.yml`의 `env:`는 다음처럼 유지:
```yaml
env:
  PROJECT_ID: quali-journal-prod
  REGION: asia-northeast3
  SERVICE_NAME: quali-admin-domap
  ENTRYPOINT: "uvicorn server_quali:app --host 0.0.0.0 --port 8080"
```

배포 후 트래픽을 자동 최신화하려면 스텝 추가:
```yaml
- name: Force traffic to latest
  if: github.ref == 'refs/heads/main'
  run: |
    gcloud run services update-traffic "$SERVICE_NAME" --to-latest --region "$REGION" --project "$PROJECT_ID"
```

---

## 7) 응급 복구 2줄

```powershell
gcloud run services update-traffic $SERVICE_NAME --to-latest --region $REGION --project $PROJECT_ID
Invoke-WebRequest "https://$DOMAIN/?v=$(Get-Random)" -Headers @{"Cache-Control"="no-cache"} -OutFile "deployed_index_domain.html"
```

---

## 메모 (재발 방지 키워드)
- PowerShell **홑따옴표(')** = 변수 미확장, **쌍따옴표(")** = 변수 확장.
- 도메인 매핑 **routeName** = 실제로 사용자가 보는 **라이브 서비스**.
- 배포 후 **트래픽 최신화 + 캐시 무시** = 새 UI 보장.

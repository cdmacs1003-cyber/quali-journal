
# QualiJournal 관리자 시스템 — 최종 운영 Runbook (Snapshot v1.1)
**작성 시각:** 2025-10-19 08:57:57 UTC+09:00

> 본 문서는 **오늘 상태 스냅샷**입니다. 운영에 필요한 SSOT 값·점검 절차·복구 플랜을 한 파일에 모았습니다.  
> 대상: 운영자(일일 점검), 개발자(CI/CD·도메인·시크릿 관리)

---

## 0) 한눈에 보는 요약
- **서비스(도메인용)**: `quali-admin-domap` (Cloud Run, `asia-northeast1`)
- **커스텀 도메인**: `https://admin.standardai.co.kr`
- **도메인 매핑**: `admin.standardai.co.kr` → **CNAME** → `ghs.googlehosted.com`
- **스케줄러**: `daily-report` (`asia-northeast1`, 매일 09:00 Asia/Seoul, `POST /api/report`, 헤더 `X-Admin-Token`)
- **시크릿**: `ADMIN_TOKEN` (Secret Manager, `latest` 사용)
- **상태(스모크)**: `/health` 200, `/api/status` 200, `/api/report` 200  — 2025‑10‑19 확인
- **Seoul(asia‑northeast3) 한계**: Cloud Run **도메인 매핑 미지원** → 도쿄 리전에 “도메인 전용” 서비스로 대응
- **보안 기본**: 공개 엔드포인트는 헬스만, 관리 API는 토큰 헤더 필수

---

## 1) SSOT(단일 출처 값)
- **프로젝트**: `quali-journal-prod`
- **리전**
  - 도메인용 Cloud Run: `asia-northeast1`
  - 스케줄러: `asia-northeast1`
- **서비스 이름**
  - 도메인용: `quali-admin-domap`
- **도메인**
  - 베이스: `standardai.co.kr` (Search Console 검증 완료)
  - 연결: `admin.standardai.co.kr` (CNAME→`ghs.googlehosted.com`)
- **시크릿**
  - 이름: `ADMIN_TOKEN` (Secret Manager, `latest` 사용; 값 노출 금지)

---

## 2) 운영자 “3줄 스모크” (하루 1회)
```powershell
$DOMAIN="admin.standardai.co.kr"
$PROJECT_ID="quali-journal-prod"
$TOKEN=(gcloud secrets versions access latest --secret=ADMIN_TOKEN --project $PROJECT_ID | Out-String).Trim()

curl.exe -s -D - "https://$DOMAIN/health" | findstr /R "^HTTP/.* 200"
curl.exe -s -D - -H "X-Admin-Token: $TOKEN" "https://$DOMAIN/api/status"  | findstr /R "^HTTP/.* 200"
curl.exe -s -D - -H "X-Admin-Token: $TOKEN" "https://$DOMAIN/api/report"  | findstr /R "^HTTP/.* 200"
```

---

## 3) 스케줄러(일일 보고서)
- 잡 이름: `daily-report` (`asia-northeast1`)
- 동작: 매일 09:00 KST, `POST /api/report`, 헤더 `X-Admin-Token: <ADMIN_TOKEN>`
- 상태 확인 & 즉시 실행
```powershell
$LOCATION="asia-northeast1"
gcloud scheduler jobs describe daily-report --location $LOCATION | findstr /I "state lastAttempt lastScheduleTime"
gcloud scheduler jobs run daily-report --location $LOCATION
```

- 도메인 인증 완료 후 URI 교체(이미 적용되어 있으면 생략)
```powershell
gcloud scheduler jobs update http daily-report --location asia-northeast1 --uri "https://admin.standardai.co.kr/api/report"
```

---

## 4) 시크릿(ADMIN_TOKEN) 회전 절차
> 노출 의심·정기 교체 시 즉시 실행. Cloud Run은 `latest`를 참조하므로 서비스 재배포 불필요.
```powershell
$PROJECT_ID="quali-journal-prod"
$NEW=[Convert]::ToBase64String([Guid]::NewGuid().ToByteArray()) + (Get-Random)
$NEW | gcloud secrets versions add ADMIN_TOKEN --project $PROJECT_ID --data-file=-

# 스케줄러 헤더 갱신
gcloud scheduler jobs update http daily-report --location asia-northeast1 --headers "X-Admin-Token=$NEW,Content-Length=0"
```

---

## 5) 도메인 매핑 운용
### 5.1 현재 값 확인
```powershell
$DOMAIN="admin.standardai.co.kr"; $REGION="asia-northeast1"
$dm = gcloud beta run domain-mappings describe --domain $DOMAIN --region $REGION --format="json" | ConvertFrom-Json
$dm.status.resourceRecords | Format-Table type,name,rrdata
$dm.status.conditions      | Format-Table type,status,reason,message
```
- 기대: `CNAME admin.standardai.co.kr → ghs.googlehosted.com`, `Ready=True`, `Certificate* = True`

### 5.2 인증서 지연 시
```powershell
# 전역 DNS 확인(둘 다 CNAME=ghs.googlehosted.com이면 전파 OK)
nslookup -type=cname admin.standardai.co.kr 8.8.8.8
nslookup -type=cname admin.standardai.co.kr 1.1.1.1

# 매핑 재생성(안전; 트래픽 영향 없음)
gcloud beta run domain-mappings delete  --domain admin.standardai.co.kr --region asia-northeast1 -q
gcloud beta run domain-mappings create  --service quali-admin-domap --domain admin.standardai.co.kr --region asia-northeast1
```

### 5.3 대안(필요 시)
- HTTP(S) Load Balancer + **Serverless NEG**로 라우팅. (다중 리전·고급 설정 필요 시 선택)

---

## 6) CI/CD·WIF 핵심 체크(요약)
- GitHub Actions 워크플로우 `permissions:`  
  - `contents: read`, `id-token: write`
- Workload Identity Federation
  - Provider: `github-oidc`(main), `gha-pr`(PR) — **enabled**
  - SA 권한
    - GitHub SA: `roles/run.admin`, `roles/secretmanager.secretAccessor`, `roles/serviceusage.serviceUsageAdmin`
    - Cloud Build SA: `roles/artifactregistry.writer`, `roles/storage.admin`
  - 빠른 점검(발췌):
```powershell
gcloud iam workload-identity-pools providers describe github-oidc --workload-identity-pool github-wif --location global --project quali-journal-prod --format="yaml(name, disabled, oidc.issuerUri, attributeCondition)"
gcloud projects get-iam-policy quali-journal-prod --flatten="bindings[].members" --format="table(bindings.role,bindings.members)" | Select-String "github-deploy@"
```

---

## 7) 장애 대응 플레이북(Top 6)
- **/health 405** → `curl -I`(HEAD) 대신 `GET` 사용.
- **/api/status 401** → 헤더 키 확인(`X-Admin-Token` 또는 `Authorization: Bearer`), `ADMIN_TOKEN` 최신값 재주입.
- **도메인 매핑 501(서울 리전)** → 지원 리전(`asia-northeast1`)에 “도메인용” 서비스로 구성.
- **도메인 Ready/Cert=Unknown 지속** → 전역 DNS 확인 후 매핑 삭제→재생성.
- **Actions 422** → `gh workflow run ... --ref main` 로 수정.
- **SECRET 교체 후 스케줄러 실패** → `jobs update http ... --headers "X-Admin-Token=<NEW>"` 재적용.

---

## 8) 백업·로깅(권장)
- **백업**: GCS 버저닝+보존정책(예: 365d) 활성화 후, 내보내기(MD/CSV) 업로드 스케줄링.
- **로그**: Cloud Run 로그에서 4xx/5xx 검색, 스케줄러 실행 로그 주간 점검.

---

## 9) 부록 — 붙여넣기 블록
### 9.1 도메인 인증서 폴링
```powershell
$DOMAIN="admin.standardai.co.kr"; $REGION="asia-northeast1"
for($i=1;$i -le 36; $i++){
  $dm = gcloud beta run domain-mappings describe --domain $DOMAIN --region $REGION --format="json" | ConvertFrom-Json
  $ready = ($dm.status.conditions | ? {{$_.type -eq "Ready"}}).status
  $cert  = ($dm.status.conditions | ? {{$_.type -eq "CertificateProvisioned" -or $_.type -eq "CertificateReady"}}).status
  Write-Host ("[{0}] Ready={1}  Cert={2}" -f $i,$ready,$cert)
  if($ready -eq "True" -and $cert -eq "True"){ break }
  Start-Sleep -Seconds 10
}
```

### 9.2 스케줄러 생성(완전체)
```powershell
$PROJECT_ID="quali-journal-prod"; $REGION_RUN="asia-northeast1"; $LOCATION="asia-northeast1"; $SERVICE="quali-admin-domap"
gcloud services enable cloudscheduler.googleapis.com cloudtasks.googleapis.com --project $PROJECT_ID
$TOKEN=(gcloud secrets versions access latest --secret=ADMIN_TOKEN --project $PROJECT_ID | Out-String).Trim()
$U=gcloud run services describe $SERVICE --region $REGION_RUN --format="value(status.url)"
gcloud scheduler jobs delete daily-report --location $LOCATION -q 2>$null
gcloud scheduler jobs create http daily-report --location $LOCATION --schedule "0 9 * * *" --time-zone "Asia/Seoul" --http-method POST --uri "$U/api/report" --headers "X-Admin-Token=$TOKEN,Content-Length=0"
```

---

## 10) 변경 이력
- **v1.1 (2025‑10‑19)**: 도쿄 리전 도메인용 서비스 배포, 커스텀 도메인 매핑, 스케줄러 생성, 스모크 200 OK 확인, Runbook 스냅샷 작성.

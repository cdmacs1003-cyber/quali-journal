# QualiJournal Admin — Cloud Run/운영 통합 가이드북 (SSOT 풀버전 v1.0)

> 이 문서는 **QualiJournal 관리자 시스템(prod 환경)**에 대해,  
> 프로젝트/서비스/리전/버킷 등 *조직 내 실제 값*을 모두 채운 **SSOT 기반 실행 가이드북**입니다.  
> 신규 운영자·개발자가 이 파일 하나만 보고도, **배포 · 도메인 · 스케줄러 · 백업 · 스모크 테스트**를 재현할 수 있도록 설계되었습니다.

---

## 0. 문서 위치와 활용 방식

### 0.1 이 문서가 다루는 범위

- 대상 시스템: `QualiJournal` 관리자 시스템 (Admin 패널 + API)
- 환경: **프로덕션 GCP 프로젝트 `quali-journal-prod`**
- 구성 요소:
  - Cloud Run 서비스 2개
    - `quali-journal-admin` (서울, 소스 빌드용)
    - `quali-admin-domap` (도쿄, 도메인 연결용 라이브)
  - 커스텀 도메인: `https://admin.standardai.co.kr`
  - Cloud Scheduler: `daily-report` (일일 키워드 보고서 자동 실행)
  - GCS 버킷: `gs://qualijournal-archive` (발행본/백업 아카이브)
  - Secret Manager: `ADMIN_TOKEN` (관리자 인증 토큰)

이 문서는 위 구성 요소에 대한 **배포·운영 표준 절차와 명령어**를 모두 포함합니다.  
GitHub, Dockerfile, 애플리케이션 코드 등은 별도의 코드 레포지토리를 참조하되, **인프라 관점 SSOT**는 이 문서를 기준으로 합니다.

### 0.2 대상 독자

- 신규/기존 **운영자(Ops)** — 일일 스모크·스케줄러·백업·장애 대응을 담당
- **백엔드/플랫폼 개발자** — 코드 수정 후 재배포, IAM/보안 설정 조정
- **SRE/DevOps 담당자** — CI/CD·Cloud Run·WIF·도메인·DNS·Artifact Registry를 관리

개발자가 코드를 바꾸더라도, *인프라 값(프로젝트/리전/서비스/버킷/도메인/스케줄러 이름)*은 이 파일 내용을 정답으로 봅니다.

### 0.3 다른 문서와의 관계

- 이 문서는 **“Custom Template” 기반 풀버전 가이드북**입니다.
- 기존 문서가 “왜(Why)”와 “배경”까지 자세히 설명했다면, 이 문서는 **“무엇을(What) 어떻게(How)”**에 집중합니다.
- 정책 변경이나 리소스 이름 변경이 있을 경우, *반드시 이 문서부터 업데이트*하고, 다른 문서에서는 이 파일을 참조하도록 유지합니다.

---

## 1. 시스템 개요 및 전체 워크플로우

### 1.1 QualiJournal 개념 요약

- 하루에 **하나의 키워드**를 선정합니다. (예: `IPC-A-610`, `ESA 우주선`, `반도체 공급망`)
- 관리자 페이지에서 키워드를 기준으로 다음을 수행합니다.
  1. 뉴스/블로그/학술/커뮤니티 등 원문 수집
  2. 요약/번역/스코어링
  3. 편집자 승인/코멘트 입력
  4. Markdown/CSV 형식의 **하루치 특집 페이지** 생성
  5. 결과물을 GCS 버킷에 백업·아카이브

관리자 시스템은 이 전체 과정을 **한 화면에서, 단일 API 세트로** 다룰 수 있게 설계되어 있습니다.

### 1.2 Admin 시스템의 역할

- **편집자/운영자 입장**
  - 키워드를 입력하고 버튼을 클릭하는 것만으로 **수집 → 요약 → 승인 → 발행 → 백업**을 실행합니다.
  - 실패 시 에러 메시지·로그를 보고 재시도하거나, 개발자에게 문의합니다.
- **개발자/플랫폼 입장**
  - FastAPI 기반 백엔드 + 정적 HTML/JS 어드민 UI를 하나의 컨테이너로 빌드해 Cloud Run에 배포합니다.
  - Cloud Scheduler와 Secret Manager, GCS를 활용해 **완전 자동화된 일일 발행 파이프라인**을 유지합니다.

---

## 2. SSOT(단일 출처 값) 요약

### 2.1 GCP / Cloud Run

- **프로젝트 ID**
  - `quali-journal-prod`
- **Cloud Run 리전**
  - 소스(관리) 서비스: `asia-northeast3`  _(서울)_
  - 도메인 매핑용 라이브 서비스: `asia-northeast1`  _(도쿄)_
- **Cloud Run 서비스 이름**
  - 소스 빌드용: `quali-journal-admin`
  - 도메인 연결 라이브: `quali-admin-domap`
- **커스텀 도메인**
  - 서비스 도메인: `https://admin.standardai.co.kr`
  - DNS: `CNAME admin.standardai.co.kr → ghs.googlehosted.com`

### 2.2 Secret / 환경 변수

- **Secret Manager**
  - 관리자 토큰: Secret 이름 `ADMIN_TOKEN`, 사용 버전 `latest`
  - 실제 토큰 문자열은 **Secret Manager에서만 조회**하며, 어떤 문서에도 값 자체를 적지 않습니다.
- **주요 환경 변수 (Cloud Run 컨테이너 기준)**
  - `ADMIN_TOKEN` : Secret Manager 시크릿을 컨테이너 환경 변수로 주입
  - `API_KEY` : (선택) 외부 요약/번역 API 키
  - `DB_URL` : (선택) DB 연결 문자열
  - `GCS_BUCKET` : `qualijournal-archive`
  - `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` : 파이썬 UTF-8 출력을 위해 사용

### 2.3 Cloud Scheduler (일일 보고서)

- **잡 이름**: `daily-report`
- **리전**: `asia-northeast1`
- **스케줄**: 매일 09:00 (Asia/Seoul)
- **요청 URL**: `https://admin.standardai.co.kr/api/report`
- **인증 방식**: HTTP 헤더 `X-Admin-Token: <ADMIN_TOKEN 현재값>`

### 2.4 GCS 버킷 (아카이브/백업)

- **버킷 이름**: `qualijournal-archive`
- **권장 정책**
  - 객체 버전 관리: **ON**
  - 보존 기간: **365일(1년)**
- **용도**
  - 발행된 Markdown/CSV 파일 영구 보관
  - 복구 리허설 및 장애 시 롤백에 활용

---

## 3. 사전 준비 및 공통 PowerShell 설정

### 3.1 필수 요구사항

1. **로컬 환경**
   - Windows 10/11 + PowerShell 5.1 이상
   - 최신 **gcloud CLI**, **gsutil** 설치 및 `gcloud auth login` 완료
2. **GCP 권한**
   - `quali-journal-prod` 프로젝트에 대한 최소 권한
     - Cloud Run Viewer/Developer
     - Cloud Scheduler Admin (또는 Job Admin 수준)
     - Secret Manager Secret Accessor
     - Storage Object Admin (또는 대상 버킷 권한)
3. **소스 코드 접근**
   - QualiJournal 레포지토리 read 권한
   - `admin/` 디렉터리에 접근 가능해야 함

### 3.2 PowerShell 공통 변수 초기화

> 아래 블록은 **모든 작업의 공통 헤더**로 사용합니다. 세션 시작 시 1회 실행을 권장합니다.

```powershell
# 공통 SSOT 변수
$PROJECT   = "quali-journal-prod"
$REG_SEO   = "asia-northeast3"    # 서울 (소스 서비스 리전)
$REG_TYO   = "asia-northeast1"    # 도쿄 (도메인 매핑 리전)

$SRC       = "quali-journal-admin"  # 소스 빌드용 Cloud Run 서비스
$LIVE      = "quali-admin-domap"    # 도메인 연결 라이브 서비스
$DOMAIN    = "admin.standardai.co.kr"
$BUCKET    = "qualijournal-archive"

# 관리자 토큰은 Secret Manager에서만 조회 (값은 문서에 기재하지 않음)
$TOKEN = (gcloud secrets versions access latest `
  --secret=ADMIN_TOKEN `
  --project $PROJECT | Out-String).Trim()
```

### 3.3 자주 사용하는 진단 명령 모음

```powershell
# 현재 프로젝트 확인
gcloud config get-value project

# Cloud Run 서비스 URL 확인
gcloud run services describe $LIVE `
  --region  $REG_TYO `
  --project $PROJECT `
  --format "value(status.url)"

# Secret Manager에 저장된 ADMIN_TOKEN 최신 버전 확인 (값은 화면에 노출하지 말 것)
gcloud secrets versions list ADMIN_TOKEN --project $PROJECT
```

---

## 4. 표준 운영 시나리오

이 섹션은 **실제 업무 흐름** 기준으로, “상황별로 무엇을 어떻게 할지”를 정리합니다.

### 4.1 신규 환경 초기화 / 재구축 시나리오

> “처음부터 다시 깔아야 한다”는 가정에서, **완전 초기 상태 → 운영 가능 상태**까지의 풀 세트 절차입니다.

#### 4.1.1 코드 준비

1. 레포지토리 클론 또는 최신 main 브랜치 업데이트
   ```powershell
   cd "C:\path\to\repo"
   git checkout main
   git pull origin main
   ```

2. `admin` 디렉터리로 이동
   ```powershell
   cd ".\admin"
   ```

#### 4.1.2 소스 서비스 배포 (`quali-journal-admin`, 서울)

```powershell
cd "C:\path\to\repo\admin"

gcloud run deploy $SRC `
  --source . `
  --project $PROJECT `
  --region  $REG_SEO `
  --platform managed `
  --allow-unauthenticated `
  --set-secrets "ADMIN_TOKEN=ADMIN_TOKEN:latest" `
  --set-env-vars "PYTHONUTF8=1,PYTHONIOENCODING=utf-8,GCS_BUCKET=$BUCKET"
```

- 기대 결과
  - 새로운 리비전이 생성되고, `status.url`에 서울 리전용 URL이 설정됩니다.
  - Cloud Run 콘솔에서 컨테이너 로그에 오류가 없어야 합니다.

#### 4.1.3 라이브 서비스 배포 (`quali-admin-domap`, 도쿄)

1. 소스 서비스에서 현재 사용 중인 이미지 URI 추출
   ```powershell
   $IMG = gcloud run services describe $SRC `
     --region  $REG_SEO `
     --project $PROJECT `
     --format  "value(spec.template.spec.containers[0].image)"

   $IMG  # 확인용 출력
   ```

2. 동일 이미지를 라이브 서비스에 배포
   ```powershell
   gcloud run deploy $LIVE `
     --image   $IMG `
     --region  $REG_TYO `
     --project $PROJECT `
     --platform managed `
     --allow-unauthenticated `
     --set-secrets "ADMIN_TOKEN=ADMIN_TOKEN:latest" `
     --set-env-vars "PYTHONUTF8=1,PYTHONIOENCODING=utf-8,GCS_BUCKET=$BUCKET"
   ```

3. 최신 리비전에 100% 트래픽 전환
   ```powershell
   gcloud run services update-traffic $LIVE `
     --to-latest `
     --region  $REG_TYO `
     --project $PROJECT
   ```

#### 4.1.4 도메인 매핑 설정

> 이미 매핑되어 있다면 이 단계는 넘어가도 됩니다. 재구성이 필요한 경우만 실행합니다.

```powershell
gcloud beta run domain-mappings create `
  --service $LIVE `
  --domain  $DOMAIN `
  --region  $REG_TYO `
  --project $PROJECT
```

- 가비아 DNS 설정 개요
  - 타입: `CNAME`
  - 이름: `admin`
  - 값: `ghs.googlehosted.com`
- DNS 전파 후 Cloud Run 도메인 매핑의 상태를 확인합니다.
  - `Ready=True`, `CertificateReady=True` 여야 정상.

#### 4.1.5 Cloud Scheduler 생성 및 확인

> 일일 보고서를 09:00(KST)에 자동으로 실행하는 HTTP Job 템플릿입니다.

```powershell
$LOCATION = "asia-northeast1"

gcloud scheduler jobs create http daily-report `
  --project  $PROJECT `
  --location $LOCATION `
  --schedule "0 9 * * *" `
  --time-zone "Asia/Seoul" `
  --http-method POST `
  --uri  "https://$DOMAIN/api/report" `
  --headers "X-Admin-Token=$TOKEN,Content-Length=0"
```

- 상태 확인 및 수동 실행
  ```powershell
  gcloud scheduler jobs describe daily-report `
    --project  $PROJECT `
    --location $LOCATION `
    --format "value(state,lastAttemptTime,lastScheduleTime)"

  gcloud scheduler jobs run daily-report `
    --project  $PROJECT `
    --location $LOCATION
  ```

#### 4.1.6 GCS 버전 관리 및 보존 정책 적용

```powershell
# 객체 버전 관리 활성화
gsutil versioning set on gs://$BUCKET

# 보존 기간 365일 설정 (예: 1년)
gsutil retention set 365d gs://$BUCKET

# 설정 확인
gsutil versioning get gs://$BUCKET
gsutil retention  get gs://$BUCKET
```

#### 4.1.7 일일 스모크 3줄로 최종 검증

```powershell
$URL = gcloud run services describe $LIVE `
  --region  $REG_TYO `
  --project $PROJECT `
  --format "value(status.url)"

$TOKEN = (gcloud secrets versions access latest `
  --secret=ADMIN_TOKEN `
  --project $PROJECT | Out-String).Trim()

Write-Host "HEALTH :" (curl.exe -s -o NUL -w "%{http_code}" "$URL/health")
Write-Host "STATUS :" (curl.exe -s -o NUL -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$URL/api/status")
Write-Host "REPORT :" (curl.exe -s -o NUL -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$URL/api/report")
```

- 기대값
  - HEALTH: `200`
  - STATUS: `200` (토큰 O), `401` (토큰 X)
  - REPORT: `200` (토큰 O)

여기까지 통과하면 **“최소 운영 가능 상태(MVP Ops)”**에 도달한 것입니다.

#### 4.1.8 L3 표준 리뷰 A1 Gate 상태머신 스모크 & CI 체크

> 목적: 표준 리뷰 A1 상태머신(`HOLD -> REVIEWED -> PUBLISHED + decision=PASS`)이  
>       운영 환경에서도 헌법/CI와 동일하게 동작하는지 빠르게 확인한다.

1. **PowerShell 스모크 (운영자 수동 점검)**  
   - `QualiJournal_Admin_Custom_Template_가이드북` 의  
     “L3 표준 리뷰 A1 Gate 스모크 템플릿” 코드를 사용한다.
   - 전제:
     - `3. 사전 준비 및 공통 PowerShell 설정` 을 먼저 수행해 `$PROJECT`, `$LIVE`, `$REG_TYO`, `$URL`, `$TOKEN` 이 세팅되어 있어야 한다.
   - 기대 흐름:
     - `test/init(reset=true)` 호출 후, `TEST-STD-1` 카드가 `HOLD`, `approved_by=[]` 인 상태로 존재.
     - `approve(r1)`, `approve(r2)` 호출 후, `REVIEWED` 상태 + `approved_by=["r1","r2"]`.
     - `publish` 호출 후, `PUBLISHED` 상태 + `decision="PASS"` 로 조회된다.

2. **CI 체크(GitHub Actions / Admin Tests)**  
   - 워크플로: `ci-test-admin.yml`
   - 잡: `Admin Tests (pytest only)` / `test-admin`
   - 포함 테스트:
     - `test_standards_reviews_a1_state_machine_flow`
     - `test_standards_reviews_a1_state_machine_error_cases`
   - 운영자는 배포 또는 중요한 코드 변경 후,
     - PR 화면에서 `Admin Tests (pytest only)` 체크가 **녹색(PASS)** 인지 확인하면,
     - happy-path + 주요 에러(404 / 409) 상태머신 규칙까지 모두 만족한다고 판단한다.

> 요약:  
> - **로컬/CI**: pytest로 상태머신 + 에러 케이스를 자동 검증한다.  
> - **운영 환경**: 템플릿 스크립트로 한 번만 수동 스모크를 돌려 최종 확인한다.

---

### 4.2 코드 수정 후 재배포 시나리오

> 예: `index.html`의 버튼 동작을 수정했거나, FastAPI 라우트를 변경한 경우.

1. GitHub main 브랜치에 PR 머지 → CI(GitHub Actions) 배포 성공 확인
2. 필요 시 로컬에서 수동 배포 (4.1.2~4.1.3 재사용)
3. Cloud Run 리비전 상태 및 트래픽 확인
   ```powershell
   gcloud run revisions list --service $LIVE --region $REG_TYO --project $PROJECT
   gcloud run services describe $LIVE --region $REG_TYO --project $PROJECT --format="value(status.traffic)"
   ```
4. 도메인에서 **캐시 무시 새로고침**
   ```powershell
   Invoke-WebRequest "https://$DOMAIN/?v=$(Get-Random)" `
     -Headers @{"Cache-Control"="no-cache"} `
     -OutFile "deployed_index_domain.html"
   ```
   - 파일을 열어 수정한 HTML/JS가 포함되어 있는지 확인합니다.
5. Admin UI에서 실제 버튼 동작 확인 (토큰 입력/해제, 새로고침, KPI 갱신, 보고서 생성, Export 등)

> 증상: 새 UI가 안 보이고 구버전인 것 같다면  
> ⇒ **(1) 최신 리비전에 트래픽 100%인지** 다시 확인,  
> ⇒ **(2) 브라우저 캐시 완전 무시(개발자 도구에서 Disable cache + Ctrl+F5)** 후 재시도 합니다.

---

### 4.3 ADMIN_TOKEN 교체 시나리오

> 토큰 노출 의심 또는 정기 교체 필요 시 수행합니다.

1. Secret Manager에 새 버전 추가
   ```powershell
   $NEW = [Convert]::ToBase64String([Guid]::NewGuid().ToByteArray()) + (Get-Random)
   $NEW | gcloud secrets versions add ADMIN_TOKEN --project $PROJECT --data-file=-
   ```

2. Cloud Scheduler 헤더 업데이트
   ```powershell
   $LOCATION = "asia-northeast1"
   gcloud scheduler jobs update http daily-report `
     --project  $PROJECT `
     --location $LOCATION `
     --uri      "https://$DOMAIN/api/report" `
     --headers  "X-Admin-Token=$NEW,Content-Length=0"
   ```

3. 스모크 테스트로 새 토큰 확인
   ```powershell
   $TOKEN=$NEW
   # 4.1.7의 스모크 스크립트 재사용
   ```

> Cloud Run 서비스는 `latest` 버전을 참조하도록 설정되어 있으므로, **시크릿 새 버전 추가만으로 런타임 토큰이 교체**됩니다.  
> 일반적으로 서비스 재배포는 필요하지 않습니다.

---

### 4.4 자주 발생하는 장애 패턴 및 빠른 대응

- `/health` 405  
  - 원인: HEAD 요청 사용, 엔드포인트가 GET만 허용하는 경우
  - 대처: `curl -X GET` 혹은 브라우저에서 GET으로 확인, 스모크 스크립트는 GET 기준으로 유지
- `/api/status` 401  
  - 원인: 토큰 누락 또는 잘못된 헤더 키 사용 (`Authorization` 대신 `X-Admin-Token` 필요 등)
  - 대처: 스모크 스크립트에서 사용하는 방식대로 `Authorization: Bearer` 또는 `X-Admin-Token` 헤더 설정
- 도메인 매핑 Pending / HTTPS 인증서 지연  
  - 원인: 가비아 CNAME 설정 미완료 또는 DNS 전파 지연
  - 대처: `nslookup -type=cname admin.standardai.co.kr`로 최종 CNAME을 확인, Cloud Run `domain-mappings describe`로 Ready/CertificateReady 상태 확인
- 패치 미적용(구버전 HTML 서빙)  
  - 원인: 최신 리비전에 트래픽이 가지 않거나, 브라우저 캐시
  - 대처: `gcloud run services update-traffic $LIVE --to-latest ...`, 이후 캐시 무시 새로고침
- 컨테이너 import 실패 / Revision not ready  
  - 원인: Cloud Run 서비스 계정에 Artifact Registry Reader 권한 없음
  - 대처: 서비스 계정에 `roles/artifactregistry.reader` 부여 후 재배포

---

## 5. Cloud Run 상세 설정 가이드

### 5.1 소스 기반 배포 vs 이미지 기반 배포

- **소스 기반 배포 (`--source .`)**
  - 장점: 개발자가 코드 수정 후 바로 배포하기 쉽고, Cloud Build가 Dockerfile을 자동해석
  - 용도: `quali-journal-admin` (서울) 소스 빌드용 서비스
- **이미지 기반 배포 (`--image ...`)**
  - 장점: 한 번 빌드한 이미지를 다수 서비스/리전에 재사용
  - 용도: `quali-admin-domap` (도메인 연결 서비스)에 동일 이미지 사용

두 서비스 모두 **동일한 컨테이너 이미지**를 사용하지만, 리전과 도메인/트래픽 설정이 다릅니다.

### 5.2 리비전과 트래픽 관리

- 리비전 목록 조회
  ```powershell
  gcloud run revisions list --service $LIVE --region $REG_TYO --project $PROJECT
  ```
- 서비스가 현재 어떤 리비전에 트래픽을 보내는지 확인
  ```powershell
  gcloud run services describe $LIVE `
    --region $REG_TYO `
    --project $PROJECT `
    --format="value(status.traffic)"
  ```
- 항상 최신 리비전만 사용하도록 강제
  ```powershell
  gcloud run services update-traffic $LIVE `
    --to-latest `
    --region $REG_TYO `
    --project $PROJECT
  ```

### 5.3 서비스 계정과 권한

각 Cloud Run 서비스는 고유한 서비스 계정으로 실행됩니다. 다음 권한들이 필요합니다.

- `roles/artifactregistry.reader` : 컨테이너 이미지를 Artifact Registry에서 읽기
- `roles/secretmanager.secretAccessor` : `ADMIN_TOKEN` 시크릿 값 읽기
- (선택) `roles/storage.objectAdmin` : GCS 버킷에 결과 파일 쓰기

서비스 계정 확인 예시:

```powershell
$SA_SRC = gcloud run services describe $SRC `
  --region $REG_SEO --project $PROJECT `
  --format="value(spec.template.spec.serviceAccountName)"

$SA_LIVE = gcloud run services describe $LIVE `
  --region $REG_TYO --project $PROJECT `
  --format="value(spec.template.spec.serviceAccountName)"
```

필요 권한 부여 예시:

```powershell
gcloud projects add-iam-policy-binding $PROJECT `
  --member="serviceAccount:$SA_SRC" `
  --role="roles/artifactregistry.reader"

gcloud projects add-iam-policy-binding $PROJECT `
  --member="serviceAccount:$SA_LIVE" `
  --role="roles/artifactregistry.reader"

gcloud projects add-iam-policy-binding $PROJECT `
  --member="serviceAccount:$SA_SRC" `
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT `
  --member="serviceAccount:$SA_LIVE" `
  --role="roles/secretmanager.secretAccessor"
```

### 5.4 보안 옵션 (공개/비공개)

- `--allow-unauthenticated`
  - Admin UI를 일반 HTTPS로 열 수 있도록 **프론트 페이지 접근은 허용**하는 방식
  - API는 **토큰으로 보호**되므로, 외부에서 호출해도 토큰 없이는 사용 불가
- (선택) Cloud Run Invoker 권한 최소화
  - 운영 정책상 더 강한 보호가 필요하다면 `--no-allow-unauthenticated` 설정 후
  - Cloud Scheduler용 서비스 계정, GitHub Actions 서비스 계정 등에게만 Invoker 권한을 부여하는 전략도 사용 가능

---

## 6. 커스텀 도메인 & DNS & 인증서 운용

### 6.1 도메인 매핑 생성/삭제

- 생성
  ```powershell
  gcloud beta run domain-mappings create `
    --service $LIVE `
    --domain  $DOMAIN `
    --region  $REG_TYO `
    --project $PROJECT
  ```

- 삭제 후 재생성 (문제 발생 시)
  ```powershell
  gcloud beta run domain-mappings delete `
    --domain $DOMAIN `
    --region $REG_TYO `
    --project $PROJECT -q
  ```

### 6.2 DNS (가비아) 설정 요약

- 레코드 타입: `CNAME`
- 호스트: `admin`
- 값: `ghs.googlehosted.com`

DNS 전파 후, 글로벌 DNS 서버(예: `8.8.8.8`, `1.1.1.1`) 기준으로 `admin.standardai.co.kr`이 `ghs.googlehosted.com`으로 보이는지 확인합니다.

```powershell
nslookup -type=cname admin.standardai.co.kr 8.8.8.8
nslookup -type=cname admin.standardai.co.kr 1.1.1.1
```

### 6.3 매핑 상태 및 인증서 확인

```powershell
gcloud beta run domain-mappings describe `
  --domain $DOMAIN `
  --region $REG_TYO `
  --project $PROJECT `
  --format="yaml(status.resourceRecords,status.conditions)"
```

- 기대 상태
  - `Ready: True`
  - `CertificateReady: True`
  - 리소스 레코드: CNAME `admin.standardai.co.kr → ghs.googlehosted.com`

---

## 7. Cloud Scheduler — 일일 보고서 자동화 상세

### 7.1 Job 생성/수정/삭제

- 생성 예시는 4.1.5 참조

- 토큰 변경 시 헤더 업데이트
  ```powershell
  gcloud scheduler jobs update http daily-report `
    --project  $PROJECT `
    --location "asia-northeast1" `
    --uri      "https://$DOMAIN/api/report" `
    --headers  "X-Admin-Token=$TOKEN,Content-Length=0"
  ```

- 삭제
  ```powershell
  gcloud scheduler jobs delete daily-report `
    --project $PROJECT `
    --location "asia-northeast1" -q
  ```

### 7.2 Job 상태 모니터링

```powershell
gcloud scheduler jobs describe daily-report `
  --project  $PROJECT `
  --location "asia-northeast1"
```

- 주요 필드
  - `state`: ENABLED / PAUSED / DISABLED
  - `lastAttemptTime`, `lastScheduleTime`: 최근 실행 정보
  - 에러 발생 시 `status` 메시지 확인

### 7.3 실패 시 대응 팁

- 401/403: `X-Admin-Token` 값 및 Cloud Run Invoker 권한 확인
- 404/500: `/api/report` 핸들러 코드/배포 상태 점검
- 타임아웃: Cloud Run 타임아웃 설정 및 뉴스 수집량 점검

---

## 8. GCS 백업/아카이브 전략

### 8.1 정책 설정

이미 4.1.6에서 명령을 실행했다면, 버킷은 다음 상태를 만족해야 합니다.

- Versioning: `Enabled`
- Retention: `365d`

확인 예시:

```powershell
gsutil versioning get gs://$BUCKET
gsutil retention  get gs://$BUCKET
```

### 8.2 일일/주간 백업 동작

- `/api/report` 실행 시 생성된 Markdown/CSV 파일이 `gs://qualijournal-archive`에 업로드됩니다.
- 동일 이름 파일이 재업로드되면, 버전 관리 정책에 따라 이전 버전도 유지됩니다.

### 8.3 월 1회 복구 리허설(권장)

1. 특정 일자의 MD/CSV 파일 1개를 선택해 로컬로 다운로드
2. 로컬 환경 또는 별도 테스트 버킷에 동일 경로로 복원
3. 파일 내용이 손상 없이 열리는지 확인
4. 문제가 없다면 운영 로그에 “복구 리허설 완료”로 기록

---

## 9. 일일 스모크 테스트 & 운영 루틴

### 9.1 일일 스모크 3줄 (운영자 기준)

> 운영자는 매일 아침, 아래 스크립트를 실행해 **서비스 기본 상태**를 확인합니다.

```powershell
$URL = gcloud run services describe $LIVE `
  --region  $REG_TYO `
  --project $PROJECT `
  --format "value(status.url)"

$TOKEN = (gcloud secrets versions access latest `
  --secret=ADMIN_TOKEN `
  --project $PROJECT | Out-String).Trim()

Write-Host "HEALTH :" (curl.exe -s -o NUL -w "%{http_code}" "$URL/health")
Write-Host "STATUS :" (curl.exe -s -o NUL -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$URL/api/status")
Write-Host "REPORT :" (curl.exe -s -o NUL -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$URL/api/report")
```

### 9.2 결과 해석 및 대응

- HEALTH = 200, STATUS = 200, REPORT = 200  
  → 정상
- STATUS = 401  
  → 토큰 문제: Secret Manager 혹은 헤더 전송 로직 점검
- REPORT = 5xx  
  → `/api/report` 내부 로직/외부 의존성(뉴스 소스, 요약 API 등) 점검, Cloud Run 로그 확인

### 9.3 일일/주간/월간 체크리스트 예시

- **일일**
  - [ ] 스모크 테스트 3줄 실행 (HEALTH/STATUS/REPORT 확인)
  - [ ] Cloud Scheduler `daily-report` 실행 로그 확인
  - [ ] GCS 버킷에 금일 날짜의 MD/CSV 파일 생성 여부 확인
- **주간**
  - [ ] 도메인/인증서 상태(만료 예정일) 점검
  - [ ] ADMIN_TOKEN 교체 필요성 검토
- **월간**
  - [ ] 백업 복구 리허설 1회 수행
  - [ ] Cloud Run 리비전 및 트래픽 설정 검토 (불필요 리비전 정리)

---

## 10. Definition of Done (운영 가능 상태)

다음 체크리스트를 모두 만족하면, **“운영 가능한 상태”**로 인정합니다.

- [ ] `quali-journal-prod` 프로젝트에 `quali-journal-admin`, `quali-admin-domap` 서비스가 존재한다.
- [ ] 두 서비스 모두 `ADMIN_TOKEN`(Secret), `GCS_BUCKET=qualijournal-archive` 환경 변수를 갖는다.
- [ ] `quali-admin-domap` 최신 리비전에 트래픽 100%가 할당되어 있다.
- [ ] `admin.standardai.co.kr` → `quali-admin-domap` 도메인 매핑이 `Ready=True`, `CertificateReady=True` 상태다.
- [ ] Cloud Scheduler `daily-report` 잡이 `asia-northeast1` 리전에 존재하고, 09:00 Asia/Seoul에 `/api/report`를 호출한다.
- [ ] `gs://qualijournal-archive` 버킷에 버전 관리 및 365일 보존 정책이 적용되어 있다.
- [ ] 일일 스모크 3줄 실행 시 HEALTH/STATUS/REPORT 의 HTTP 코드가 기대값을 만족한다.

---

## 11. 부록 — 빠른 참조용 명령 모음

### 11.1 핵심 변수 요약

```powershell
$PROJECT = "quali-journal-prod"
$REG_SEO = "asia-northeast3"
$REG_TYO = "asia-northeast1"
$SRC     = "quali-journal-admin"
$LIVE    = "quali-admin-domap"
$DOMAIN  = "admin.standardai.co.kr"
$BUCKET  = "qualijournal-archive"
```

### 11.2 응급 복구 2줄

```powershell
gcloud run services update-traffic $LIVE --to-latest --region $REG_TYO --project $PROJECT
Invoke-WebRequest "https://$DOMAIN/?v=$(Get-Random)" -Headers @{"Cache-Control"="no-cache"} -OutFile "deployed_index_domain.html"
```

위 두 줄로 **최신 리비전 강제 적용 + 새 HTML 강제 로드**까지 한번에 처리 가능합니다.

---

> 이 가이드북은 “SSOT 실행 템플릿”을 풀버전으로 확장한 문서입니다.  
> 인프라 값이 변경될 경우, 항상 **이 문서 먼저 수정** 후 다른 문서에서 이 파일을 참조하도록 유지해 주세요.

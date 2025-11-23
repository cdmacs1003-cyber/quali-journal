
# QualiJournal 관리자 모드 SSOT (Admin DoD v1.1)

> **Single Source of Truth (SSOT)**  
> 이 문서는 QualiJournal 관리자 모드(Admin)의 개발·테스트·배포·운영을 위한 단일 기준(Single Source of Truth)이다.  
> 관리자 모드 관련 모든 논의·결정·작업은 아래 내용을 기본 전제로 한다.

---

## 0. 문서 개요

### 0.1 목적

- 관리자 모드의 **개발 완료 기준(DoD, Definition of Done)** 을 한 곳에 모아 정리한다.
- 기능, 비기능, 운영/환경, DevOps, Cloud SQL 정책, PR 체크리스트를 **하나의 기준선**으로 통합한다.
- “관리자 모드 인크리먼트가 Done인가?”를 판단할 때 이 문서를 가장 먼저 참고한다.

### 0.2 Ground Truth 문서

이 SSOT는 다음 다섯 문서를 종합·요약한 것이다.

1. **「1115 관리자 모드 개발 완료 기준 정의서」**  fileciteturn1file0  
2. **「DoD(Definition of Done) 충족 상태 점검 요약서」**  fileciteturn1file1  
3. **「QualiJournal Cloud SQL Stop Policy」**  fileciteturn1file2  
4. **「QualiJournal_DoD_런북_인수인계_v1.1__2025-11-15.md」**  fileciteturn1file3  
5. **「QualiJournal_PR_Template_DevOps_Checklist_v1_2025-11-15_1.md」**  fileciteturn1file4  

원문이 변경될 경우, 이 SSOT를 함께 업데이트해야 한다.

---

## 1. 기능 DoD 체크리스트 (요약)

관리자 모드는 다음 기능이 “정의서 기준”으로 동작해야 **기능 DoD를 충족**한 것으로 본다. fileciteturn1file0  

### 1.1 KPI 및 상태 대시보드 (/api/status, /api/ready)

- **설명**: 전체 기사 수, 승인/대기 수, 게이트 기준치, 게이트 통과 여부 등을 한눈에 보여주는 관리용 대시보드.  
  - `/api/status` 는 KPI·게이트 기준을 상세 지표로 보여주고,  
  - `/api/ready` 는 같은 정보와 내부 점검 결과를 바탕으로 “지금 배포가 Gate PASS 상태인지”를 한 줄로 요약하는 Ready API 이다.

- **완료 기준**
  - 관리자 UI 대시보드와 `/api/status` 응답 값이 서로 일관된다.
  - 최소 필드: `total`, `ready_count`, `gate_required`, `ready_rate`, `gate_pass` 등은 기존 정의를 따른다.
  - `/api/ready` 응답이 **공통 Admin 응답 스키마(성공 시 `ok=true`)** 를 따르며,
    Ready 판단에 필요한 필드를 모두 포함한다(예: `ready`, `gate_pass`, 내부 점검 요약 등).

- **검증 방법**
  - 스테이징 또는 로컬 환경에서 테스트 데이터(예: 기사 100건 중 80건 승인)를 준비한다.
  - `/api/status` 응답에서 위 필드 값이 기대값과 일치하는지 확인하고, UI 대시보드 숫자와 교차 검증한다.
  - `/api/ready` 를 호출해 현재 배포 상태가 기대하는 Ready 판정과 일치하는지 확인한다.
  - 자동화 테스트: `pytest admin/tests/test_admin_api_schema.py` 실행 시  
    `/api/status`, `/api/ready` 응답 필드·타입·상태 코드가 깨지면 테스트가 실패해야 한다.


### 1.2 비동기 플로우 실행 및 로그 확인 (/api/flow/*, /api/tasks/flow)

- **설명**: 뉴스 수집, 일일 플로우, 키워드 플로우 등 백그라운드 작업을 수동으로 실행하고, 결과 로그를 조회하는 기능. fileciteturn1file0turn1file1  
- **완료 기준**
  - 운영자가 일일/키워드 플로우를 버튼 또는 API로 실행할 수 있다.
  - 플로우 실행 시, 대응되는 로그 파일이 `/api/logs` 목록에 새로 생긴다.
- **검증 방법**
  - `/api/flow/daily` 또는 `/api/tasks/flow?type=daily` 를 호출한다.
  - 작업 완료 후 `/api/logs` 를 호출해 신규 로그가 생성되었는지 확인한다.
  - 필요 시 `/api/tasks/{job_id}/stream` SSE로 진행 로그를 확인한다.

### 1.3 기사 목록 확인 및 승인/거절/코멘트

- **설명**: 수집된 기사 목록을 확인하고, 승인/거절 처리 및 편집자 코멘트를 저장하는 기능. fileciteturn1file0  
- **완료 기준**
  - 대기 중(ready 전) 기사 목록을 UI와 API(`/api/items`, `/api/community`)에서 조회할 수 있다.
  - 각 기사에 대해 승인/거절이 가능하며, 승인 시 상태가 “준비(ready)”로 변경된다.
  - 코멘트 입력 시 데이터에 저장된다.
- **검증 방법**
  - 테스트용 기사 몇 개를 준비하고 승인/거절 버튼을 눌러본다.
  - 처리 후 상태 필드와 코멘트 저장 여부를 API 응답 또는 데이터 저장소에서 확인한다.

### 1.4 발행 및 게이트 설정 (/api/items/{id}/publish, /api/config/gate_required)

- **설명**: 승인된 기사를 실제 사용자 서비스에 발행하고, 발행 최소 기준(게이트)을 설정·조정하는 기능. fileciteturn1file0turn1file1  
- **완료 기준**
  - `/api/items/{id}/publish` 호출 시 해당 기사가 발행 상태로 변경되고, 사용자 화면에 노출된다.
  - `/api/config/gate_required` 값 변경 시 `/api/status` 의 `gate_required`가 즉시 갱신된다.
- **검증 방법**
  - 특정 기사에 대해 발행 API를 호출한 뒤 사용자 화면 노출 여부를 확인한다.
  - 게이트 값을 10→15 등으로 변경하고 `/api/status` 응답에서 값이 즉시 반영되는지 본다.

### 1.5 보고서 생성 및 결과물 내보내기 (/api/report, /api/export/*)

- **설명**: 특정 날짜/키워드에 대한 요약 보고서를 생성하고, MD/CSV 파일로 다운로드하는 기능. fileciteturn1file0turn1file1  
- **완료 기준**
  - `/api/report` 호출 시 `ok=true`와 함께 보고서 파일 경로가 반환된다.
  - `/api/export/md`, `/api/export/csv` 로 해당 파일을 실제로 다운로드할 수 있다.
- **검증 방법**
  - 1일치 보고서를 생성하고 반환된 `path` 를 사용해 export API를 호출한다.
  - 다운로드된 MD/CSV 내용이 비어 있지 않은지 확인한다.

### 1.6 로그 및 백업 상태 확인 (/api/logs, /api/backup/status)

- **설명**: 시스템 로그와 백업 상태를 조회하고, 최근 성공/실패 여부를 모니터링한다. fileciteturn1file0turn1file1  
- **완료 기준**
  - `/api/logs` 에서 로그 파일 목록이 JSON으로 조회된다.
  - `/api/logs/{name}`, `/api/logs/{name}/download` 로 개별 로그 열람·다운로드가 가능하다.
  - `/api/backup/status` 에서 최근 백업 성공/실패 시각과 메시지가 노출된다.
- **검증 방법**
  - `/api/logs` 응답에 대표 로그 파일 이름이 포함되는지 확인한다.
  - 그 중 하나를 열어 최신 내용이 출력되는지 본다.
  - `/api/backup/status` 응답에 마지막 백업 성과 시간이 올바르게 표시되는지 확인한다.

### 1.7 헬스체크 및 디버그용 API (/health, /api/debug/*)

- **설명**: 서비스 상태를 점검하는 헬스체크와, 운영 편의를 위한 디버그 정보 제공용 API. fileciteturn1file0turn1file1  
- **완료 기준**
  - `/health` 는 항상 200 OK 와 단순한 “ok” 메시지를 반환한다.
  - `/api/debug/*` 계열 엔드포인트는 런타임/설정/메트릭 등의 JSON 정보를 제공한다(운영/개발 환경에서만 사용).
- **검증 방법**
  - `/health` 를 여러 번 호출해 200 응답을 확인한다.
  - ` /api/debug/runtime`, `/api/debug/config` 등을 호출해 내부 정보가 JSON으로 반환되는지 확인한다.

---

## 2. 비기능 DoD 체크리스트 (요약)

### 2.1 응답 JSON 구조 및 HTTP 상태 코드 일관성

- **요구사항** fileciteturn1file0turn1file1  
  - 성공: HTTP 200 + `{ ok: true, data: {...} }`
  - 실패: HTTP 4xx/5xx + `{ ok: false, error, error_code, detail? }`
  - 모든 엔드포인트가 정의된 스키마(필드명·타입)를 공유해야 한다.
- **검증**
  - 대표 API 여러 개에 대해 성공/실패 케이스를 만들어 응답 구조와 상태 코드를 비교한다.
  - Pydantic `BaseModel` 등을 활용해 응답 스키마를 코드로도 강제한다.

### 2.2 인증/권한 안전성

- **요구사항** fileciteturn1file0turn1file3  
  - 관리자 모드 API는 환경 변수 `ADMIN_TOKEN`(또는 동등한 비밀 값) 기반으로 보호한다.
  - 민감 API는 **유효한 토큰 헤더**(예: `Authorization: Bearer <token>`, `X-Admin-Token`)가 있어야만 처리한다.
  - Cloud Run 서비스는 Invoker 권한이 특정 서비스 계정 등으로 제한된 **private 서비스**를 기본으로 한다.
- **검증**
  - 토큰 없이 호출 시 401/403 이 반환되는지 확인한다.
  - 올바른 토큰으로 호출 시 200 응답이 나오는지 확인한다.
  - Cloud Run 콘솔에서 Invoker 권한이 특정 주체에만 부여되어 있는지 점검한다.

### 2.3 보고서 및 플로우 성능

- **요구사항** fileciteturn1file0turn1file1  
  - 일반적인 기사량(수십 건 기준)에서 보고서 생성은 수 초 내에 끝나야 한다.
  - 일일/키워드 플로우 등 일괄 작업도 Cloud Run 타임아웃(기본 15분, 상한 60분) 내에서 완료되어야 한다.
  - 메모리 사용량은 설정 한도(예: 512MiB)를 넘지 않아야 한다.
- **검증**
  - 테스트 데이터 50~200건 정도를 준비해 `/api/report`, `/api/flow/daily` 처리 시간과 메모리를 모니터링한다.
  - Cloud Run 로그/모니터링 대시보드에서 타임아웃·OOM(No) 여부를 확인한다.

### 2.4 유지보수 가능성

- **요구사항** fileciteturn1file0turn1file1  
  - 데이터 모델과 요청/응답 스키마가 Pydantic 모델로 일관되게 선언되어 있어야 한다.
  - 상수·구조는 중복 없이 “한 곳에서만 정의”하는 단일 소스 원칙을 따른다.
  - 핵심 기능에 대한 자동화 테스트가 존재해 리팩토링 시 회귀를 빠르게 발견할 수 있어야 한다.
- **검증**
  - 새로운 필드를 추가해도 모델 한 곳만 수정하면 전체 테스트가 통과하는지 확인한다.
  - 동일 상수가 여러 파일에 하드코딩돼 있지 않은지 코드 리뷰로 점검한다.

---

## 3. 운영/환경 DoD 체크리스트 (요약)

### 3.1 Cloud Run 환경 제약 준수

- **요구사항** fileciteturn1file0turn1file1  
  - 모든 요청 처리가 Cloud Run 최대 타임아웃 이내에 끝나야 한다.
  - 메모리 사용량이 설정 한도를 넘지 않아야 한다.
  - 부하가 걸려도 인스턴스가 타임아웃/OOM으로 자주 죽지 않도록 설계한다.
- **검증**
  - 대용량 시나리오(기사 200건 수준)를 실행해 가장 긴 요청 시간이 타임아웃보다 충분히 짧은지 확인한다.
  - 모니터링에서 CPU/메모리 그래프를 확인해 여유를 확인한다.

### 3.2 배포 및 IAM 구성

- **요구사항** fileciteturn1file0turn1file3  
  - GitHub Actions 등 CI/CD 파이프라인이 main 브랜치 변경을 자동으로 Cloud Run에 배포한다.
  - 배포 후 최신 리비전에 100% 트래픽이 연결되어 있어야 한다.
  - Cloud Run Invoker 권한은 배포용/운영용 서비스 계정에만 부여한다.
  - `ADMIN_TOKEN` 등 비밀은 Secret Manager를 통해 환경 변수로 안전하게 주입한다.
- **검증**
  - 테스트 브랜치에서 PR/머지 후 새로운 리비전이 생성되고 트래픽이 정상 전환되는지 확인한다.
  - Cloud Run IAM 탭에서 Invoker 범위를 점검한다.
  - 환경 변수 화면에서 비밀 값이 정상 주입되었는지 확인한다.

---

## 4. DoD 충족 상태 점검 요약 (런북 v1.1 기반)

다음 내용은 **DoD(Definition of Done) 충족 상태 점검 요약서 + 인수인계 런북 v1.1** 를 바탕으로 한 “실행용 요약”이다. fileciteturn1file1turn1file3  

### 4.1 단계별 실행 계획(Plan – 프롬프트 체이닝 관점)

0단계. **기준 정합화**
- 입력: 정의서, Q&A, 현재 Cloud Run 리비전/트래픽/도메인, ADMIN_TOKEN 위치.
- 산출물: 엔드포인트·스키마·HTTP 코드·권한 맵, 현재 배포 스냅샷.
- 스모크: `/health`, `/api/status(토큰有/無)` 확인.

1단계. **API 명세·스키마 정합화**
- 목표: 모든 주요 API가 공통 응답 구조와 HTTP 코드 규약을 따른다.
- 작업: Pydantic 응답 모델 정의, 에러 코드 테이블 작성, 엔드포인트별 스펙 표준화.

2단계. **보안/IAM 하드닝**
- 목표: 토큰+Cloud Run IAM 2중 방어.
- 작업: 익명 접근 차단, Invoker 최소화, Scheduler/SA 권한 정리, 토큰 회전 절차.

3단계. **로그·백업·플로우 모니터링**
- 목표: `/api/logs`, `/api/backup/status`, `/api/tasks/flow`가 연계된 관제 체계를 갖춘다.
- 작업: 백업 상태 응답 형식 표준화, 로그/다운로드 케이스 정리, 플로우 실행→로그 연결 확인.

4단계. **성능/리소스·운영 자동화**
- 목표: 타임아웃/메모리 한도 내 처리 + 드리프트(리비전·트래픽) 방지.
- 작업: 성능 측정, 부하 테스트, CI에서 `update-traffic --to-latest` 정례화.

5단계. **릴리스 판정(Go/No-Go)**
- 목표: 기능/비기능/보안/운영 기준을 한 번에 보고 결정한다.
- 작업: 차단 이슈 목록과 완화책 정리, Go/No-Go 표 작성.

---

## 5. Cloud SQL Stop Policy (운영 정책 SSOT)

Cloud SQL 관련 내용은 **별도 정책 문서**를 SSOT로 삼는다. fileciteturn1file2  

### 5.1 기본 원칙

- Cloud SQL 인스턴스 `quali-pg` 는 **STOPPED + activationPolicy=NEVER** 상태가 “기본”이다.
- DB가 필요한 작업(배포, 마이그레이션, 스키마 점검 등)이 있을 때만 일시적으로 `ALWAYS` 로 변경해 사용한다.
- 작업 후 반드시 다시 `NEVER` 로 되돌려 STOP 상태를 유지한다.
- 애플리케이션에는 `DB_DISABLED=true` 를 설정해, DB가 꺼져 있어도 에러 없이 동작하도록 방어한다.

### 5.2 일상 스모크(매일 3줄)

- Cloud SQL 상태, Cloud Run 트래픽, Admin 도메인 HTML을 한 번에 확인한다.

```bash
gcloud sql instances list --format="table(name,region,state,settings.activationPolicy)"
gcloud run services describe quali-admin-domap --region asia-northeast1 --format="value(status.traffic)"
Invoke-WebRequest "https://admin.standardai.co.kr/?v=$(Get-Random)" -Headers @{"Cache-Control"="no-cache"} -OutFile "deployed_index.html"
```

- 기대 결과
  - SQL: `STOPPED / NEVER`
  - Run: `percent: 100`
  - HTML: 정상 다운로드

### 5.3 배포 차단 훅(CI)

- SQL이 STOP 상태이면 **배포를 실패**시키는 훅을 CI에 추가한다.

```bash
STATE=$(gcloud sql instances describe quali-pg --format="value(state)")
if [ "$STATE" = "STOPPED" ]; then
  echo "::error ::Cloud SQL is STOPPED; deployment blocked."
  exit 1
fi
```

---

## 6. DevOps 품질 게이트 (DoD v1.1 – Admin Tests + main 브랜치 보호)

이 절은 **QualiJournal 관리자 모드 – DevOps 품질 게이트 패키지 v1.1** 의 내용을 DoD 관점에서 요약한 것이다. fileciteturn1file3  

### 6.1 Admin Tests (자동화 테스트 품질 게이트)

- **목적**
  - 관리자 모드 배포 전 최소 동작을 보장하는 **자동화 스모크 테스트**.
  - Admin Tests 가 실패하면 main 브랜치에 병합할 수 없다.
- **필수 조건**
  - GitHub Actions 워크플로 `Admin Tests (pytest only)` 의 `test-admin` 잡이 성공 상태여야 한다.
  - `POST /api/report` 에 대해:
    - HTTP 200 OK
    - `ok=true`, `op="report"`, 유효한 `path`, 정수형 `count` 필드 존재
  - 해당 `path` 로 `GET /api/archive/{path}` 호출 시:
    - 200 OK
    - `Content-Type` 이 text/markdown 계열
    - 본문이 비어 있지 않은 리포트여야 한다.
- **체크리스트**
  - [ ] 최신 PR에서 `test-admin` 이 성공 상태다.
  - [ ] `/api/report` → `/api/archive/{path}` 까지 수동 스모크 결과가 기대와 같다.

### 6.2 main 브랜치 보호(브랜치 품질 게이트)

- **목적**
  - main 브랜치를 관리자 모드 배포의 **단일 진실 소스(SSOT)** 로 보호한다.
  - 직접 push 를 막고, 품질 게이트를 통과한 PR만 병합하게 한다.
- **필수 설정(GitHub Branch Protection)**
  - “Require a pull request before merging” = ON
  - “Require status checks to pass before merging” = ON
    - 필수 status check 목록에 `test-admin` 포함
  - “Require signed commits” = ON (모든 커밋 Verified)
  - “Require linear history” = ON (squash/rebase 기반 선형 히스토리)
- **체크리스트**
  - [ ] main 에 직접 push 하지 않고 PR로만 변경한다.
  - [ ] `test-admin` 이 필수 status check 로 설정되어 있다.
  - [ ] main 커밋이 모두 Verified 이다.
  - [ ] 로그가 선형 히스토리로 유지되고 있다.


### 6.3 DevOps 품질 게이트 선언

관리자 모드 인크리먼트가 **“Done”** 으로 인정되려면 반드시 아래 두 조건을 동시에 만족해야 한다.

1. 6.1 의 Admin Tests 를 통과한다.  
   - 현재 기준선(`main-signed-ssot`)에서 Admin Tests(job `test-admin`)는 다음 명령을 실행한다.  
     - `pytest admin/tests/test_admin_api_schema.py admin/tests/test_tools_quick.py`
   - `test_admin_api_schema.py`  
     - `/api/status`, `/api/ready` 응답 필드·타입·상태 코드가 Admin DoD·응답 스키마 SSOT와 일치하는지 검증한다.
   - `test_tools_quick.py`  
     - Quick Tools Gate(approve_top / repair) 의 기본 성공 + B-Mode 성공 + 대표 실패 조합을 검증한다.  
     - B-Mode 조건: `rc==127` + `sync_log.ok==True` + `"not found"` 포함 시 성공으로 간주되는 규칙을 테스트한다.

2. 6.2 의 main 브랜치 보호 규칙을 충족하는 PR을 통해 main 에 병합된다.


---

## 7. PR 템플릿 – DevOps 체크리스트 v1 (요약)

**목적**: 개발자가 PR을 만들 때 DoD-DevOps 기준을 간단히 다시 확인하도록 돕는 템플릿이다. fileciteturn1file4  

### 7.1 PR 기본 정보 섹션

- 제목, 관련 이슈, 변경 유형(신규 기능/버그 수정/리팩터링/설정 변경), 주요 변경 내용 3줄, 영향 범위 등을 간단히 적는다.

### 7.2 DevOps 신호등 체크리스트

1. **Admin Tests**
   - [ ] Actions 탭에서 `Admin Tests (pytest only)` / `test-admin` 이 green/passed 인지 확인했다.
2. **main 브랜치 보호**
   - [ ] main 에 직접 push 하지 않고 PR을 사용했다.
   - [ ] main 보호 규칙이 켜져 있는 화면을 확인했다.
3. **서명 커밋**
   - [ ] PR에 포함된 커밋이 GitHub 에서 Verified 로 보이는지 확인했다.
4. **선형 히스토리**
   - [ ] 머지 시 “Squash and merge” 또는 “Rebase and merge” 만 사용한다.

---

## 8. 릴리스 Go / No-Go 기준

다음 표는 DoD 점검 결과를 기반으로 릴리스 여부를 결정할 때 사용하는 상위 기준이다. fileciteturn1file1  

| 영역 | Go 기준 | No-Go(차단) | 기본 조치 |
|------|---------|-------------|-----------|
| 기능 | 정의서의 기능 항목을 모두 통과 | 주요 엔드포인트 1개라도 실패 | 핫픽스 또는 롤백 |
| 응답/코드 | 공통 JSON 스키마·HTTP 코드 일치 | 엔드포인트마다 구조가 제각각 | 응답 모델 정비 후 재배포 |
| 보안/IAM | 토큰+Invoker 최소 권한, 401/403 흐름 정상 | 공개 Invoker, 토큰 미적용 | 권한 재설정·익명 차단 |
| 배포/도메인 | 최신 리비전 100% 트래픽, 최신 HTML/JS | 구버전 UI/JS 노출 | 트래픽 최신화·캐시 무시 재배포 |
| 스케줄러 | 7일 연속 자동 보고서 성공률 ≥ 99% | 잦은 실패 또는 401 | 토큰 회전·잡 수정 |
| 백업 | 상태 OK, 버저닝/보존 정책 적용 | 마지막 성공 시각 없음 | 백업 잡/정책 보강 |
| Cloud SQL | STOP 정책 준수(기본 STOP) | 활성 상태 방치 | STOP 복구·차단 훅 적용 |

---

## 9. 샘플 명령 모음 (발췌)

### 9.1 헬스·상태·플로우·보고서 (cURL)

```bash
# 헬스 체크
curl -i "https://<DOMAIN>/health"

# 상태(KPI) – 성공/실패 비교
curl -s "https://<DOMAIN>/api/status" -H "Authorization: Bearer DUMMY_ADMIN_TOKEN"
curl -s "https://<DOMAIN>/api/status"  # 기대: 401

# 플로우 트리거
curl -s -X POST "https://<DOMAIN>/api/tasks/flow?type=daily" -H "Authorization: Bearer DUMMY_ADMIN_TOKEN"

# 보고서 생성 → 내보내기
curl -s -X POST "https://<DOMAIN>/api/report" -H "Authorization: Bearer DUMMY_ADMIN_TOKEN"
curl -s "https://<DOMAIN>/api/export/md?date=YYYY-MM-DD"  -H "Authorization: Bearer DUMMY_ADMIN_TOKEN" -o report.md
curl -s "https://<DOMAIN>/api/export/csv?date=YYYY-MM-DD" -H "Authorization: Bearer DUMMY_ADMIN_TOKEN" -o report.csv
```

### 9.2 Cloud Run 트래픽 최신화 & 캐시 우회 (PowerShell)

```powershell
$SERVICE="quali-admin-domap"
$REGION="asia-northeast1"
$PROJECT="quali-journal-prod"

gcloud run services update-traffic $SERVICE --to-latest --region $REGION --project $PROJECT
Invoke-WebRequest "https://admin.standardai.co.kr/?v=$(Get-Random)" -Headers @{"Cache-Control"="no-cache"} -OutFile "deployed_index.html"
```

---

## 10. 운영자가 이 문서를 어떻게 사용하면 좋은지

1. **새 기능 개발 시작 전**  
   - 1~3장을 읽고 “이번 변경이 어느 DoD 항목과 관련돼 있는지” 표시한다.
2. **PR 열기 전**  
   - 6~7장의 DevOps 체크리스트를 따라 Admin Tests, 브랜치 보호, 서명 커밋 상태를 확인한다.
3. **릴리스 전 최종 점검**  
   - 4장(실행 계획)과 8장(Go/No-Go 기준)을 기반으로 체크리스트를 채우고, No-Go 항목이 없는지 확인한다.
4. **장애/이슈 발생 시**  
   - 1~3장으로 기능/환경 요구를 다시 확인하고, 5~6장으로 Cloud SQL·DevOps 쪽 설정이 어긋나지 않았는지 점검한다.

이 문서는 “관리자 모드 개발·운영의 기준선”으로 사용되며, 변경이 필요할 경우 **반드시 PR + 리뷰 + DevOps 품질 게이트**를 거쳐 갱신한다.
<!-- SSOT 변경 테스트 v2 (라벨 없이 CI 반응 확인용) -->

## 11. Admin API 응답 스키마 SSOT v1

> QualiJournal 관리자 모드 · 기능/비기능 DoD 기준 정렬용 응답 스키마 헌법

이 절은 관리자 모드(Admin)의 핵심 API에 대해
1) 대상 엔드포인트 목록(①),  
2) 공통 JSON 응답 스키마 정의(②),  
3) 엔드포인트별 매핑 표(③)  
를 한 곳에 모아 정리한 단일 기준(SSOT)이다.

### 11.1 대상 엔드포인트 목록 (①)

목적: 관리자가 실제로 사용하는 **핵심 Admin API**를 한눈에 정리하고,
각 엔드포인트의 **역할·현재 응답 구조(대략)**를 기록한다.

#### 11.1.1 핵심 엔드포인트 표

> 주의: `현재 응답 구조(대략)`은 정의서/런북에서 설명된 동작과 기존 구현을 바탕으로 한 요약이다.  
> 향후 실제 서버 코드 기준으로 재검증하여 필요 시 업데이트한다.

| path | method | 용도(한 줄) | 현재 응답 구조(요약) |
| --- | --- | --- | --- |
| `/api/status` | GET | KPI/게이트 상태 대시보드용 요약 지표 | `{ ok: true, total, ready_count, gate_required, ready_rate, gate_pass, ... }` |
| `/api/report` | POST | 특정 날짜/키워드 보고서 생성 | `{ ok: true, path }` (보고서 파일 경로) |
| `/api/export/md` | GET | MD 보고서 파일 다운로드 | MD 파일 스트림 응답 또는 파일 다운로드 응답 |
| `/api/export/csv` | GET | CSV 보고서 파일 다운로드 | CSV 파일 스트림 응답 또는 파일 다운로드 응답 |
| `/api/enrich/keyword` | POST | 키워드 기반 기사 보강/요약 작업 트리거 | `{ ok: true, job_id, ... }` |
| `/api/enrich/selection` | POST | 선정 기사 셋에 대한 추가 요약/보강 트리거 | `{ ok: true, job_id, ... }` |
| `/api/logs` | GET | 시스템 로그 파일 목록 조회 | `{ ok: true, logs: [{ name, size, mtime, ... }], ... }` |
| `/api/logs/{name}` | GET | 개별 로그 파일 내용 조회 | `text/plain` 또는 `{ ok: true, content: "..." }` |
| `/api/logs/{name}/download` | GET | 개별 로그 파일 다운로드 | 로그 파일 다운로드 응답 |
| `/api/backup/status` | GET | 정기 백업 최근 성공/실패 여부 및 시각 | `{ ok: true, last_success_at, last_failure_at, ... }` |
| `/api/config/gate_required` | GET | 현재 게이트 기준 값 조회 | `{ ok: true, gate_required }` |
| `/api/config/gate_required` | PATCH | 게이트 기준 값 변경 | `{ ok: true, gate_required }` (변경 후 값) |
| `/api/items` | GET | Ready/대기 기사 목록 조회 + 승인/거절 대상 | `{ ok: true, items: [{ id, title, source, state, ... }], ... }` |
| `/api/items/{id}/publish` | POST | 개별 기사 발행 | `{ ok: true, id, state: "published", ... }` |
| `/health` | GET | 헬스 체크 | `{ ok: true }` 또는 `"OK"` |

필요 시 `/api/debug/*` 등 추가적인 디버그/운영용 엔드포인트도 같은 형식으로 표에 추가한다.

---

### 11.2 공통 JSON 응답 스키마 정의 (②)

목적: 모든 Admin API에서 **성공/실패 응답 구조와 HTTP 코드 규칙**을 통일하기 위한 헌법을 정의한다.

#### 11.2.1 성공 응답 공통 스키마

- HTTP 코드: `200` (성공)
- JSON 형태(기본형):

```json
{
  "ok": true,
  "data": {
    "...": "엔드포인트별 payload"
  }
}
```

- 규칙
  - `ok`: 항상 `true`
  - `data`: 실제 비즈니스 데이터(엔드포인트마다 구조 상이)
  - 필요 시 페이징/메타 정보는 `data` 안 또는 `meta` 필드로 일관되게 추가한다.

- 예시
  - `/api/status`

    ```json
    {
      "ok": true,
      "data": {
        "total": 100,
        "ready_count": 80,
        "gate_required": 80,
        "ready_rate": 0.8,
        "gate_pass": true
      }
    }
    ```

  - `/api/report`

    ```json
    {
      "ok": true,
      "data": {
        "path": "reports/2025-11-15.md"
      }
    }
    ```

#### 11.2.2 실패 응답 공통 스키마

- HTTP 코드: `4xx` (클라이언트 오류) / `5xx` (서버 오류)
- JSON 형태(기본형):

```json
{
  "ok": false,
  "error": "사람이 읽을 수 있는 오류 메시지",
  "error_code": "E.ADMIN.SCOPE_CODE",
  "detail": {
    "...": "선택: 디버깅용 추가 정보"
  }
}
```

- 규칙
  - `ok`: 항상 `false`
  - `error`: 한 줄 오류 설명(한국어/영어 병행 가능, 운영자는 한국어 우선)
  - `error_code`: 기계가 읽기 좋은 짧은 코드(로그/모니터링 필터용)
  - `detail`: stack trace, 필드 오류 목록 등 선택 정보

#### 11.2.3 에러 코드 네이밍 규칙

- 접두어: `E.ADMIN.`
- 영역 코드(예시)
  - `STATUS`, `REPORT`, `EXPORT`, `ENRICH`, `LOGS`, `BACKUP`, `CONFIG`, `AUTH`, `COMMON` …
- 예시
  - `E.ADMIN.REPORT_NO_DATA`
  - `E.ADMIN.STATUS_UNAUTHORIZED`
  - `E.ADMIN.EXPORT_NOT_FOUND`

에러 코드 테이블은 별도 부록 또는 운영 문서에서 목록화할 수 있으며,
여기서는 네이밍 규칙과 대표 예시만 정의한다.

---

### 11.3 엔드포인트별 매핑 표 (③ 정의서 ↔ 실제 ↔ 타겟 스키마)

목적: 각 엔드포인트에 대해  
- 정의서/런북이 요구하는 동작,  
- 현재 구현(응답 구조),  
- 공통 스키마를 적용했을 때의 **타겟 스키마**  
를 한 줄로 비교해, 추후 구현/리팩터링 시 기준선으로 사용한다.

#### 11.3.1 매핑 표 (초안)

| path | method | 용도 | 현재 응답 구조(요약) | 타겟 스키마 (공통 규약 적용 후) |
| --- | --- | --- | --- | --- |
| `/api/status` | GET | KPI/게이트 요약 | `{ ok: true, total, ready_count, gate_required, ready_rate, gate_pass, ... }` | `200 + { ok: true, data: { total, ready_count, gate_required, ready_rate, gate_pass, ... } }` |
| `/api/report` | POST | MD/CSV 보고서 생성 | `{ ok: true, path }` | `200 + { ok: true, data: { path } }` |
| `/api/export/md` | GET | MD 보고서 다운로드 | MD 파일 스트림 또는 파일 응답 | `200 + 파일 응답` 또는 `302 + 파일 위치` (예외 정책으로 별도 정의) |
| `/api/export/csv` | GET | CSV 보고서 다운로드 | CSV 파일 스트림 또는 파일 응답 | 위와 동일 정책 적용 |
| `/api/logs` | GET | 로그 파일 목록 | `{ ok: true, logs: [...], ... }` | `200 + { ok: true, data: { logs: [{ name, size, mtime, ... }] } }` |
| `/api/logs/{name}` | GET | 로그 내용 보기 | `text/plain` 또는 `{ ok: true, content }` | `200 + { ok: true, data: { name, content } }` (대량 로그일 경우 스트리밍 전략 별도 정의) |
| `/api/logs/{name}/download` | GET | 로그 파일 다운로드 | 파일 다운로드 응답 | `200 + 파일 응답` (JSON 예외, 문서에 명시) |
| `/api/backup/status` | GET | 백업 상태 | `{ ok: true, last_success_at, last_failure_at, ... }` | `200 + { ok: true, data: { last_success_at, last_failure_at, last_success_status, ... } }` |
| `/api/config/gate_required` | GET | 게이트 기준 조회 | `{ ok: true, gate_required }` | `200 + { ok: true, data: { gate_required } }` |
| `/api/config/gate_required` | PATCH | 게이트 기준 변경 | `{ ok: true, gate_required }` | `200 + { ok: true, data: { gate_required } }` (변경 후 `/api/status` 결과와 일치) |
| `/api/items` | GET | 기사 목록 조회(Ready/대기) | `{ ok: true, items: [{ id, title, state, ... }], ... }` | `200 + { ok: true, data: { items: [...] } }` |
| `/api/items/{id}/publish` | POST | 기사 발행 | `{ ok: true, id, state: "published", ... }` | `200 + { ok: true, data: { id, state: "published", ... } }` |
| `/api/enrich/keyword` | POST | 키워드 enrich 작업 트리거 | `{ ok: true, job_id, ... }` | `200 + { ok: true, data: { job_id, ... } }` |
| `/api/enrich/selection` | POST | 선정 기사 enrich 작업 | `{ ok: true, job_id, ... }` | `200 + { ok: true, data: { job_id, ... } }` |
| `/health` | GET | 헬스 체크 | `{ ok: true }` 또는 `"OK"` | `200 + { ok: true }` (헬스도 공통 스키마 사용 권장) |

실제 구현과의 차이는 향후 서버 코드(server_quali.py 등)를 기준으로 재검증해,
필요 시 `현재 응답 구조` 열을 수정하거나, 구현을 타겟 스키마에 맞게 정렬한다.

---

### 11.4 다음 단계 메모

1. 위 표에서 `현재 응답 구조`가 실제 코드와 다른 부분은 실제 구현 기준으로 정정한다.  
2. 공통 스키마(11.2)를 FastAPI Pydantic 모델(BaseResponse 등)로 옮기고,  
   각 엔드포인트의 응답 타입을 해당 모델을 상속/활용하는 구조로 단계적으로 전환한다.  
3. 자동화 테스트에서 성공/실패 응답 스냅샷을 비교해,
   공통 스키마가 지켜지는지 회귀 테스트를 수행한다.

## 12. Domap 표준 리뷰 API 배포 파이프라인 (DoD)

### 12.1 목적

- QualiJournal Admin domap 서비스에서 **표준 리뷰 API(`/api/standards/reviews`)** 를 포함한 관리자 기능을
  안전하게 배포·검증하기 위한 표준 절차를 정의한다.
- dev 환경에서 한 번 빌드한 이미지를 **SSOT(단일 진실 공급원)** 으로 사용하고,
  prod(domap)에서는 해당 이미지만 승급(promote)하는 구조를 고정한다.

---

### 12.2 파이프라인 개요 (텍스트 플로우)

1. **트리거**
   - `main` 브랜치에 변경이 merge되거나
   - GitHub Actions에서 `ci-deploy-prod.yml`을 수동 실행한다(`workflow_dispatch`).

2. **환경 승인(Review deployments)**
   - `jobs.deploy.environment.name = production` 으로 설정.
   - GitHub Environments `production`에서 Required reviewer = `cdmacs1003-cyber`.
   - 배포 시작 전, `Review deployments → Approve and deploy` 승인 필수.

3. **인증 및 기본 세팅**
   - `actions/checkout@v4`로 소스 코드 가져오기.
   - `google-github-actions/auth@v2` + WIF(OIDC)로 GCP 인증.
   - `google-github-actions/setup-gcloud@v2`로 `gcloud` CLI 준비.

4. **Cloud SQL STOP Policy 가드레일**
   - 인스턴스: `quali-pg`, 프로젝트: `quali-journal-prod`.
   - `state`와 `activationPolicy` 조회:
     - `state != RUNNABLE` 이면 **즉시 실패** → 배포 중단.
     - `RUNNABLE` 인 경우에만 다음 단계 진행.

5. **dev 이미지 SSOT 조회**
   - env:
     - `DEV_REGION = asia-northeast3`
     - `DEV_SERVICE_NAME = quali-journal-admin-dev`
   - `gcloud run services describe quali-journal-admin-dev` 로
     `spec.template.spec.containers[0].image` 값을 읽어온다.
   - 이미지 값이 비어 있으면 실패(`ERROR: dev service image not found`).

6. **domap(prod) 배포**
   - 서비스: `quali-admin-domap`
   - 리전: `asia-northeast1`
   - `google-github-actions/deploy-cloudrun@v2` 사용.
   - `image = steps.dev-image.outputs.IMAGE` (dev에서 읽어온 이미지 그대로 사용).
   - 주요 설정:
     - `env_vars`: `PYTHONUTF8`, `PYTHONIOENCODING`, `COMMIT_SHA`, `ALLOWED_ORIGINS=https://admin.standardai.co.kr`
     - `secrets`: `ADMIN_TOKEN=ADMIN_TOKEN:latest`
     - `flags`: `--allow-unauthenticated`, `--service-account=quali-run-sa@...`
     - `update_traffic_flags: --to-latest` (최신 리비전에 100% 트래픽)

7. **Warmup**
   - `status.url`로부터 서비스 URL 확인:
     - 예: `https://quali-admin-domap-q2cq64ti3a-an.a.run.app`
   - `/health` 엔드포인트를 최대 60회(3초 간격) 호출:
     - 200 응답이 나오면 “health OK”, 다음 단계 진행.
     - 끝까지 200이 안 나오면 워크플로 실패 처리.

8. **스모크 테스트 (DoD 기준)**

   1) `/api/status`
   - 헤더:
     - `Authorization: Bearer <ADMIN_TOKEN>`
     - `X-Admin-Token: <ADMIN_TOKEN>`
   - 기대 응답: HTTP 200
   - 실패 시: 워크플로 즉시 실패.

   2) `/api/report`
   - 헤더 동일.
   - 기대 응답: HTTP 200
   - 실패 시: 워크플로 즉시 실패.

   3) `/api/standards/reviews`
   - 헤더 동일.
   - 기대 응답: HTTP 200
   - 실패 시: 워크플로 즉시 실패.
   - 이 API가 200을 반환해야 “표준 리뷰 API 배포 완료”로 간주한다.

---

### 12.3 DoD(Definition of Done)

Domap 표준 리뷰 배포는 아래 조건을 모두 만족해야 **완료(Done)** 로 인정한다.

1. **코드 레벨**
   - `ci-deploy-prod.yml` 의 `image` 값이 `steps.dev-image.outputs.IMAGE` 를 사용한다.
   - `DEV_REGION`, `DEV_SERVICE_NAME` env 값이 헌법(SSOT)과 일치한다.
   - `jobs.deploy.environment.name = production` 으로 설정되어 있다.

2. **CI 레벨**
   - Admin 테스트(`ci-test-admin.yml` → job `test-admin`)가 성공 상태여야 한다.
   - SSOT guard 워크플로(별도 정의)에서 “상태 = PASS” 여야 한다.

3. **배포 레벨**
   - Cloud Run 배포 로그상:
     - Cloud SQL state=RUNNABLE 확인.
     - dev 서비스에서 IMAGE 값을 정상 읽어온 기록.
     - deploy-cloudrun step이 에러 없이 완료됨.

4. **런타임 레벨(스모크)**
   - `/health` 200
   - `/api/status` 200
   - `/api/report` 200
   - `/api/standards/reviews` 200

위 4개 축을 모두 만족하지 못하면, 배포는 “실패” 또는 “부분 성공(HOLD)”로 간주하고 사람이 원인 분석 후 재시도한다.

---

### 12.4 SSOT guard 체크 항목 (초안)

> ※ 기존 SSOT 체크 워크플로 실패를 반영한 초안이며,  
> 추후 별도 문서에서 상세 규칙 + 예외 케이스를 정의한다.

SSOT guard 워크플로는 다음 조건을 검증해야 한다.

1. `ci-deploy-prod.yml` 안에 **단일 deploy job**만 존재.
2. env 블록에 다음 키가 존재:
   - `PROJECT_ID`, `REGION`, `SERVICE_NAME`, `DEV_REGION`, `DEV_SERVICE_NAME`, `CLOUD_SQL_INSTANCE`
3. `Deploy to Cloud Run (image)` step에서 사용하는 `image` 값은 **env가 아닌 `steps.dev-image.outputs.IMAGE`** 여야 한다.
4. Smoke 테스트 이름과 대상 경로:
   - `Smoke tests (/api/status, /api/report)`
   - `Smoke tests (/api/standards/reviews)`
5. `jobs.deploy.environment.name` 가 `production` 인지 확인.

위 항목 중 하나라도 불일치하면 PR에서 SSOT guard 체크를 **FAIL** 처리하고,  
사람이 도큐먼트(본 섹션)와 YAML을 다시 비교하도록 요구한다.


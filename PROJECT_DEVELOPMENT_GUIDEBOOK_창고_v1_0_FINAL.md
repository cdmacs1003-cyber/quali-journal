# PROJECT_DEVELOPMENT_GUIDEBOOK_창고 v1.0 FINAL

## 문서 식별

| 항목 | 내용 |
|---|---|
| 문서명 | PROJECT_DEVELOPMENT_GUIDEBOOK_창고 v1.0 FINAL |
| 문서 성격 | 퀄리AI/QLIB Warehouse 개발·운영·승격·봉인 실행 가이드북 |
| 기준일 | 2026-05-13 |
| 적용 범위 | QualiJournal / QualiAI Warehouse, 도서관 승격 전 자료 보관·검역·검수·승격 준비 체계 |
| 상위 기준 | 최상위 개발 헌법, 퀄리AI 공통 개발 가이드북 |
| 문서 지위 | 상위 헌법 하위의 창고 개발 실행 SSOT v1.0 |
| 변경 원칙 | 수정 불가 고정본으로 사용하되, 변경이 필요하면 새 버전 또는 부록 Addendum으로만 갱신 |

---

## 고정 운영 관리 메모

본 문서는 창고 개발 기준본 v1.0으로 고정하여 사용할 수 있다. 단, 고정본의 안정성을 유지하기 위해 다음 관리 원칙을 함께 적용한다.

1. `PROJECT_DEVELOPMENT_MEMORY.md`가 실제 프로젝트 기준 문서로 존재할 경우, 출처 문서 목록과 작업 시작 전 확인 문서에 반드시 포함한다.
2. 공식 테스트 명령, 배포 대상 환경, 배포 승인자, 롤백 승인자는 이 문서 본문을 직접 수정하지 않고 별도 `ADDENDUM_YYYYMMDD.md` 또는 `PROJECT_DEVELOPMENT_GUIDEBOOK_창고_v1_1.md`에서 확정한다.
3. v1.0 고정본은 임의 수정하지 않는다. 운영 중 보완이 필요하면 원본을 직접 덮어쓰지 않고 Addendum 또는 새 버전으로 변경 이력, 승인 근거, 적용 범위를 분리 기록한다.

관리 판정:

```text
FREEZE_DECISION=PASS
USE_AS_WAREHOUSE_DEVELOPMENT_GUIDEBOOK=YES
FREEZE_VERSION=v1.0
CHANGE_POLICY=ADDENDUM_OR_NEW_VERSION_ONLY
PROJECT_MEMORY_LINK=REQUIRED_IF_EXISTS
DEPLOYMENT_AND_TEST_COMMANDS=ADDENDUM_REQUIRED
```

---

## 0. 최종 판정

```text
GUIDEBOOK_DECISION=PASS_TO_FREEZE_AS_WAREHOUSE_DEVELOPMENT_GUIDEBOOK_V1_0
ROLE=Warehouse project development and operation guide
COMMON_ALIGNMENT=PASS
QLIB_ALIGNMENT=PASS
WAREHOUSE_ROLE_COMPLETENESS=PASS
IMMUTABLE_FREEZE_ALLOWED=YES
USE_SCOPE=Warehouse Manifest / Item Schema / Review / Approval / Promotion Trace / Backup / Deployment Gate / Handover
```

본 문서는 기존 `PROJECT_DEVELOPMENT_GUIDEBOOK_창고.md`의 실행 절차서 구조를 유지하면서, 창고 역할 완성에 필요한 Warehouse Manifest, Item Schema, Promotion Trace, Warehouse Gate, Backup/Restore, Release Board, Quality Threshold를 본문 기준으로 고정한 완성본이다.

본 문서는 헌법이 아니다. 상위 헌법과 퀄리AI 공통 가이드북을 대체하지 않는다. 본 문서는 창고 개발자가 실제 구현·검증·운영·배포·인수인계를 수행할 때 따르는 창고 전용 실행 기준이다.

---

## 1. 최상위 원칙

### 1.1 루트 구조

```text
QLIB_ROOT=Quali Library
WAREHOUSE=Library promotion pre-stage quarantine and evidence store
LIBRARY_CORE=Approved canonical knowledge store
BRIDGE=Warehouse-to-Library promotion and UI/operation bridge
SKILLUP=Education and training application layer
```

1. 퀄리도서관은 루트다.
2. 창고는 도서관 승격 전 자료를 보관·검역·검수·증거화하는 장소다.
3. 창고는 루트가 아니며, 창고 자료는 승인 전까지 도서관 정본이 아니다.
4. 창고의 목적은 자료를 무한히 쌓는 것이 아니라, 도서관 승격 가능성을 판정할 수 있게 원본·근거·상태·리뷰·해시·trace를 보존하는 것이다.
5. 장기기억과 ProofPack은 창고에서 도서관으로 이어지는 결정과 증거를 보존한다.

### 1.2 사용자 최신 요청과 상위 헌법의 관계

사용자의 최신 명시 요청은 작업 범위를 구체화할 수 있다. 그러나 다음 상위 안전 규칙은 어떠한 최신 요청으로도 완화할 수 없다.

- 승인 없는 삭제 금지
- 승인 없는 배포 금지
- 승인 없는 외부 호출 또는 과금 발생 작업 금지
- 비밀 값, API key, 토큰, 개인정보, 유료 표준 원문 노출 금지
- 증거 없는 PASS 금지
- 검증하지 않은 항목은 `NOT_VERIFIED`로 기록
- 실행하지 않은 항목은 `NOT_EXECUTED`로 기록
- 창고 자료를 도서관 정본처럼 표시 금지

### 1.3 독립 제품화 금지

창고 UI, API, 자동화, 리포트, 수집 기능은 창고 상태 확인, 게이트 복구, 리뷰 증거 확보, 리포팅 증거 확보, 도서관 승격 준비를 위한 수단이다. 창고 자체를 독립 제품처럼 계속 확장하지 않는다.

---

## 2. 창고의 최종 정의

### 2.1 창고란 무엇인가

창고는 다음 자료를 도서관 승격 전 단계에서 보관하는 검역소다.

| 자료 유형 | 설명 | 도서관 승격 가능 여부 |
|---|---|---|
| 전문가 암묵지 | 전문가의 경험, 노하우, 구두 설명, 판단 기준 | 리뷰·근거화 후 가능 |
| 현장 노하우 | 공정, 검사, 교육, 고객 대응 과정에서 나온 실제 경험 | 사례화·익명화 후 가능 |
| 개인 논문 | 개인 연구, 비공식 논문, 초안, 기술 보고서 | 권리·근거 확인 후 가능 |
| 리포트 | 조사 보고서, 분석 보고서, 운영 보고서 | 출처·검토 후 가능 |
| 기고문 | 커뮤니티 기고, 사내 기고, 기술 칼럼 | 권리·검토 후 가능 |
| 실패 기록 | 오류, 배포 실패, 검증 실패, HOLD 기록 | 교훈/재발방지 카드로 가능 |
| 리뷰 메모 | 검수자 판단, 보류 사유, 승인 근거 | 승인 trace 근거로 가능 |
| 표준 해석 메모 | 조항 해석, 적용 조건, 예외 판단 | Standard/Reference 연결 후 가능 |
| 교육 개선 후보 | 교육 콘텐츠 후보, 질문, 오해 사례 | Skillup 콘텐츠 후보로 가능 |

### 2.2 창고가 아닌 것

창고는 다음이 아니다.

- 공식 도서관 정본
- 유료 표준 원문 저장소
- 무제한 뉴스 수집 제품
- 공개 콘텐츠 발행 플랫폼
- 검증 없는 AI 답변 데이터셋
- 개인정보 또는 비밀 값 보관소
- UI 기능 확장 실험장

---

## 3. 공식 Warehouse Manifest

### 3.1 공식 루트

모든 경로는 프로젝트 루트 기준 상대 경로로 고정한다.

```text
PROJECT_ROOT=H:\a\퀄리저널_pr_clean
WAREHOUSE_MANIFEST=data/warehouse/warehouse_manifest.json
WAREHOUSE_ROOT=data/warehouse
RAW_ROOT=data/warehouse/raw
DERIVED_ROOT=data/warehouse/derived
TRACE_ROOT=data/warehouse/trace
SCHEMA_ROOT=data/warehouse/schema
REVIEW_ROOT=data/warehouse/review
BACKUP_ROOT=backup/warehouse
PROOFPACK_ROOT=reports/proofpacks/warehouse
LOG_ROOT=logs/warehouse
RELEASE_ROOT=releases/warehouse
HANDOVER_ROOT=handover/warehouse
```

### 3.2 Manifest 필수 필드

`data/warehouse/warehouse_manifest.json`은 아래 필드를 반드시 가진다.

```json
{
  "manifest_version": "1.0",
  "project_name": "QualiAI Warehouse",
  "project_root": "H:\\a\\퀄리저널_pr_clean",
  "warehouse_root": "data/warehouse",
  "raw_root": "data/warehouse/raw",
  "derived_root": "data/warehouse/derived",
  "trace_root": "data/warehouse/trace",
  "schema_root": "data/warehouse/schema",
  "review_root": "data/warehouse/review",
  "backup_root": "backup/warehouse",
  "proofpack_root": "reports/proofpacks/warehouse",
  "log_root": "logs/warehouse",
  "release_root": "releases/warehouse",
  "handover_root": "handover/warehouse",
  "required_item_fields": [
    "warehouse_item_id",
    "item_type",
    "status",
    "raw_hash",
    "provenance",
    "rights_status",
    "sensitivity",
    "visibility",
    "created_at",
    "updated_at"
  ],
  "required_promotion_fields": [
    "promotion_trace_id",
    "warehouse_item_id",
    "raw_hash",
    "promoted_library_id",
    "validation_result",
    "policy_result"
  ],
  "approval_required": true,
  "promotion_dry_run_required": true,
  "backup_restore_test_required": true
}
```

### 3.3 Manifest PASS 기준

| 항목 | PASS 기준 |
|---|---|
| 경로 존재 | 모든 root 경로가 존재하거나 생성 계획이 있다 |
| schema 존재 | item schema와 promotion trace schema가 존재한다 |
| raw 불변성 | raw 파일 직접 수정 금지 규칙이 있다 |
| review 필수 | 승인 상태로 가려면 review 기록이 필요하다 |
| dry-run 필수 | 승격 전 promotion dry-run이 필요하다 |
| backup 필수 | 창고 전체 백업과 복구 dry-run이 필요하다 |

---

## 4. Warehouse Item Schema

### 4.1 item_type

창고 항목 유형은 아래 중 하나로 고정한다.

| item_type | 설명 |
|---|---|
| `tacit_knowledge` | 전문가 암묵지, 구두 노하우, 경험 기반 판단 |
| `expert_knowhow` | 전문가가 정리한 실무 기술, 공정 노하우 |
| `personal_paper` | 개인 논문, 연구 초안, 비공식 학술 자료 |
| `report` | 조사 보고서, 분석 보고서, 운영 보고서 |
| `reporter_note` | 리포터 메모, 취재 메모, 현장 기록 |
| `contribution` | 기고문, 칼럼, 외부 기여 자료 |
| `community_contribution` | 커뮤니티 글, 토론, 질의응답 |
| `failure_record` | 오류, 실패, HOLD, 배포 실패 기록 |
| `review_memo` | 검수 메모, 승인/보류 판단 기록 |
| `standard_note` | 표준 해석, 적용 메모, 조항 관련 판단 |
| `field_case` | 현장 사례, 고객 사례, 공정 사례 |
| `education_seed` | 스킬업 교육 후보, 교육 질문, 오해 사례 |
| `analytics_improvement_candidate` | 사용성·분석 개선 후보 |
| `raw_document` | 분류 전 원본 문서 |

### 4.2 status lifecycle

상태값은 아래만 사용한다.

```text
captured
untriaged
triaged
needs_source
pending_review
private_tacit
hold
duplicate_suspected
rejected
archived
approved_for_library
promoted
```

### 4.3 상태 의미

| status | 의미 | 도서관 승격 가능 여부 |
|---|---|---|
| `captured` | 원본 수집 완료 | 불가 |
| `untriaged` | 아직 분류 전 | 불가 |
| `triaged` | 기본 분류 완료 | 불가 |
| `needs_source` | 출처 보강 필요 | 불가 |
| `pending_review` | 검수 대기 | 불가 |
| `private_tacit` | 비공개 암묵지 | 원칙상 불가, 익명화 후 재검토 |
| `hold` | 보류 | 불가 |
| `duplicate_suspected` | 중복 의심 | 불가 |
| `rejected` | 거절 | 불가 |
| `archived` | 보관 종료 | 불가 |
| `approved_for_library` | 도서관 승격 승인 | dry-run 후 가능 |
| `promoted` | 도서관 승격 완료 | 완료 상태 |

### 4.4 허용 상태 전이

```text
captured -> untriaged
untriaged -> triaged
triaged -> needs_source
triaged -> pending_review
triaged -> private_tacit
triaged -> duplicate_suspected
triaged -> rejected
needs_source -> pending_review
pending_review -> hold
pending_review -> rejected
pending_review -> approved_for_library
approved_for_library -> promoted
hold -> pending_review
hold -> archived
duplicate_suspected -> archived
duplicate_suspected -> pending_review
rejected -> archived
```

아래 전이는 금지한다.

```text
captured -> approved_for_library
untriaged -> approved_for_library
triaged -> promoted
pending_review -> promoted
hold -> promoted
rejected -> promoted
private_tacit -> public export
secret sensitivity -> public export
```

### 4.5 필수 item 필드

```yaml
warehouse_item_id: string
item_type: enum
status: enum
title: string
summary: string
raw_text_ref: string
raw_hash: sha256
raw_mime_type: string
raw_size_bytes: integer
provenance:
  source_type: enum
  source_title: string
  source_author: string
  source_org: string
  source_date: string
  captured_by: string
  captured_at: string
  source_locator_hash: string
rights_status: enum
sensitivity: enum
visibility: enum
quality_score: integer
confidence: enum
review:
  reviewer_id: string
  review_date: string
  review_note: string
  decision: enum
  approver_id: string
  approval_date: string
promotion:
  promotion_target: enum
  promotion_dry_run_id: string
  promoted_library_id: string
  promoted_graph_node_id: string
  promotion_trace_id: string
created_at: string
updated_at: string
```

### 4.6 rights_status

```text
owned
licensed
permission_granted
public_reference
internal_only
no_export
unknown
```

`unknown`은 승격 금지다. `no_export`는 공개 report, 공개 교육자료, 공개 도서관 카드로 내보낼 수 없다.

### 4.7 sensitivity

```text
public
internal
restricted
private
secret
```

`secret`은 도서관 승격과 공개 export가 금지된다. `private`와 `restricted`는 익명화·권리 검토 후 별도 승인 없이는 승격할 수 없다.

### 4.8 visibility

```text
warehouse_only
reviewer_only
library_candidate
library_internal
public_summary_allowed
no_export
```

`warehouse_only`, `reviewer_only`, `no_export`는 공개 UI나 공개 리포트에 표시하지 않는다.

---

## 5. 원본 보존 원칙

### 5.1 raw immutable

원본은 수정하지 않는다. 오탈자 수정, 요약, 번역, 재작성, 분류 결과는 derived에 저장한다.

```text
RAW_EDIT_ALLOWED=NO
DERIVED_EDIT_ALLOWED=YES_WITH_TRACE
RAW_HASH_REQUIRED=YES
```

### 5.2 raw 저장 규칙

| 항목 | 규칙 |
|---|---|
| raw path | `data/warehouse/raw/YYYY/MM/warehouse_item_id.ext` |
| raw hash | SHA256 필수 |
| raw metadata | item JSON에 별도 기록 |
| raw 삭제 | 금지. 삭제가 필요하면 archive 또는 redaction trace 사용 |
| raw 공개 | 금지. 공개 가능한 요약만 별도 생성 |

### 5.3 derived 저장 규칙

| 항목 | 규칙 |
|---|---|
| 요약 | `data/warehouse/derived/summary/` |
| 번역 | `data/warehouse/derived/translation/` |
| 리뷰 | `data/warehouse/review/` |
| 검증 결과 | `data/warehouse/derived/validation/` |
| 승격 dry-run | `data/warehouse/trace/dry_run/` |

---

## 6. 품질 기준과 도서관 승격 기준

### 6.1 승격 Hard Gate

아래 조건 중 하나라도 실패하면 `approved_for_library`가 될 수 없다.

| Gate | 조건 | 실패 시 상태 |
|---|---|---|
| H-G1 Raw | raw_hash 존재 | `hold` |
| H-G2 Provenance | provenance 필드 완성 | `needs_source` |
| H-G3 Rights | rights_status가 `unknown` 아님 | `needs_source` |
| H-G4 Sensitivity | sensitivity가 `secret` 아님 | `hold` |
| H-G5 Review | reviewer_id, review_note 존재 | `pending_review` |
| H-G6 Approval | approver_id 존재 | `pending_review` |
| H-G7 Dry-run | promotion dry-run PASS | `approved_for_library` 유지, 승격 보류 |
| H-G8 Export Rule | no_export 항목 공개 export 금지 | `hold` |
| H-G9 Standard Safety | 유료 표준 원문 장문 출력 금지 | `hold` |
| H-G10 Trace | promotion_trace 생성 가능 | `approved_for_library` 유지, 승격 보류 |

### 6.2 quality_score 산정

총점 100점 기준으로 산정한다.

| 항목 | 배점 | 설명 |
|---|---:|---|
| source_quality | 20 | 출처 신뢰도, 작성자 전문성, 기관성 |
| field_value | 20 | 현장 적용 가치, 교육 가치, 표준 적용 가치 |
| evidence_strength | 20 | 근거 수, 원문성, 재현 가능성 |
| review_confidence | 15 | 검수자 신뢰도, 이견 여부 |
| rights_clearance | 10 | 권리 상태 명확성 |
| promotion_fit | 15 | 도서관 카드, Reference, 교육자료로의 적합성 |

### 6.3 승격 점수 기준

| 점수 | 판정 | 상태 |
|---:|---|---|
| 90-100 | 강한 승격 후보 | `approved_for_library` 가능 |
| 80-89 | 승격 후보 | `approved_for_library` 가능 |
| 60-79 | 추가 검토 | `hold` 또는 `pending_review` |
| 40-59 | 낮은 신뢰 | `hold` 또는 `rejected` |
| 0-39 | 승격 부적합 | `rejected` |

단, Hard Gate가 실패하면 점수와 무관하게 승격 금지다.

### 6.4 LOW_CONFIDENCE 기준

의미 매칭, 표준 조항 연결, 출처 신뢰도, 리뷰 신뢰도 중 핵심 유사도 또는 신뢰도가 0.85 미만이면 `LOW_CONFIDENCE`로 표시한다. 이 경우 사람이 리뷰하기 전에는 도서관 승격이 금지된다.

```text
LOW_CONFIDENCE_THRESHOLD=0.85
LOW_CONFIDENCE_PROMOTION_ALLOWED=NO
HUMAN_REVIEW_REQUIRED=YES
```

---

## 7. Review와 Approval 계약

### 7.1 reviewer 역할

| 역할 | 권한 |
|---|---|
| Capturer | 원본 등록, captured 상태 생성 |
| Triage Reviewer | item_type, status 1차 분류 |
| Subject Reviewer | 기술 내용 검수, 품질 점수 부여 |
| Rights Reviewer | 권리, 공개 가능성, no_export 판단 |
| Approver | approved_for_library 승인 |
| Librarian | 도서관 승격 실행 및 trace 봉인 |
| Operator | 백업, 복구, 배포 게이트 확인 |
| Developer | 구현, 테스트, validator 작성 |

### 7.2 review record 필수 필드

```yaml
review_id: string
warehouse_item_id: string
reviewer_id: string
reviewer_role: string
review_date: string
review_decision: enum
review_note: string
quality_score: integer
confidence: enum
rights_status_confirmed: boolean
sensitivity_confirmed: boolean
promotion_recommendation: enum
```

### 7.3 review_decision

```text
needs_source
pending_review
hold
rejected
approved_for_library
```

### 7.4 승인 금지 조건

다음 경우 승인자는 `approved_for_library`로 변경할 수 없다.

- raw_hash가 없다.
- provenance가 불완전하다.
- rights_status가 `unknown`이다.
- sensitivity가 `secret`이다.
- visibility가 `no_export`인데 공개 산출물이 목적이다.
- review_note가 비어 있다.
- approver_id가 없다.
- quality_score가 80 미만이다.
- LOW_CONFIDENCE 상태가 해소되지 않았다.

---

## 8. Promotion Trace 계약

### 8.1 승격은 복사가 아니라 검증된 변환이다

창고 항목을 도서관으로 올리는 행위는 단순 파일 복사가 아니다. 승격은 검증, 권리 확인, 리뷰, dry-run, trace 생성, 도서관 등록, 그래프 연결, evidence 봉인을 포함하는 절차다.

### 8.2 Promotion Trace 필수 필드

```yaml
promotion_trace_id: string
warehouse_item_id: string
warehouse_item_hash: sha256
raw_hash: sha256
source_item_status: approved_for_library
promotion_target: enum
promoted_library_id: string
promoted_graph_node_id: string
promoted_evidence_ids: list
validation_result:
  manifest_pass: boolean
  raw_pass: boolean
  provenance_pass: boolean
  rights_pass: boolean
  sensitivity_pass: boolean
  review_pass: boolean
  dry_run_pass: boolean
policy_result:
  no_secret: boolean
  no_paid_standard_long_quote: boolean
  no_private_export: boolean
  no_internal_path_public_output: boolean
output_artifacts:
  library_card_path: string
  graph_node_path: string
  evidence_card_path: string
  proofpack_path: string
created_by: string
created_at: string
```

### 8.3 promotion_target

```text
library_standard_card
library_reference_card
library_case_card
library_training_seed
library_failure_lesson
library_tailoring_note
library_graph_node_only
```

### 8.4 Promotion dry-run

승격 전 반드시 dry-run을 수행한다.

Dry-run은 다음을 검사한다.

- warehouse_item_id 존재
- raw_hash 존재 및 일치
- status = approved_for_library
- rights_status 승격 가능
- sensitivity 승격 가능
- quality_score >= 80
- LOW_CONFIDENCE 아님
- library target 생성 가능
- graph node 연결 가능
- output artifact 경로 생성 가능
- 공개 출력 금지 위반 없음

Dry-run PASS 전에는 `promoted` 상태로 바꿀 수 없다.

---

## 9. Warehouse API 계약

### 9.1 API 목록

| API | 목적 | 위험도 | 필수 게이트 |
|---|---|---|---|
| `create_item` | 창고 항목 등록 | L2 | raw_hash, provenance |
| `list_items` | 상태별 목록 조회 | L1 | auth, visibility |
| `read_item` | 개별 항목 조회 | L1 | auth, visibility |
| `update_status` | 상태 변경 | L3 | state transition validator |
| `add_review` | 리뷰 추가 | L3 | reviewer role |
| `approve_for_library` | 도서관 승격 승인 | L4 | Hard Gate, quality_score |
| `promotion_dry_run` | 승격 사전 검증 | L4 | promotion validator |
| `promote` | 도서관 승격 실행 | L5 | dry-run PASS, approval |
| `explain_trace` | 역추적 설명 | L2 | trace exists |
| `backup_run` | 창고 백업 | L4 | backup manifest |
| `restore_dry_run` | 복구 검증 | L4 | restore validator |
| `validate_manifest` | manifest 검증 | L2 | schema |
| `validate_item` | item 검증 | L2 | item schema |
| `validate_trace` | trace 검증 | L2 | trace schema |
| `warehouse_status` | 창고 상태 | L1 | count/state |

### 9.2 API 공통 금지

- secret 값을 응답에 포함 금지
- raw 원문을 공개 응답에 포함 금지
- 유료 표준 원문 장문 반환 금지
- internal path를 외부 사용자용 응답에 포함 금지
- no_export 항목 export 금지
- 승인 전 항목을 library item으로 표시 금지

---

## 10. Warehouse UI 계약

### 10.1 UI 목적

Warehouse UI는 창고 항목을 등록, 분류, 리뷰, 승인, dry-run, trace 확인하는 조작창이다. 독립 제품 UI가 아니다.

### 10.2 UI 필수 화면

| 화면 | 목적 |
|---|---|
| Inbox | captured/untriaged 항목 확인 |
| Triage | item_type, status 1차 분류 |
| Review Queue | pending_review, needs_source, hold 확인 |
| Approval Board | approved_for_library 후보 검토 |
| Promotion Dry-run | 승격 전 검증 결과 확인 |
| Trace Viewer | warehouse item과 library item 연결 확인 |
| Backup/Restore Status | 백업 및 복구 dry-run 확인 |
| Release Board | 배포 전 W-Gate 상태 확인 |

### 10.3 UI 금지

- 승인 전 항목을 도서관 정본처럼 표시 금지
- `private_tacit`, `secret`, `no_export` 항목을 공개 리포트에 노출 금지
- Warehouse UI에서 공개 publish 버튼 제공 금지
- 품질 점수 없이 승인 버튼 활성화 금지
- dry-run 없이 promote 버튼 활성화 금지
- 내부 경로, secret, token, API key 표시 금지
- 유료 표준 원문 장문 미리보기 금지

---

## 11. Warehouse Validator

### 11.1 Validator 목록

| Validator | 검사 대상 | PASS 기준 |
|---|---|---|
| `validate_manifest` | warehouse_manifest.json | 모든 루트와 필수 설정 존재 |
| `validate_item_schema` | item JSON | 필수 필드 존재, enum 유효 |
| `validate_raw_hash` | raw 파일 | hash 일치 |
| `validate_provenance` | provenance | 출처 정보 충분 |
| `validate_rights` | rights_status | unknown 아님 |
| `validate_sensitivity` | sensitivity | secret 승격 차단 |
| `validate_review` | review record | reviewer, note, decision 존재 |
| `validate_quality` | quality_score | hard gate + 점수 기준 |
| `validate_state_transition` | status 변경 | 허용 전이만 가능 |
| `validate_promotion_dry_run` | 승격 후보 | 승격 전 조건 PASS |
| `validate_promotion_trace` | trace | 역추적 가능 |
| `validate_backup` | backup set | manifest, raw, trace 포함 |
| `validate_restore_dry_run` | backup 복구 | restore 가능 |
| `validate_security_scan` | 산출물 | secret, token, 민감정보 없음 |

### 11.2 Validator 출력 형식

```json
{
  "ok": true,
  "validator": "validate_item_schema",
  "target": "warehouse_item_id",
  "checked_at": "YYYY-MM-DDTHH:MM:SS",
  "issues": [],
  "decision": "PASS"
}
```

실패 시:

```json
{
  "ok": false,
  "validator": "validate_item_schema",
  "target": "warehouse_item_id",
  "issues": [
    {
      "code": "RAW_HASH_MISSING",
      "severity": "BLOCKER",
      "message": "raw_hash is required before review approval"
    }
  ],
  "decision": "HOLD"
}
```

---

## 12. Warehouse Gate

### 12.1 Gate 목록

| Gate | 이름 | PASS 기준 |
|---|---|---|
| W-G1 | Manifest Gate | manifest 존재, 루트 경로, schema 참조 PASS |
| W-G2 | Raw Gate | raw 파일 존재, raw_hash 일치, raw 불변성 PASS |
| W-G3 | Provenance Gate | source/provenance/rights/sensitivity PASS |
| W-G4 | Review Gate | review state machine, reviewer note, quality score PASS |
| W-G5 | Approval Gate | approver_id, approval note, Hard Gate PASS |
| W-G6 | Promotion Gate | promotion dry-run, trace, library target PASS |
| W-G7 | Backup Gate | full backup, restore dry-run, backup manifest PASS |
| W-G8 | Security Gate | secret/token/개인정보/internal path scan PASS |
| W-G9 | Release Gate | release board, smoke, rollback, handover PASS |

### 12.2 Gate 실패 처리

| 실패 유형 | 조치 |
|---|---|
| BLOCKER | 즉시 중단, HOLD 기록 |
| HIGH | 승격/배포 중단, 원인 분석 |
| MEDIUM | 보류 또는 제한 승인 |
| LOW | 기록 후 후속 개선 |
| UNKNOWN | PASS 금지, 확인 필요 |

---

## 13. Backup과 Restore

### 13.1 리포트 export는 전체 백업이 아니다

Markdown, CSV, HTML report는 산출물 증거일 뿐이다. 창고 전체 백업은 raw, derived, trace, schema, manifest, review, release board, proofpack을 포함해야 한다.

### 13.2 Full backup 포함 항목

| 항목 | 포함 여부 |
|---|---|
| warehouse_manifest.json | 필수 |
| item schema | 필수 |
| promotion trace schema | 필수 |
| raw files | 필수 |
| item JSON | 필수 |
| derived summaries | 필수 |
| review records | 필수 |
| promotion traces | 필수 |
| validator outputs | 필수 |
| proofpacks | 필수 |
| release board | 필수 |
| logs | 권장 |
| report exports | 권장 |

### 13.3 backup manifest

```yaml
backup_id: string
backup_date: string
backup_scope: full_warehouse
included_roots:
  - data/warehouse/raw
  - data/warehouse/derived
  - data/warehouse/trace
  - data/warehouse/schema
  - data/warehouse/review
  - reports/proofpacks/warehouse
  - releases/warehouse
file_count: integer
total_bytes: integer
sha256_manifest: string
created_by: string
restore_dry_run_required: true
```

### 13.4 restore dry-run PASS 기준

- backup manifest 로드 가능
- raw file count 일치
- item JSON count 일치
- trace count 일치
- schema 로드 가능
- validator 재실행 가능
- index/proofpack 연결 재현 가능
- secret 노출 없음

---

## 14. Release Board

### 14.1 Release Board 목적

Release Board는 창고 개발, 배포, 운영 변경이 상위 헌법과 Warehouse Gate를 통과했는지 기록하는 단일 운영 표다.

### 14.2 Release Board 필수 항목

| 필드 | 설명 |
|---|---|
| release_id | 릴리즈 식별자 |
| date | 기준일 |
| scope | 변경 범위 |
| changed_files | 변경 파일 |
| validators | 실행한 validator |
| gate_results | W-G1~W-G9 결과 |
| test_results | 테스트 결과 |
| backup_id | 백업 식별자 |
| rollback_plan | 롤백 계획 |
| decision | PASS / HOLD / ROLLBACK |
| approver | 승인자 |
| handover_path | 인수인계 문서 경로 |

### 14.3 Release PASS 조건

```text
W-G1 Manifest PASS
W-G2 Raw PASS
W-G3 Provenance PASS
W-G4 Review PASS
W-G5 Approval PASS
W-G6 Promotion PASS 또는 NOT_IN_SCOPE
W-G7 Backup PASS
W-G8 Security PASS
W-G9 Release PASS
```

`NOT_IN_SCOPE`는 해당 릴리즈에서 승격 기능을 다루지 않는 경우에만 허용한다. 이 경우 이유를 기록한다.

---

## 15. 개발 작업 절차

### 15.1 기본 순서

1. 작업 요청을 해석한다.
2. 상위 문서와 충돌 여부를 확인한다.
3. 작업 모드를 판단한다.
4. 변경 규모를 판단한다.
5. 영향 파일을 식별한다.
6. 수정 전 상태와 해시를 기록한다.
7. 최소 변경안을 작성한다.
8. 사용자 승인 필요 여부를 확인한다.
9. 승인 후 구현한다.
10. 자체 점검을 수행한다.
11. validator를 실행한다.
12. 테스트를 수행한다.
13. 증거를 저장한다.
14. Release Board를 갱신한다.
15. 인수인계를 작성한다.
16. 다음 작업 1개를 정리한다.

### 15.2 작업 모드

| 모드 | 목적 | 허용 작업 | 금지 작업 | 승인 |
|---|---|---|---|---|
| MODE 0 | 질문/분석 | 분석, 위험 식별 | 파일 수정, 명령 실행 | 불필요 |
| MODE 1 | 문서화 | 문서 초안, 설계 | 승인 없는 저장 | 저장 전 필요 |
| MODE 2 | 읽기 전용 점검 | 상태 확인 | 수정, 삭제, 배포 | 읽기 승인 필요 |
| MODE 3 | 제한 구현 | 승인된 최소 변경 | 범위 밖 수정 | 명시 승인 필요 |
| MODE 4 | 위험 실행 | 배포, 삭제, 권한, 과금 | 승인 없는 실행 | 별도 명시 승인 필수 |

### 15.3 변경 규모

| 등급 | 예시 | 승인 | 백업 | 테스트 | 롤백 |
|---|---|---|---|---|---|
| L0 | 질문, 분석 | 불필요 | 아니오 | 없음 | 아니오 |
| L1 | 문서 작성 | 저장 시 필요 | 권장 | 문서 점검 | 권장 |
| L2 | 단일 파일 수정 | 필요 | 권장 | 단위 테스트 | 예 |
| L3 | 여러 파일 수정 | 필요 | 예 | 단위+통합 | 예 |
| L4 | schema, DB, CI | 강한 승인 | 필수 | 전체 회귀 | 필수 |
| L5 | 배포, 삭제, 권한, 과금 | 별도 승인 | 필수 | 배포 전후 검증 | 필수 |

---

## 16. 코드 수정 원칙

- 비파괴 변경을 우선한다.
- additive only를 우선한다.
- 대규모 리팩터링은 금지한다.
- 사용자 승인 없는 삭제는 금지한다.
- raw 원본 직접 수정은 금지한다.
- 변경 파일 수를 최소화한다.
- 변경 이유를 기록한다.
- 변경 전후 차이를 설명한다.
- 불확실한 코드는 `🟡확인 필요`로 표시한다.
- Secret, 개인정보, 유료 표준 원문은 코드·로그·문서에 노출하지 않는다.
- 승인 전 항목을 도서관 정본으로 처리하지 않는다.

---

## 17. 테스트 절차

| 테스트 | 목적 | 실행 조건 | 기대 결과 | 실패 시 조치 |
|---|---|---|---|---|
| 정적 검사 | 문법/타입/구조 오류 확인 | 구현 후 | 오류 없음 | 수정 후 재검사 |
| 단위 테스트 | 함수 단위 검증 | 관련 코드 변경 후 | 관련 테스트 PASS | 원인 분석 |
| 통합 테스트 | API/파일 흐름 검증 | 여러 모듈 변경 후 | 주요 흐름 PASS | 변경 범위 축소 |
| Warehouse Validator | 창고 계약 검증 | schema/상태/승격 관련 변경 후 | W-Gate PASS | HOLD |
| 스모크 테스트 | 서버/핵심 API 확인 | 배포 전후 | health/API 정상 | 배포 중단 |
| 실패 테스트 | 잘못된 입력 처리 확인 | API/검수 로직 변경 후 | 안전한 오류 응답 | 예외 처리 보강 |
| 회귀 테스트 | 기존 기능 보호 | 리팩터링 후 | 기존 기능 유지 | 롤백 검토 |
| 백업 복구 dry-run | 복구 가능성 확인 | 배포/릴리즈 전 | restore dry-run PASS | 배포 중단 |

---

## 18. 배포 절차

### 18.1 배포 전 필수 확인

- [ ] 사용자 명시 승인
- [ ] 배포 대상 환경 확정
- [ ] 변경 범위 확정
- [ ] W-Gate 결과 확인
- [ ] 테스트 결과 확인
- [ ] 백업 완료
- [ ] restore dry-run PASS
- [ ] rollback plan 존재
- [ ] Release Board 갱신
- [ ] Secret 노출 scan PASS
- [ ] 인수인계 초안 작성

### 18.2 배포 후 확인

- [ ] health check
- [ ] warehouse_status
- [ ] list_items state count
- [ ] review queue count
- [ ] approval board count
- [ ] promotion dry-run sample
- [ ] backup status
- [ ] logs error check
- [ ] rollback 필요 여부 판단

### 18.3 배포 금지 조건

- W-G1 Manifest 실패
- W-G2 Raw 실패
- W-G3 Provenance 실패
- W-G7 Backup 실패
- Security Gate 실패
- rollback plan 없음
- 사용자 승인 없음
- 비밀 값 노출 가능성 있음
- 유료 표준 원문 장문 노출 가능성 있음

---

## 19. 롤백 절차

1. 롤백 조건을 확인한다.
2. 롤백 전 백업 존재를 확인한다.
3. 롤백 대상 release_id를 식별한다.
4. 승인된 롤백 방법만 실행한다.
5. 롤백 후 health와 warehouse_status를 확인한다.
6. item count, raw count, trace count가 롤백 전 기준과 맞는지 확인한다.
7. restore dry-run을 다시 실행한다.
8. 롤백 결과를 Release Board와 HANDOVER_REPORT에 기록한다.
9. 재발 방지책을 기록한다.

---

## 20. 운영 절차

| 주기 | 점검 항목 |
|---|---|
| 일일 | captured/untriaged/pending_review/hold count, 오류 로그, 백업 상태 |
| 주간 | 오래된 hold, duplicate_suspected, rejected reason, restore dry-run |
| 월간 | 권리 상태, 개인정보/비밀값 scan, 품질 점수 분포, 문서 최신성 |
| 장애 시 | 영향 범위, 임시 우회, 롤백, 원인 분석, 재발 방지 |
| 승격 전 | raw_hash, provenance, rights, review, dry-run, trace 확인 |
| 배포 전 | W-Gate, Backup, Release Board, rollback plan 확인 |

---

## 21. 문서화와 증거팩

### 21.1 증거팩 필수 요소

| 요소 | 설명 |
|---|---|
| 분석 txt | 실행 결과, 판단 근거 |
| 캡처 PNG | Count=1, ReadOnly, UI/게이트 결과 |
| SHA256 | 증거 파일 해시 |
| validator result | W-Gate 결과 |
| backup manifest | 백업 식별자와 포함 범위 |
| release board | 배포/릴리즈 판단 |
| handover report | 다음 작업을 위한 사람용 문서 |

### 21.2 index_SSOT 반영 규칙

증거가 장기기억 proofpack에 들어갈 경우 아래 원칙을 따른다.

```text
PowerShell-only
append-only
idempotent
Count=1
ReadOnly reseal
```

---

## 22. 인수인계 절차

인수인계 보고서에는 다음을 포함한다.

- 오늘 목표
- 실제 완료
- 완료하지 못한 것
- 검증한 것
- 검증하지 못한 것
- 수정 파일
- 생성 파일
- 테스트 결과
- W-Gate 결과
- backup/restore 결과
- 증거 위치
- 남은 위험
- 다음 첫 작업 1개

### 22.1 인수인계 판정 형식

```text
FINAL_DECISION=PASS_OR_HOLD_OR_ROLLBACK
SCOPE=WAREHOUSE
PATCH=YES_OR_NO
PROMOTION=YES_OR_NO_OR_NOT_IN_SCOPE
BACKUP=PASS_OR_HOLD
RESTORE_DRY_RUN=PASS_OR_HOLD
NEXT_ONE=...
```

---

## 23. 금지 목록

- 승인 전 창고 항목을 도서관 정본처럼 표시하지 않는다.
- raw 원본을 직접 수정하지 않는다.
- raw 원문을 공개 리포트에 장문 출력하지 않는다.
- 유료 표준 원문을 저장·노출하지 않는다.
- private_tacit 항목을 공개 export하지 않는다.
- secret 항목을 어떤 공개 출력에도 포함하지 않는다.
- rights_status unknown 항목을 승격하지 않는다.
- review_note 없는 항목을 승인하지 않는다.
- dry-run 없는 항목을 promote하지 않는다.
- trace 없는 승격을 완료로 말하지 않는다.
- backup/restore 검증 없는 배포를 PASS로 말하지 않는다.
- 증거 없는 PASS를 만들지 않는다.

---

## 24. 최종 Definition of Done

창고 개발 완료는 아래를 모두 만족해야 한다.

| 항목 | 완료 조건 |
|---|---|
| Manifest | W-G1 PASS |
| Raw | W-G2 PASS |
| Provenance | W-G3 PASS |
| Review | W-G4 PASS |
| Approval | W-G5 PASS |
| Promotion | W-G6 PASS 또는 NOT_IN_SCOPE 기록 |
| Backup | W-G7 PASS |
| Security | W-G8 PASS |
| Release | W-G9 PASS |
| Item Schema | 필수 필드와 enum 검증 PASS |
| Promotion Trace | 역추적 가능 PASS |
| Quality Threshold | Hard Gate + 80점 기준 적용 |
| LOW_CONFIDENCE | 0.85 미만 사람 리뷰 강제 |
| Handover | 다음 첫 작업 1개 포함 |
| ProofPack | txt, PNG, SHA256, validator 결과 존재 |

### 24.1 PASS 선언 가능 조건

```text
WAREHOUSE_DEVELOPMENT_DONE=PASS
MANIFEST=PASS
RAW_IMMUTABLE=PASS
PROVENANCE=PASS
REVIEW_STATE_MACHINE=PASS
APPROVAL_RECORD=PASS
PROMOTION_DRY_RUN=PASS_OR_NOT_IN_SCOPE
PROMOTION_TRACE=PASS_OR_NOT_IN_SCOPE
SECURITY_SCAN=PASS
BACKUP_RESTORE=PASS
RELEASE_BOARD=UPDATED
HANDOVER=READY
```

### 24.2 PASS 선언 금지 조건

- 검증하지 않은 항목이 있다.
- Hard Gate 실패가 있다.
- raw_hash가 없다.
- provenance가 없다.
- rights_status가 unknown이다.
- secret 또는 private 항목이 공개 export에 포함된다.
- dry-run 없이 승격했다.
- trace 없이 승격 완료라고 했다.
- backup/restore dry-run 없이 배포했다.

---

## 25. 최종 체크리스트

### 개발 시작 전

- [ ] 상위 헌법과 QLIB 공통 가이드북을 확인했다.
- [ ] 작업 모드를 판단했다.
- [ ] 변경 규모를 판단했다.
- [ ] Warehouse Manifest를 확인했다.
- [ ] 비용, 권한, 개인정보, 권리 위험을 확인했다.
- [ ] 테스트 가능 여부를 확인했다.
- [ ] 롤백 가능 여부를 확인했다.

### 구현 후

- [ ] 변경 파일 확인
- [ ] 의도하지 않은 변경 없음
- [ ] raw 원본 직접 수정 없음
- [ ] Secret 노출 없음
- [ ] 권리/개인정보 위험 없음
- [ ] item schema 위반 없음

### 테스트 후

- [ ] validator 실행 항목 기록
- [ ] 미실행 항목 `NOT_EXECUTED`
- [ ] 미검증 항목 `NOT_VERIFIED`
- [ ] W-Gate 결과 기록
- [ ] 증거 저장

### 배포 전

- [ ] 사용자 승인
- [ ] 배포 대상 확정
- [ ] full backup 완료
- [ ] restore dry-run PASS
- [ ] rollback plan 확정
- [ ] release board 준비

### 배포 후

- [ ] health check
- [ ] warehouse_status 확인
- [ ] item state count 확인
- [ ] review queue 확인
- [ ] logs 확인
- [ ] 오류율 확인
- [ ] rollback 필요 여부 판단

### 인수인계 전

- [ ] 완료/미완료 구분
- [ ] 검증/미검증 구분
- [ ] 수정/생성 파일 기록
- [ ] 증거 위치 기록
- [ ] 다음 첫 작업 1개 기록

---

## 26. Self-Check

| 항목 | 상태 |
|---|---|
| 상위 헌법 위반 여부 | PASS |
| QLIB 공통 가이드북 정합성 | PASS |
| 창고 역할 정의 | PASS |
| 전문가 암묵지 저장 기준 | PASS |
| 개인 논문 저장 기준 | PASS |
| 리포트·기고문 저장 기준 | PASS |
| 도서관 승격 기준 | PASS |
| Warehouse Manifest | PASS |
| Item Schema | PASS |
| Promotion Trace | PASS |
| Warehouse Gate | PASS |
| Backup/Restore | PASS |
| Release Board | PASS |
| Quality Threshold | PASS |
| Secret/개인정보/유료 원문 보호 | PASS |
| 독립 제품화 방지 | PASS |
| 수정 불가 고정본 사용 가능성 | PASS |

---

## 27. 출처 문서 목록

문서 내부에는 외부 연결을 넣지 않는다. 출처는 문서명과 기준일 중심으로 기록한다.

- COMMON_DEVELOPMENT_WORKFLOW.md, 2026
- PROJECT_DEVELOPMENT_MEMORY.md, 2026 (프로젝트별 헌법, 실제 존재 시 최신본 확인)
- QLIB_COMPLETE_DEVELOPMENT_GUIDEBOOK_20260511_v1_2.md, 2026
- PROJECT_DEVELOPMENT_GUIDEBOOK_창고.md, 2026
- QLIB_QJ_WAREHOUSE_MASTER_SSOT_20260511.md, 2026
- QLIB_QJ_WAREHOUSE_INTEGRATED_STABLE_OPERATION_20260511.md, 2026
- 2026-05-11_QLIB_QJ_WAREHOUSE_문제점_해결방안_운영배포_분석보고서.md, 2026

---

## 28. 최종 한 줄

```text
창고는 도서관 승격 전 자료를 보관·검역·검수·증거화하는 공식 검역소이며, 전문가 암묵지·개인 논문·리포트·기고문은 raw_hash, provenance, rights, sensitivity, review, approval, dry-run, promotion trace, backup/restore proofpack을 통과한 경우에만 도서관으로 승격된다.
```

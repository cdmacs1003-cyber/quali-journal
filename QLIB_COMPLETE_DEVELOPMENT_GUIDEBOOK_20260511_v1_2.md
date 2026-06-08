# QLIB 통합 개발 가이드북 v1.2

작성일: 2026-05-11  
문서 등급: L0-L3 통합 실행 기준서  
적용 범위: QLIB Warehouse, QLIB Library Core, QLIB Bridge, QLIB Skillup Education, Usage & Analytics, 통합 관리자 UI, ProofPack, Release Board  
문서 상태: 문서·설계 정합성 PASS, 실제 구현 정합성 NOT_VALIDATED  
배포 판정: HOLD UNTIL IMPLEMENTATION PROOFPACK PASS  
운영 원칙: 하나의 제품 거버넌스 아래 모듈형 구현을 유지한다.
개정 범위: v1.1에 L0/L1 문서 동기화, 안전한 레이어 의존, Greenfield/Seeded 로드맵, Bridge query transient 원칙, 멀티테넌트 경계, 계약 버전 관리, 멱등성, AI 비용·Rate Limit, 유료 표준 권한, 운영 관측·사고 대응 기준을 추가한다.

---

## 0. 이 문서의 최종 선언

QLIB는 단일 앱이 아니다. QLIB는 전자생산 관련 지식의 수집, 검역, 정본화, 교육 활용, 사용 로그 분석을 연결하는 지식 운영 제품군이다.

```text
QLIB Warehouse
  승인 전 후보 지식, 전문가 암묵지, 현장 노하우, 리포트, 커뮤니티 기고, 실패 기록을 수집·검역한다.

QLIB Library Core
  승인된 지식만 정본으로 저장하고, 표준 카드, Reference 카드, Evidence, 온톨로지, 그래프, Tailoring Pack, 인덱스를 관리한다.

QLIB Bridge
  Library Core와 Skillup Education 사이의 유일한 지식 호출 계약이다. 직접 DB 접근을 차단하고 Evidence와 Trace만 전달한다.

QLIB Skillup Education
  Library Core의 정본 지식을 교육 효과에 맞게 사용하는 교육 실행 모듈이다.

Usage & Analytics
  사용 로그를 집계·가명화·익명화하여 교육 고도화, 콘텐츠 개선, 마케팅 후보 분석에 사용한다. 정본을 직접 변경하지 않는다.
```

핵심 원칙은 다음과 같다.

```text
제품 비전은 하나다.
개발 구조는 모듈형이다.
운영 판정은 통합형이다.
정본은 Library Core가 가진다.
Warehouse는 정본 승격 전 검역소다.
Skillup과 향후 IPC·ECSS·NASA 교육모듈은 Bridge 계약으로만 Library Evidence를 사용한다.
사용 로그는 정본이 아니라 개선 후보이며, Analytics를 거쳐 Warehouse 후보로만 되돌아간다.
```

이 문서는 기존 5개 문서를 하나의 실행 기준서로 재구성한 완성본이다. 단일 MD 파일이지만 내부 계층은 유지한다.

| 계층 | 역할 | 이 문서의 위치 |
|---|---|---|
| L0 | 제품군 철학, 공통 거버넌스, lifecycle, release board | 0~4장, 12~18장 |
| L1-W | Warehouse 모듈 기준 | 5장 |
| L1-L | Library Core 모듈 기준 | 6장 |
| L1-S | Skillup Education 모듈 기준 | 8장 |
| L2-C | Shared Contracts, Bridge, Evidence, Analytics 계약 | 4장, 7장, 9장 |
| L3 | ProofPack, Release Board, 검증, 운영 인수 기준 | 12~18장 |

---

## 1. 기준 문서와 반영 범위

본 문서는 아래 문서의 구조와 판단을 계승하고, 미흡했던 계약·증거·검증 항목을 보강한다.

| 기준 문서 | 반영 내용 |
|---|---|
| QLIB 통합 마스터 가이드북 20260511 | 하나의 제품군, 3개 모듈 분리, 공통 lifecycle, 통합 UI, release board |
| QLIB_WAREHOUSE_MODULE_GUIDEBOOK_20260511.md | 창고 검역, raw 보존, provenance, review, approval, promotion trace |
| QLIB_LIBRARY_CORE_MODULE_GUIDEBOOK_20260511.md | 정본 지식, Standard/Reference 분리, Evidence Pointer, 온톨로지, 그래프, index |
| QLIB_SKILLUP_EDUCATION_MODULE_GUIDEBOOK_20260511.md | 교육모듈, course binding, Bridge 기반 evidence 답변, 학습 로그, analytics |
| QLIB 정합성 검수 보고서 20260511 | 문서 정합성 100점 판정, 실제 구현 검증 분리, 5개 시나리오 검수 |
| Library Language SSOT Decision | canonical_lang, source_lang, standard_node_id, Reference 저장 규칙, UI 언어 뱃지 |
| F07 Library Admin UI Spec | /lib add, /lib verify, /lib map, /lib show, Browse/Evidence/Tailoring 출력 계층 |
| LTM Integrated Constitution | 계약 우선, 레이어드 모듈성, backup, graph, 실행 루프, 사람 최종 책임 원칙 |

보강한 핵심은 다음이다.

```text
1. Shared Contracts를 독립 계층으로 승격
2. Bridge Contract를 독립 실행 계약으로 분리
3. evidence_id를 1급 ID로 승격
4. standard_node_id / library_id / graph_node_id / evidence_id 역할 분리
5. Language SSOT 정렬
6. 검수 보고서 PASS를 문서·설계 PASS로 제한
7. 구현 검수, 운영 검수, 배포 검수를 별도 단계로 분리
8. ProofPack 산출물명을 Release Board에 직접 연결
9. Usage & Analytics를 독립 거버넌스 후보로 분리
10. 통합 관리자 UI에서 Warehouse, Library, Bridge, Skillup, Analytics 상태를 한 화면에서 연결
11. 단일 L0 기준서와 L1 모듈 문서의 동기화 원칙 고정
12. 안전한 레이어 의존 규칙과 Analytics 이벤트 발행 경계 보강
13. Greenfield와 Seeded Library Track 구현 로드맵 분리
14. Bridge query 원문 transient-only 처리와 로그 저장 금지 고정
15. tenant_id, organization_id, cohort_id 기반 멀티기관 경계 추가
16. contract_version, migration_policy, backward_compatibility, deprecated_after 추가
17. promote, publish, approve 작업의 idempotency_key와 revision 체계 추가
18. AI 모델 라우팅, 비용 상한, Rate Limit, fallback 정책 추가
19. 유료 표준 license entitlement 기록 추가
20. SLO, alert threshold, emergency HOLD, incident review 추가
```

### 1.1 L0 단일 기준서와 L1 모듈 문서 동기화 원칙

본 파일은 단일 MD 완성본이지만, 운영 원칙은 단일 거대 문서가 아니다. 본 파일은 L0 통합 기준서로 사용하고, 구현·검증·배포는 L1 모듈 문서와 계약 스키마로 분리해 관리한다.

```text
L0 통합 기준서
  QLIB_COMPLETE_DEVELOPMENT_GUIDEBOOK_20260511_v1_2.md

L1-W Warehouse Module
  QLIB_WAREHOUSE_MODULE_GUIDEBOOK_20260511.md

L1-L Library Core Module
  QLIB_LIBRARY_CORE_MODULE_GUIDEBOOK_20260511.md

L1-S Skillup Education Module
  QLIB_SKILLUP_EDUCATION_MODULE_GUIDEBOOK_20260511.md

L1-B Bridge Contract Spec
  QLIB_BRIDGE_CONTRACT_SPEC_20260511.md

L1-A Analytics Governance Module
  QLIB_ANALYTICS_GOVERNANCE_MODULE_GUIDEBOOK_20260511.md

L1-O Operations Runbook
  QLIB_OPERATIONS_INCIDENT_RUNBOOK_20260511.md
```

동기화 규칙:

1. L0는 전체 철학, 공통 계약, cross-module lifecycle, release gate를 고정한다.
2. L1은 각 모듈의 구현 세부, API, DB, UI, 테스트, runbook을 고정한다.
3. L0와 L1이 충돌하면 L0가 우선하되, 실제 구현 상세는 L1에 patch로 반영한다.
4. L1 변경이 공통 ID, Bridge, Evidence, Analytics, Release Board에 영향을 주면 L0도 같은 revision으로 개정한다.
5. L0 단일본은 교육·인수인계·최종 검토에 사용하고, 개발자는 L1 모듈 문서와 계약 스키마를 기준으로 구현한다.
6. 단일본에만 존재하고 L1 또는 schemas 디렉터리에 없는 구현 상세는 운영 기준으로 인정하지 않는다.

동기화 ProofPack:

```text
reports/proofpacks/docs/l0_l1_sync_matrix_YYYYMMDD.md
reports/proofpacks/docs/l1_contract_diff_YYYYMMDD.md
reports/proofpacks/docs/schema_doc_alignment_YYYYMMDD.md
```

---

## 2. 문서 상태와 배포 상태 구분

문서 정합성과 실제 구현 정합성은 분리한다.

| 구분 | 현재 판정 | 의미 |
|---|---|---|
| 문서 세트 정합성 | PASS | 가이드북 간 논리 충돌이 없고 lifecycle이 연결된다. |
| 설계 정합성 점수 | 100 / 100 | 문서상 상호 정합성, 기능 연결성, 관리 연결성, 데이터 연결성, UI 연결성, 확장성 기준이다. |
| Contract Test | NOT_VALIDATED | JSON/YAML schema, request/response, error code 검증이 아직 실행되지 않았다. |
| Implementation Test | NOT_VALIDATED | 코드, DB, API, UI 실행 증거가 아직 붙지 않았다. |
| Operational Test | NOT_VALIDATED | backup, restore, rollback, release board, proofpack 증거가 아직 붙지 않았다. |
| 배포 판정 | HOLD | 구현 ProofPack 전까지 운영 배포 금지다. |

금지 표현:

```text
"최종 검수 PASS" 단독 표기 금지
"배포 가능" 단독 표기 금지
"구현 완료" 단독 표기 금지
```

허용 표현:

```text
문서 세트 정합성 검수: PASS
GUIDEBOOK_COHERENCE_SCORE=100
기준: 문서·설계 정합성
실제 구현 정합성: NOT_VALIDATED
배포 가능 판정: HOLD UNTIL IMPLEMENTATION PROOFPACK PASS
```

---

## 3. 목표 아키텍처

### 3.1 전체 구조

```text
External / Internal Sources
  - 전문가 암묵지
  - 현장 노하우
  - 리포트
  - 자료
  - 커뮤니티 기고
  - 실패 기록
  - 표준 적용 해석
  - IPC / ECSS / NASA / 사내 기준
        |
        v
QLIB Warehouse
  - capture
  - raw preservation
  - raw_hash
  - provenance
  - classification
  - review
  - approval
  - promotion dry-run
  - promotion trace
        |
        v
QLIB Library Core
  - Standard Card
  - Reference Card
  - Evidence Pointer
  - Ontology Node
  - Graph Relation
  - Tailoring Pack
  - Library Index
  - Bridge Trace Index
        |
        v
QLIB Bridge
  - resolve_terms
  - retrieve_evidence
  - check_policy
  - explain_trace
  - bind_course_library
        |
        v
QLIB Skillup Education
  - course
  - module
  - learner question
  - evidence answer
  - HOLD
  - review
  - assessment
  - qa logs
        |
        v
Usage & Analytics
  - event aggregation
  - pseudonymization
  - anonymization
  - learning effectiveness
  - content improvement candidates
  - marketing aggregate dataset
        |
        v
Warehouse Candidate Feedback Loop
  - 개선 후보만 Warehouse로 재입고
  - 자동 Library 반영 금지
```

### 3.2 책임 경계

| 모듈 | 책임 | 금지 |
|---|---|---|
| Warehouse | 후보 지식 수집, raw 보존, provenance, classification, review, approval, promotion trace | 승인 전 항목을 Library 정본처럼 표시, raw 임의 수정, Library 카드 직접 덮어쓰기 |
| Library Core | 정본 지식 저장, Standard/Reference 분리, Evidence, Ontology, Graph, Tailoring, Index | 승인 없는 지식 수용, 유료 표준 원문 장문 출력, 교육 로그 자동 정본화 |
| Bridge | 계약 호출, Evidence 조회, 정책 확인, trace 발급, fail-closed | 내부 DB table, local path, secret, raw prompt, route detail 노출 |
| Skillup Education | 교육 UX, 질문, 답변, 복습, 평가, 권한, qa/event logs | Library DB 직접 조회, Warehouse DB 직접 조회, evidence 없는 단정 답변 |
| Usage & Analytics | 집계, 가명화, 익명화, 교육효과 분석, 개선 후보 생성 | 개인 질문 원문 마케팅 사용, 정본 자동 수정, 동의 없는 식별 데이터 활용 |
| Governance | 공통 ID, status, role, approval, release board, proofpack | 모듈별 임의 예외 승인 |

### 3.3 레이어 의존 규칙

```text
CORE        = 정본 저장소, 원문 불변, schema, backup, restore
CONTRACT    = ID, Evidence, Bridge, Promotion Trace, Analytics 계약
INTEGRATION = Validate -> Ingest -> Publish -> Bind -> Use
ONTOLOGY    = node, relation, graph, tailoring
APPLICATION = Warehouse UI, Library UI, Skillup UI, Integrated Console
ANALYTICS   = usage event, mart, aggregate, feedback candidate
```

안전한 기본 의존 방향:

```text
APPLICATION -> INTEGRATION -> CONTRACT -> CORE
ONTOLOGY    -> CONTRACT -> CORE
BRIDGE      -> CONTRACT -> CORE_INDEX_ONLY
```

Analytics는 Application의 필수 선행 의존이 아니다. Application은 Analytics에 조회 의존하지 않고, 이벤트만 발행한다. Analytics는 운영 집계와 개선 후보 생성만 담당한다.

```text
APPLICATION -> ANALYTICS_EVENT_BUS only
ANALYTICS   -> WAREHOUSE_FEEDBACK_CANDIDATE only
ANALYTICS   -X-> LIBRARY_DIRECT_WRITE
ANALYTICS   -X-> SKILLUP_RUNTIME_DECISION
```

금지:

1. Skillup이 Library DB 또는 Warehouse DB를 직접 읽는 구조.
2. Application이 Analytics Mart를 필수 런타임 의존으로 삼는 구조.
3. Analytics가 Library 정본을 직접 수정하는 구조.
4. Warehouse UI가 Library publish를 직접 수행하는 구조.
5. Bridge가 내부 table, local path, raw prompt를 외부에 노출하는 구조.

---

## 4. Shared Contracts

Shared Contracts는 모든 모듈이 공유하는 실행 계약이다. 이 장은 별도 파일로 분리해도 되는 수준의 L2 계약이며, 본 단일본에서는 4장 전체를 독립 계약으로 취급한다.

### 4.1 ID Contract

모든 객체는 아래 ID를 사용한다.

| ID | 형식 | 생성 위치 | 의미 | 변경 가능 여부 |
|---|---|---|---|---|
| source_id | SRC-YYYYMMDD-HHMMSS-XXXX | Warehouse | 원천 자료 단위 | 불가 |
| warehouse_item_id | WHI-YYYYMMDD-HHMMSS-XXXX | Warehouse | 창고 검역 단위 | 불가 |
| raw_hash | sha256:<hex> | Warehouse | raw 또는 pointer 무결성 | raw revision 시 신규 |
| promotion_trace_id | PTR-YYYYMMDD-HHMMSS-XXXX | Warehouse / Library | 창고에서 도서관으로 승격된 이력 | 불가 |
| standard_node_id | ORG:DOC_CODE@EDITION_OR_REV | Library | 표준 또는 Reference 앵커의 언어 독립 논리 ID | revision별 신규 |
| library_id | LIB-<domain>-<slug>-<version> | Library | 도서관 카드/정본 레코드 ID | 판본별 신규 |
| graph_node_id | GND-<domain>-<slug>-<version> | Library | 내부 온톨로지/그래프 노드 ID | 판본별 신규 |
| evidence_id | EVD-YYYYMMDD-HHMMSS-XXXX | Library | 근거 포인터 ID | 불가 |
| tailoring_pack_id | TPK-<project>-<version> | Library | Tailoring 묶음 ID | 판본별 신규 |
| standard_pack_id | SPK-<org>-<domain>-<version> | Library | IPC, ECSS, NASA 등 표준팩 | 판본별 신규 |
| module_id | MOD-<family>-<slug>-<version> | Skillup | 교육모듈 ID | 판본별 신규 |
| course_id | CRS-<org>-<slug>-<cohort> | Skillup | 교육 과정 운영 단위 | 과정별 신규 |
| binding_id | BND-<course_id>-<module_id>-<version> | Skillup | course, module, library scope 연결 | 판본별 신규 |
| request_id | REQ-YYYYMMDD-HHMMSS-XXXX | Skillup | 질문 또는 사용자 행위 요청 | 불가 |
| trace_id | TRC-YYYYMMDD-HHMMSS-XXXX | Bridge | 답변, evidence, HOLD 추적 | 불가 |
| event_id | EVT-YYYYMMDD-HHMMSS-XXXX | Skillup / Analytics | 로그 이벤트 | 불가 |
| user_id_hash | USH-<hash> | Identity / Analytics | 사용자 가명 식별자 | rotation 정책 적용 |
| consent_id | CST-YYYYMMDD-HHMMSS-XXXX | Analytics | 동의 기록 ID | 불가 |
| tenant_id | TEN-<slug> | Governance | 고객사·기관·서비스 단위 격리 ID | 불가 |
| organization_id | ORGUNIT-<slug> | Governance | 조직 또는 부서 단위 ID | 조직 변경 시 신규 가능 |
| cohort_id | COH-YYYYMM-<slug> | Skillup | 교육 기수·반·고객 그룹 ID | 과정별 신규 |
| license_entitlement_id | LIC-YYYYMMDD-XXXX | Library / Governance | 유료 표준 접근 권한 ID | 갱신 시 신규 |
| approval_event_id | AEV-YYYYMMDD-HHMMSS-XXXX | Warehouse / Library / Release | 승인 이벤트 ID | 불가 |
| idempotency_key | IDEMP-<hash> | API caller | 중복 실행 방지 키 | 요청 단위 불가 |
| revision | integer | 각 모듈 | 낙관적 잠금과 변경 추적용 revision | 변경 시 증가 |

### 4.2 ID 역할 분리

아래 ID는 서로 대체할 수 없다.

```text
standard_node_id
  = ORG:DOC_CODE@EDITION_OR_REV
  = 표준 또는 Reference 앵커의 외부/논리 식별자
  = 언어와 무관하다.

library_id
  = LIB-<domain>-<slug>-<version>
  = 도서관 카드/정본 레코드 식별자
  = UI 카드, Library Index, Bridge 응답에 쓰인다.

graph_node_id
  = GND-<domain>-<slug>-<version>
  = 내부 그래프 노드 식별자
  = graph relation과 ontology traversal에 쓰인다.

evidence_id
  = EVD-YYYYMMDD-HHMMSS-XXXX
  = 근거 포인터 식별자
  = 답변 trace와 분리된다.

trace_id
  = TRC-YYYYMMDD-HHMMSS-XXXX
  = 특정 요청·답변·HOLD 실행의 추적 식별자
  = evidence_id와 혼용 금지다.
```

### 4.3 ID 연결 규칙

```text
source_id -> warehouse_item_id -> promotion_trace_id -> library_id -> graph_node_id -> evidence_id -> trace_id
```

필수 규칙:

1. `source_id`는 전체 lifecycle의 출발점이다.
2. `promotion_trace_id`는 `warehouse_item_id`, `library_id`, `graph_node_id`, `evidence_id`를 연결해야 한다.
3. `library_id`는 하나 이상의 `graph_node_id`와 연결된다.
4. `evidence_id`는 반드시 `library_id` 또는 `graph_node_id` 중 하나 이상을 참조한다.
5. `trace_id`는 `request_id`, `course_id`, `module_id`, `binding_id`, `evidence_id` 또는 HOLD 사유를 연결한다.
6. `user_id_hash`는 원본 사용자 ID를 직접 노출하지 않는다.
7. 사용자 출력에는 raw path, internal path, secret, raw prompt, raw standard text를 포함하지 않는다.
8. `tenant_id`, `organization_id`, `cohort_id`는 사용자 로그, course binding, analytics event, entitlement record에 포함한다.
9. approval, publish, promote, rollback은 `idempotency_key`와 `approval_event_id` 없이는 실행하지 않는다.
10. `revision`은 객체별 단조 증가 값으로 관리하고, 이전 상태는 `prev_status`로 보존한다.

### 4.4 Language SSOT Contract

도서관 언어 정책은 다음으로 고정한다.

```yaml
language_ssot:
  canonical_lang: EN
  source_lang: BCP47
  node_id_pattern: ORG:DOC_CODE@EDITION_OR_REV
  ui_default_language: EN
  ui_on_demand_language: KO
```

필수 규칙:

1. `canonical_lang`은 항상 `EN`으로 저장한다.
2. `source_lang`은 Non-SSOT 보조 메타이며 BCP 47 태그로만 저장한다.
3. 예시는 `en`, `ko`, `zh-Hans`, `zh-Hant`, `ja`, `und`이다.
4. EN 원문 Standard가 있으면 타언어 번역본과 해설본은 Reference로 저장한다.
5. EN 원문 Standard가 없으면 존재하는 타언어 원문을 Standard로 저장할 수 있다.
6. `source_lang="und"`는 UI에서 `미상(und)`로 표시한다.
7. 카드 상단에는 `source_lang` 뱃지를 항상 표시한다.
8. 뱃지는 2글자 대문자 축약을 기본으로 하되, script/region은 상세 또는 tooltip에서 원문 태그를 보존한다.

뱃지 매핑 예시:

| source_lang | Badge | 상세 표시 |
|---|---|---|
| en | EN | en |
| ko | KO | ko |
| zh-Hans | ZH | 중국어 간체 · zh-Hans |
| zh-Hant | ZH | 중국어 번체 · zh-Hant |
| ja | JA | ja |
| und | 미상(und) | und |

### 4.5 Standard와 Reference 저장 규칙

| 구분 | 저장 기준 | ID 기준 | 정책 |
|---|---|---|---|
| Standard | 공식 표준 원문 또는 공식 변형판 | standard_node_id | 가능하면 EN 원문을 앵커로 사용 |
| Reference | 번역본, 해설서, 교육자료, 기술서적, 리포트, 현장 메모 | standard_node_id 또는 reference_node_id | Standard와 연결되어야 함 |
| Training Reference | 교육모듈 보조 자료 | library_id + graph_node_id | 학생/강사 visibility 구분 |
| Practice Evidence | 현장 사례, 실패 기록 | evidence_id 중심 | raw 공개 제한 가능 |
| Tailoring Decision | 적용/제외/강화/완화 결정 | tailoring_pack_id | 별도 승인 필요 |

Reference `doc_code` 규칙:

```text
ISBN이 있으면 ISBN 사용
ISBN이 없으면 짧은 slug 사용
slug는 ASCII 소문자, 숫자, hyphen 기반으로 자동 생성하고 저장 전 사람이 확정할 수 있다.
```

### 4.6 Status Contract

#### Warehouse 상태

```text
captured
  -> classified
  -> source_verified
  -> review_ready
  -> reviewed
  -> approved_for_library
  -> promotion_dry_run_pass
  -> promoted
```

예외:

```text
hold_source_missing
hold_sensitive
hold_copyright
hold_review_needed
rejected
archived
```

#### Library 상태

```text
draft_card
  -> evidence_linked
  -> ontology_linked
  -> graph_validated
  -> tailoring_ready
  -> review_approved
  -> published
```

예외:

```text
hold_no_evidence
hold_policy_violation
hold_graph_mismatch
hold_review_needed
deprecated
superseded
```

#### Skillup 상태

```text
course_draft
  -> binding_ready
  -> content_ready
  -> bridge_ready
  -> cohort_ready
  -> active
  -> completed
  -> archived
```

예외:

```text
hold_no_binding
hold_no_evidence
hold_permission
hold_bridge_error
hold_policy
```

### 4.7 Tenant Boundary Contract

향후 회사별 교육, 고객별 표준팩, 기관별 마케팅·통계 운영을 지원하기 위해 모든 운영 객체는 tenant 경계를 가져야 한다. `course_id`에 고객사를 녹이는 방식은 보조 식별에는 유용하지만 데이터 격리 기준으로는 부족하다.

```yaml
tenant_context:
  tenant_id: TEN-customer-or-service
  organization_id: ORGUNIT-department-or-company
  cohort_id: COH-202605-skillup-01 | null
  data_region: KR | US | EU | INTERNAL
  isolation_level: shared_schema | dedicated_schema | dedicated_instance
  visibility_scope: tenant_private | org_shared | global_reference
```

필수 적용 객체:

| 객체 | tenant_id | organization_id | cohort_id |
|---|---:|---:|---:|
| warehouse_item | 필수 | 필수 | 선택 |
| promotion_trace | 필수 | 필수 | 선택 |
| library_card | 필수 | 필수 | 선택 |
| evidence_pointer | 필수 | 필수 | 선택 |
| standard_pack | 필수 | 필수 | 선택 |
| module_manifest | 필수 | 필수 | 선택 |
| course_library_binding | 필수 | 필수 | 필수 |
| qa_log | 필수 | 필수 | 필수 |
| analytics_mart_event | 필수 | 필수 | 선택 |
| license_entitlement | 필수 | 필수 | 선택 |

격리 규칙:

```text
TENANT_DEFAULT_DENY=True
CROSS_TENANT_QUERY_ALLOWED=False
CROSS_TENANT_ANALYTICS_ALLOWED=False unless anonymized_aggregate=True and governance_approved=True
TENANT_ID_REQUIRED_ON_WRITE=True
TENANT_ID_REQUIRED_ON_READ=True
ORGANIZATION_SCOPE_ENFORCED=True
COHORT_SCOPE_ENFORCED_FOR_SKILLUP=True
```

### 4.8 Contract Versioning and Migration Contract

`schema_version`은 데이터 구조 버전이고, `contract_version`은 모듈 간 실행 계약 버전이다. 둘은 분리한다.

```yaml
contract_metadata:
  schema_version: 1
  contract_version: 1.0.0
  min_supported_contract_version: 1.0.0
  backward_compatibility: true
  migration_policy: none | additive | transform_required | breaking
  deprecated_after: null | YYYY-MM-DD
  replacement_contract: null | ContractName@version
```

버전 정책:

| 변경 유형 | 예시 | 정책 |
|---|---|---|
| additive | optional field 추가 | minor 증가, 기존 계약 유지 |
| transform_required | 필드명 변경, enum 확장 | migration plan과 dry-run 필수 |
| breaking | 필수 필드 삭제, 의미 변경 | major 증가, deprecated_after 명시 |
| policy_only | 보존기간, rate limit 변경 | policy revision 기록 |

마이그레이션 PASS 기준:

```text
MIGRATION_PLAN_EXISTS=True
BACKWARD_COMPATIBILITY_DECLARED=True
DEPRECATED_AFTER_DECLARED_IF_NEEDED=True
ROLLBACK_PLAN_EXISTS=True
MIGRATION_DRY_RUN_PASS=True
AFFECTED_TENANT_COUNT_RECORDED=True
PROOFPACK_EXISTS=True
```

### 4.9 Idempotency, Revision, and Approval Event Contract

`approve`, `promote`, `publish`, `bind`, `release`는 중복 실행 위험이 있으므로 멱등 계약을 필수로 적용한다.

```yaml
state_transition_request:
  schema_version: 1
  contract_version: 1.0.0
  idempotency_key: IDEMP-promote-sha256
  actor_id: string
  tenant_id: TEN-customer-a
  object_id: WHI-YYYYMMDD-HHMMSS-XXXX
  object_type: warehouse_item | library_card | course_binding | release
  action: approve | promote | publish | bind | release | rollback
  prev_status: reviewed
  next_status: approved_for_library
  revision: 3
  approval_event_id: AEV-YYYYMMDD-HHMMSS-XXXX
  requested_at: datetime
```

멱등 처리 규칙:

```text
SAME_IDEMPOTENCY_KEY_SAME_PAYLOAD=RETURN_PREVIOUS_RESULT
SAME_IDEMPOTENCY_KEY_DIFFERENT_PAYLOAD=HOLD_IDEMPOTENCY_CONFLICT
PREV_STATUS_MISMATCH=HOLD_STATE_CONFLICT
REVISION_MISMATCH=HOLD_REVISION_CONFLICT
APPROVAL_EVENT_ID_REQUIRED_FOR_APPROVE_PROMOTE_PUBLISH=True
ACTOR_ID_REQUIRED=True
```

### 4.10 License Entitlement Contract

유료 표준은 pointer-only 정책만으로 충분하지 않다. 실제 접근 권한을 별도 entitlement로 기록한다.

```yaml
license_entitlement:
  schema_version: 1
  contract_version: 1.0.0
  license_entitlement_id: LIC-YYYYMMDD-XXXX
  tenant_id: TEN-customer-or-service
  organization_id: ORGUNIT-company
  licensed_org: string
  standard_node_ids:
    - IPC:J-STD-001@RevH
  allowed_roles:
    - instructor
    - reviewer
    - admin
  disallowed_roles:
    - student
  allowed_actions:
    - view_pointer
    - view_summary
    - view_short_reference_label
  disallowed_actions:
    - export_raw_text
    - copy_full_clause
    - public_distribution
  effective_from: YYYY-MM-DD
  expiry_date: YYYY-MM-DD
  pointer_only_required: true
  raw_export_allowed: false
  evidence_required: true
  audit_required: true
```

License Gate:

```text
LICENSE_ENTITLEMENT_EXISTS=True
LICENSE_NOT_EXPIRED=True
ROLE_ALLOWED=True
TENANT_MATCH=True
RAW_EXPORT_ALLOWED=False
POINTER_ONLY_REQUIRED=True
```

### 4.11 Analytics Event Contract

Application은 Analytics를 직접 의존하지 않는다. Application은 이벤트를 발행하고, Analytics는 비동기적으로 수집한다.

```yaml
analytics_event:
  schema_version: 1
  contract_version: 1.0.0
  event_id: EVT-YYYYMMDD-HHMMSS-XXXX
  tenant_id: TEN-customer-a
  organization_id: ORGUNIT-customer-a-quality
  cohort_id: COH-CRS-QUALI-SKILLUP-202605-001
  user_id_hash: USH-hash
  event_type: question_asked | answer_rendered | hold_created | evidence_viewed | assessment_viewed
  request_id: REQ-YYYYMMDD-HHMMSS-XXXX
  trace_id: TRC-YYYYMMDD-HHMMSS-XXXX | null
  query_hash: sha256:string | null
  query_summary: string | null
  raw_query_stored: false
  risk_flags: []
  occurred_at: datetime
```

Analytics 이벤트는 실패해도 Application 정상 흐름을 막지 않는다. 단, 이벤트 큐가 일정 임계치를 넘거나 보안 이벤트가 누락되면 운영 게이트는 HOLD로 전환한다.

### 4.12 HOLD Reason Contract

| 코드 | 사유 | 기본 조치 |
|---|---|---|
| HOLD_SOURCE_MISSING | provenance 또는 source_ref 없음 | Warehouse 보강 요청 |
| HOLD_RAW_HASH_MISSING | raw_hash 없음 | raw 검증 재수행 |
| HOLD_COPYRIGHT | 원문 출력 정책 위반 | raw export 차단 |
| HOLD_NO_EVIDENCE | evidence_id 없음 | 답변 또는 승격 중지 |
| HOLD_POLICY_VIOLATION | role/source/export 정책 위반 | Bridge fail-closed |
| HOLD_GRAPH_MISMATCH | graph relation 불일치 | Library 검토 요청 |
| HOLD_NO_BINDING | course_library_binding 없음 | 과정 active 금지 |
| HOLD_PERMISSION | 사용자 권한 부족 | 기능 차단 |
| HOLD_BRIDGE_ERROR | Bridge 계약 응답 실패 | 안전 답변 또는 HOLD |
| HOLD_ANALYTICS_CONSENT | 동의 또는 보존 정책 불충족 | 분석/마케팅 제외 |
| HOLD_PROOFPACK_MISSING | PASS 증거 파일 없음 | release board HOLD |
| HOLD_TENANT_BOUNDARY | tenant 또는 organization 경계 위반 | 요청 차단 |
| HOLD_CONTRACT_DEPRECATED | 계약 버전 만료 또는 지원 종료 | migration 또는 계약 갱신 |
| HOLD_MIGRATION_REQUIRED | schema/contract 변환 필요 | migration dry-run 요구 |
| HOLD_IDEMPOTENCY_CONFLICT | 같은 idempotency_key에 다른 요청 | 작업 중지 및 감사 |
| HOLD_REVISION_CONFLICT | revision 불일치 | 최신 상태 재조회 |
| HOLD_LICENSE_EXPIRED | 유료 표준 권한 만료 | 접근 차단 |
| HOLD_RATE_LIMIT | rate limit 초과 | 제한 또는 대기 |
| HOLD_COST_LIMIT | 월/일/테넌트 비용 상한 초과 | 저비용 fallback 또는 HOLD |
| HOLD_SLO_BREACH | SLO 임계치 초과 | 운영 알림 및 degraded mode |
| HOLD_INCIDENT_EMERGENCY | 장애·보안 사고로 emergency HOLD 발동 | 핵심 기능 중지 |

### 4.13 Error Code Contract

| 영역 | 코드 범위 | 예시 |
|---|---|---|
| Warehouse | WH-* | WH-001 MANIFEST_MISSING, WH-010 PROVENANCE_REQUIRED |
| Library | LIB-* | LIB-001 STANDARD_NODE_ID_INVALID, LIB-020 EVIDENCE_POINTER_REQUIRED |
| Bridge | BRG-* | BRG-001 INDEX_MISSING, BRG-010 POLICY_DENIED |
| Skillup | SKL-* | SKL-001 BINDING_REQUIRED, SKL-020 ANSWER_HOLD |
| Analytics | ANA-* | ANA-001 CONSENT_REQUIRED, ANA-020 RAW_PROMPT_BLOCKED |
| Release | REL-* | REL-001 PROOFPACK_MISSING, REL-010 RESTORE_DRY_RUN_REQUIRED |
| Tenant | TEN-* | TEN-001 TENANT_REQUIRED, TEN-010 CROSS_TENANT_DENIED |
| License | LIC-* | LIC-001 ENTITLEMENT_REQUIRED, LIC-010 LICENSE_EXPIRED |
| Migration | MIG-* | MIG-001 MIGRATION_REQUIRED, MIG-010 ROLLBACK_PROOF_MISSING |
| AI Runtime | AIR-* | AIR-001 RATE_LIMIT, AIR-010 COST_LIMIT, AIR-020 FALLBACK_USED |
| Operations | OPS-* | OPS-001 SLO_BREACH, OPS-010 INCIDENT_DECLARED |

---

## 5. QLIB Warehouse Module

### 5.1 모듈 선언

Warehouse는 도서관 승격 전 후보 지식의 공식 검역소다. 목표는 많은 자료를 쌓는 것이 아니라, 도서관 정본으로 승격해도 되는 지식만 안전하게 선별하는 것이다.

저장 대상:

```text
전문가 암묵지
현장 노하우
리포트
자료
커뮤니티 기고
실패 기록
리뷰 메모
승인 전 표준 적용 해석
질문 seed
교육 개선 후보
```

### 5.2 책임

| 책임 | 설명 |
|---|---|
| Capture | 외부 또는 내부 지식 후보를 수집한다. |
| Raw Preservation | 원본 또는 원본 pointer를 변조하지 않고 보존한다. |
| Provenance | 제공자, 권리, 수집 목적, 수집 시점, 출처를 기록한다. |
| Classification | item type, sensitivity, visibility, domain을 분류한다. |
| Review | 전문가 검토, confidence, 보강 필요 여부를 기록한다. |
| Approval | 도서관 승격 가능 여부를 판정한다. |
| Promotion Dry-run | Library write 없이 승격 가능성을 사전 검증한다. |
| Promotion Trace | 어떤 창고 항목이 어떤 도서관 카드, graph node, evidence로 승격됐는지 기록한다. |
| ProofPack | 검증 증거와 백업 증거를 묶는다. |

### 5.3 금지

1. 승인 전 항목을 도서관 정본처럼 표시하지 않는다.
2. raw를 직접 수정하지 않는다. 수정은 revision으로 남긴다.
3. provenance 없는 항목은 도서관 승격 대상이 아니다.
4. 민감정보와 비공개 노하우를 공개 리포트에 노출하지 않는다.
5. Warehouse UI에서 Library publish를 직접 수행하지 않는다.
6. Analytics 개선 후보를 자동으로 Library 정본에 반영하지 않는다.

### 5.4 Warehouse Manifest

```yaml
warehouse_manifest:
  schema_version: 1
  module_id: QLIB-WAREHOUSE
  role: library_pre_approval_warehouse
  owner_project: qlib
  date: 2026-05-11
  roots:
    warehouse_root: data/warehouse
    raw_root: data/warehouse/raw
    derived_root: data/warehouse/derived
    trace_root: data/warehouse/trace
    backup_root: backup/warehouse
    proofpack_root: reports/proofpacks/warehouse
  official_indexes:
    warehouse_items: data/warehouse/warehouse_items.jsonl
    promotion_trace: data/warehouse/trace/promotion_trace.jsonl
  rules:
    raw_is_immutable: true
    every_item_requires_source_id: true
    every_item_requires_provenance: true
    every_item_requires_raw_hash: true
    approved_items_require_review: true
    promotion_requires_dry_run: true
    library_write_from_warehouse_ui: false
```

### 5.5 Warehouse Item Schema

```yaml
warehouse_item:
  schema_version: 1
  contract_version: 1.0.0
  tenant_context:
    tenant_id: TEN-customer-or-service
    organization_id: ORGUNIT-company
    cohort_id: COH-YYYYMM-slug | null
  revision: 1
  idempotency_key: IDEMP-sha256 | null
  source_id: SRC-YYYYMMDD-HHMMSS-XXXX
  warehouse_item_id: WHI-YYYYMMDD-HHMMSS-XXXX
  item_type: tacit_knowledge | report | document | community_contribution | failure_record | review_memo | standard_note | analytics_improvement_candidate
  title: string
  summary: string
  raw_pointer: string
  raw_hash: sha256:string
  provenance:
    provider_type: expert | internal_team | community | customer | public_source | analytics
    provider_ref: string
    received_at: datetime
    collection_reason: string
    rights_status: owned | licensed | public | restricted | unknown
  classification:
    domain: soldering | esd | harness | pcb_rework | quality | defense | space | general
    sensitivity: public | internal | confidential | secret
    visibility: library_candidate | internal_only | no_export
    tags: []
  review:
    status: not_started | in_review | reviewed | hold | rejected
    reviewer_id: string
    confidence: low | medium | high
    review_note: string
    reviewed_at: datetime
  approval:
    approved_for_library: true | false
    approval_event_id: AEV-YYYYMMDD-HHMMSS-XXXX | null
    approver_id: string
    approval_reason: string
    prev_status: reviewed | hold_review_needed | null
    approved_at: datetime
  status: captured | classified | source_verified | review_ready | reviewed | approved_for_library | promotion_dry_run_pass | promoted | hold_source_missing | hold_sensitive | hold_copyright | hold_review_needed | rejected | archived
```

### 5.6 Promotion Trace Schema

```yaml
promotion_trace:
  schema_version: 1
  contract_version: 1.0.0
  idempotency_key: IDEMP-sha256
  approval_event_id: AEV-YYYYMMDD-HHMMSS-XXXX
  tenant_context:
    tenant_id: TEN-customer-or-service
    organization_id: ORGUNIT-company
    cohort_id: COH-YYYYMM-slug | null
  revision: 1
  promotion_trace_id: PTR-YYYYMMDD-HHMMSS-XXXX
  source_id: SRC-YYYYMMDD-HHMMSS-XXXX
  warehouse_item_id: WHI-YYYYMMDD-HHMMSS-XXXX
  raw_hash: sha256:string
  source_status_before: approved_for_library
  source_status_after: promoted
  prev_status: approved_for_library
  next_status: promoted
  promoted_standard_node_id: ORG:DOC_CODE@EDITION_OR_REV | null
  promoted_library_id: LIB-domain-slug-v1
  promoted_graph_node_id: GND-domain-slug-v1
  promoted_evidence_ids:
    - EVD-YYYYMMDD-HHMMSS-XXXX
  promoted_tailoring_pack_id: TPK-project-v1 | null
  validation_result: PASS | HOLD | FAIL
  hold_reason: string | null
  promoted_at: datetime
  promoted_by: string
  output_artifacts:
    standard_card: string | null
    reference_card: string | null
    evidence_pointer: string
    graph_relation: string
    library_index_delta: string
  policy_result:
    paid_standard_raw_export_count: 0
    secret_leak_count: 0
    internal_path_leak_count: 0
    pii_leak_count: 0
```

### 5.7 승격 금지 조건

아래 중 하나라도 있으면 승격은 HOLD 또는 FAIL이다.

```text
status != approved_for_library
raw_hash 없음
provenance 없음
rights_status = unknown
sensitivity = secret
visibility = no_export 인데 공개 산출물 생성 시도
review_note 없음
approver_id 없음
promotion dry-run 미수행
유료 표준 원문 장문 출력 시도
internal path 또는 secret 출력 시도
```

### 5.8 Warehouse API 계약

| Operation | 목적 | 성공 기준 |
|---|---|---|
| Warehouse.create_item | 창고 항목 입고 | source_id, warehouse_item_id, raw_hash 생성 |
| Warehouse.list_items | 상태/타입별 목록 | 민감 항목 마스킹 |
| Warehouse.read_item | 단일 항목 조회 | raw, derived, provenance 분리 |
| Warehouse.update_status | 상태 전이 | 허용 전이만 PASS |
| Warehouse.add_review | 검토 기록 | reviewer, confidence, note 저장 |
| Warehouse.approve_for_library | 도서관 승격 승인 | approval_record 저장 |
| Warehouse.promotion_dry_run | 승격 미리보기 | Library write 0건 |
| Warehouse.promote | 승격 실행 | promotion_trace, Library artifacts 생성 |
| Warehouse.explain_trace | 추적 조회 | source_id부터 evidence_id까지 표시 |

### 5.9 Warehouse UI 기준

필수 탭:

```text
Captured
Classified
Needs Source
Review
Hold
Approved
Promotion Dry-run
Promoted
Trace
Backup
```

UI 금지:

```text
승인 전 항목을 "도서관 정본"으로 표시 금지
private/tacit 항목 공개 리포트 노출 금지
sensitivity=secret 항목 외부 export 금지
Warehouse UI에서 publish 버튼 제공 금지
```

### 5.10 Warehouse Gate

| Gate | PASS 기준 | ProofPack |
|---|---|---|
| W-G1 Manifest | warehouse_manifest 존재, schema valid | reports/proofpacks/warehouse/warehouse_manifest_validation_YYYYMMDD.md |
| W-G2 Raw | raw immutable, raw_hash 검증 | reports/proofpacks/warehouse/raw_hash_validation_YYYYMMDD.md |
| W-G3 Provenance | every item provenance exists | reports/proofpacks/warehouse/provenance_validation_YYYYMMDD.md |
| W-G4 Review | review state machine PASS | reports/proofpacks/warehouse/review_state_machine_YYYYMMDD.md |
| W-G5 Approval | approval record exists | reports/proofpacks/warehouse/approval_record_YYYYMMDD.md |
| W-G6 Promotion | dry-run then promotion trace | reports/proofpacks/warehouse/promotion_trace_PTR-*.md |
| W-G7 Backup | backup and restore dry-run | reports/proofpacks/warehouse/backup_restore_YYYYMMDD.md |

### 5.11 Warehouse Definition of Done

```text
WAREHOUSE_MANIFEST_PASS=True
RAW_IMMUTABLE_PASS=True
PROVENANCE_PASS=True
REVIEW_STATE_MACHINE_PASS=True
APPROVAL_RECORD_PASS=True
PROMOTION_DRY_RUN_PASS=True
PROMOTION_TRACE_PASS=True
PROMOTED_EVIDENCE_ID_EXISTS=True
SECURITY_SCAN_PASS=True
BACKUP_RESTORE_PASS=True
INTEGRATED_RELEASE_BOARD_UPDATED=True
```

---

## 6. QLIB Library Core Module

### 6.1 모듈 선언

Library Core는 QLIB의 정본 지식 심장부다. Warehouse에서 검역·승인된 지식만 받아들이고, Skillup과 향후 표준 교육모듈에는 안전한 Evidence와 Trace만 제공한다.

### 6.2 책임

| 책임 | 설명 |
|---|---|
| Canonical Knowledge | 승인된 지식만 정본으로 저장 |
| Standard Card | IPC, ECSS, NASA, 사내 표준 등 공식 표준 앵커 |
| Reference Card | 해설서, 번역본, 교육자료, 기술서적, 리포트 등 보조 지식 |
| Evidence Pointer | 원문 전체가 아니라 근거 위치와 사용 정책 연결 |
| Ontology | 용어, 공정, 결함, 조치, 위험, 표준, 교육 단원 의미 노드 |
| Graph Relation | 표준, 절차, 위험, 교육모듈 간 관계 |
| Tailoring Pack | 프로젝트별 적용/제외/강화/완화 결정 |
| Library Index | UI와 Bridge가 읽는 안전한 index |
| Bridge Trace Index | Skillup 답변 추적을 위한 trace index |

### 6.3 Library Manifest

```yaml
library_manifest:
  schema_version: 1
  module_id: QLIB-LIBRARY-CORE
  role: canonical_semantic_knowledge_store
  owner_project: qlib
  date: 2026-05-11
  roots:
    library_root: data/library
    raw_pointer_root: data/library/raw_pointers
    standard_card_root: data/library/cards/standard
    reference_card_root: data/library/cards/reference
    ontology_root: data/library/ontology
    graph_root: data/library/graph
    tailoring_root: data/library/tailoring
    export_index_root: data/library/exports/indexes
    proofpack_root: reports/proofpacks/library
  indexes:
    library_index: data/library/exports/indexes/library_index.json
    bridge_trace_index: data/library/exports/indexes/bridge_trace_index.json
    graph_index: data/library/exports/indexes/graph_index.json
  rules:
    approved_source_only: true
    pointer_only_for_paid_standards: true
    bridge_reads_index_only: true
    ui_reads_index_only: true
    tailoring_requires_separate_approval: true
    canonical_lang_fixed: EN
    source_lang_bcp47_only: true
```

### 6.4 Standard Card Schema

```yaml
standard_card:
  schema_version: 1
  contract_version: 1.0.0
  tenant_context:
    tenant_id: global | TEN-customer-or-service
    organization_id: global | ORGUNIT-company
    visibility_scope: global_reference | tenant_private
  revision: 1
  standard_node_id: IPC:J-STD-001@RevH
  library_id: LIB-ipc-jstd-001-revh-v1
  graph_node_id: GND-ipc-jstd-001-revh-v1
  source_trace:
    promotion_trace_id: PTR-YYYYMMDD-HHMMSS-XXXX
    warehouse_item_id: WHI-YYYYMMDD-HHMMSS-XXXX
    source_id: SRC-YYYYMMDD-HHMMSS-XXXX
    raw_hash: sha256:string
  standard:
    org: IPC
    doc_code: J-STD-001
    edition_or_rev: RevH
    year: string
    canonical_lang: EN
    source_lang: en
    doc_kind: STANDARD
    title_en: string
    title_ko: string
  evidence_policy:
    paid_standard_raw_export_allowed: false
    pointer_only_required: true
    license_entitlement_required: true
    license_entitlement_ids:
      - LIC-YYYYMMDD-XXXX
    student_raw_text_allowed: false
    instructor_raw_text_allowed: false
  status: draft_card | evidence_linked | ontology_linked | graph_validated | tailoring_ready | review_approved | published | deprecated | superseded
```

### 6.5 Reference Card Schema

```yaml
reference_card:
  schema_version: 1
  reference_node_id: REF:slug-or-isbn@edition
  linked_standard_node_id: IPC:J-STD-001@RevH | null
  library_id: LIB-ref-pcb-rework-field-note-v1
  graph_node_id: GND-ref-pcb-rework-field-note-v1
  source_trace:
    promotion_trace_id: PTR-YYYYMMDD-HHMMSS-XXXX
    warehouse_item_id: WHI-YYYYMMDD-HHMMSS-XXXX
    source_id: SRC-YYYYMMDD-HHMMSS-XXXX
    raw_hash: sha256:string
  reference:
    ref_type: expert_note | report | training_material | community_contribution | failure_record | translation | commentary | technical_book
    title: string
    summary: string
    canonical_lang: EN
    source_lang: ko | en | ja | zh-Hans | und
    source_policy: internal | public | restricted
  status: draft_card | evidence_linked | ontology_linked | graph_validated | review_approved | published | deprecated
```

### 6.6 Evidence Pointer Schema

Evidence는 원문 복사가 아니라 검증 가능한 근거 위치와 사용 정책을 연결하는 포인터다.

```yaml
evidence_pointer:
  schema_version: 1
  contract_version: 1.0.0
  evidence_id: EVD-YYYYMMDD-HHMMSS-XXXX
  library_id: LIB-domain-slug-v1
  standard_node_id: ORG:DOC_CODE@EDITION_OR_REV | null
  graph_node_id: GND-domain-slug-v1
  pointer_type: standard_clause | document_section | training_page | expert_note | failure_record | tailoring_decision
  pointer_label: string
  pointer_scope:
    doc_name: string
    doc_revision: string | null
    year: string | null
    section_label: string | null
    page_range: string | null
  source_policy:
    raw_text_export_allowed: false
    summary_allowed: true
    license_entitlement_id: LIC-YYYYMMDD-XXXX | null
    allowed_roles:
      - student
      - instructor
      - reviewer
      - admin
    licensed_org: string | null
    expiry_date: date | null
    student_visible: true
    instructor_visible: true
    reviewer_visible: true
    admin_visible: true
  trace:
    promotion_trace_id: PTR-YYYYMMDD-HHMMSS-XXXX
    raw_hash: sha256:string
```

Evidence PASS 기준:

```text
EVIDENCE_ID_EXISTS=True
EVIDENCE_POINTER_EXISTS=True
EVIDENCE_POLICY_EXISTS=True
RAW_TEXT_EXPORT_COUNT=0
INTERNAL_PATH_LEAK_COUNT=0
SOURCE_POLICY_EXISTS=True
```

### 6.7 Ontology Node Schema

```yaml
ontology_node:
  graph_node_id: GND-domain-slug-v1
  standard_node_id: ORG:DOC_CODE@EDITION_OR_REV | null
  node_type: Standard | Reference | Process | Defect | Risk | Action | Evidence | TrainingModule | Course | TailoringDecision
  label_ko: string
  label_en: string
  aliases: []
  domain: soldering | esd | harness | pcb_rework | quality | defense | space | general
  canonical_lang: EN
  source_lang: en | ko | zh-Hans | zh-Hant | ja | und
  status: active | deprecated | superseded
```

### 6.8 Graph Relation Schema

```yaml
graph_relation:
  schema_version: 1
  relation_id: REL-YYYYMMDD-HHMMSS-XXXX
  source_graph_node_id: GND-domain-source-v1
  target_graph_node_id: GND-domain-target-v1
  relation_type: FOLLOWS_STANDARD | APPLIES_TO | MENTIONS | SUPPORTS_RULE | HAS_EVIDENCE | USED_BY_MODULE | REQUIRES_APPROVAL | CONTRADICTS | SUPERSEDES
  evidence_ids:
    - EVD-YYYYMMDD-HHMMSS-XXXX
  confidence: low | medium | high
  created_by: string
  created_at: datetime
```

### 6.9 Tailoring Pack Schema

```yaml
tailoring_pack:
  schema_version: 1
  tailoring_pack_id: TPK-<project>-<version>
  project_scope:
    project_name: string
    domain: string
    customer_or_context: string | null
  decisions:
    - decision_id: DEC-YYYYMMDD-HHMMSS-XXXX
      action: apply | exclude | strengthen | relax | clarify
      target_standard_node_id: ORG:DOC_CODE@EDITION_OR_REV
      target_library_id: LIB-domain-slug-v1
      evidence_ids:
        - EVD-YYYYMMDD-HHMMSS-XXXX
      rationale: string
      approver_id: string
      approved_at: datetime
  status: draft | review_ready | approved | published | deprecated
```

Tailoring 금지:

```text
카드 생성의 자동 부속물로 처리 금지
승인 없는 tailoring publish 금지
evidence 없는 decision 금지
유료 표준 원문을 pack/out 산출물에 장문 포함 금지
```

### 6.10 Library Index Schema

```yaml
library_index:
  schema_version: 1
  created_at: datetime
  records:
    - library_id: LIB-domain-slug-v1
      standard_node_id: ORG:DOC_CODE@EDITION_OR_REV | null
      graph_node_ids:
        - GND-domain-slug-v1
      evidence_ids:
        - EVD-YYYYMMDD-HHMMSS-XXXX
      card_type: Standard | Reference | TrainingReference | PracticeEvidence | TailoringDecision
      title: string
      canonical_lang: EN
      source_lang: en | ko | zh-Hans | zh-Hant | ja | und
      status: published
      visibility: public | internal | restricted
      policy:
        raw_text_export_allowed: false
        pointer_only_required: true
```

### 6.11 Bridge Trace Index Schema

```yaml
bridge_trace_index:
  schema_version: 1
  created_at: datetime
  traces:
    - trace_id: TRC-YYYYMMDD-HHMMSS-XXXX
      request_id: REQ-YYYYMMDD-HHMMSS-XXXX
      course_id: CRS-...
      module_id: MOD-...
      binding_id: BND-...
      evidence_ids:
        - EVD-YYYYMMDD-HHMMSS-XXXX
      library_ids:
        - LIB-domain-slug-v1
      policy_result: PASS | HOLD | FAIL
      hold_reason: string | null
      created_at: datetime
```

### 6.12 F07 Library Admin Alignment

관리 CLI 또는 Admin UI는 다음 루프를 따른다.

```text
/lib add -> Validate -> Ingest -> Card -> Evidence -> Graph Preview -> /lib map --build -> /lib verify
```

필수 커맨드 계약:

| Command | 목적 | PASS 기준 |
|---|---|---|
| /lib add | 최소 입력 등록 | Validate 실패 시 카드 생성 금지 |
| /lib add --dry-run | 등록 계획 출력 | write 없음 |
| /lib map --build | Shelf 1D Timeline 생성 | HTML과 index 갱신 |
| /lib show <standard_node_id> | 카드 조회 | Standard/Reference/Evidence 표시 |
| /lib verify | 포인터와 중복 검증 | issues 발견 시 종료 코드 1 권장 |

### 6.13 Library Gate

| Gate | PASS 기준 | ProofPack |
|---|---|---|
| L-G1 Manifest | library_manifest schema valid | reports/proofpacks/library/library_manifest_validation_YYYYMMDD.md |
| L-G2 Language | canonical_lang=EN, source_lang BCP47 | reports/proofpacks/library/language_ssot_validation_YYYYMMDD.md |
| L-G3 ID | standard_node_id/library_id/graph_node_id/evidence_id 정렬 | reports/proofpacks/library/id_alignment_validation_YYYYMMDD.md |
| L-G4 Evidence | evidence_id exists, raw export 0 | reports/proofpacks/library/evidence_pointer_validation_YYYYMMDD.md |
| L-G5 Graph | graph relation valid | reports/proofpacks/library/graph_validation_YYYYMMDD.md |
| L-G6 Index | library_index and bridge_trace_index exist | reports/proofpacks/library/library_index_validation_YYYYMMDD.md |
| L-G7 Bridge Boundary | UI/Bridge DB direct access 0 | reports/proofpacks/library/bridge_boundary_validation_YYYYMMDD.md |
| L-G8 Backup | backup and restore dry-run | reports/proofpacks/library/backup_restore_YYYYMMDD.md |

### 6.14 Library Definition of Done

```text
LIBRARY_MANIFEST_PASS=True
LANGUAGE_SSOT_PASS=True
ID_ALIGNMENT_PASS=True
PROMOTION_TRACE_LINK_PASS=True
STANDARD_REFERENCE_SPLIT_PASS=True
EVIDENCE_POINTER_PASS=True
EVIDENCE_ID_FIRST_CLASS_PASS=True
ONTOLOGY_NODE_PASS=True
GRAPH_VALIDATION_PASS=True
TAILORING_APPROVAL_PASS=True
LIBRARY_INDEX_PASS=True
BRIDGE_TRACE_INDEX_PASS=True
UI_BRIDGE_DB_DIRECT_ACCESS_COUNT=0
SECURITY_POLICY_PASS=True
BACKUP_RESTORE_PASS=True
```

---

## 7. QLIB Bridge Contract Spec

Bridge는 Library Core와 Skillup Education 사이의 유일한 지식 접근 경계다. Bridge는 DB가 아니라 계약이다.

### 7.1 Bridge 원칙

```text
Bridge는 Library DB를 외부에 노출하지 않는다.
Bridge는 index와 policy만 읽는다.
Bridge는 evidence 없는 답변을 PASS로 만들지 않는다.
Bridge는 raw path, internal path, secret, raw prompt, paid standard raw text를 차단한다.
Bridge는 실패 시 fail-closed로 HOLD한다.
```

### 7.2 Bridge Operation 목록

| Operation | 목적 | 입력 | 출력 |
|---|---|---|---|
| Bridge.resolve_terms | 질문·단원·표준 용어 정규화 | request, course, module, query | normalized terms, candidate graph nodes |
| Bridge.retrieve_evidence | 허용 scope 안에서 evidence 조회 | binding, terms, role, policy | evidence bundle |
| Bridge.check_policy | 원문 출력, 권한, visibility, paid standard 정책 확인 | role, evidence, requested output | PASS/HOLD/FAIL |
| Bridge.explain_trace | trace_id 기반 근거 설명 | trace_id | trace summary |
| Bridge.bind_course_library | course/module/library scope 연결 검증 | course, module, standard_pack | binding result |
| Bridge.health | 계약, index, policy 상태 확인 | none | PASS/HOLD |

### 7.3 공통 Bridge Request

Bridge Request의 `query`는 전송 가능한 transient field다. 저장 금지 원칙과 충돌하지 않도록 query 원문은 요청 처리 중에만 사용하고, 로그에는 query_summary, query_hash, risk_flags만 남긴다.

```json
{
  "request_id": "REQ-YYYYMMDD-HHMMSS-XXXX",
  "tenant_id": "TEN-customer-or-service",
  "organization_id": "ORGUNIT-company",
  "cohort_id": "COH-202605-ipc-basic",
  "course_id": "CRS-QUALI-SKILLUP-202605",
  "module_id": "MOD-QUALI-PCB-REWORK-V1",
  "binding_id": "BND-CRS-QUALI-SKILLUP-202605-MOD-QUALI-PCB-REWORK-V1-V1",
  "contract_version": "1.0.0",
  "user_role": "student",
  "user_id_hash": "USH-hash",
  "question_intent": "explain | compare | apply | assess | ask_standard_raw | troubleshooting",
  "query": "student question text",
  "query_handling": {
    "query_is_transient_only": true,
    "store_raw_query": false,
    "store_query_summary": true,
    "store_query_hash": true,
    "query_hash_algorithm": "sha256"
  },
  "log_payload": {
    "query_summary": "short safe summary",
    "query_hash": "sha256:<hex>",
    "risk_flags": ["paid_standard_raw_request"]
  },
  "risk_flags": ["paid_standard_raw_request"],
  "evidence_policy": {
    "evidence_required": true,
    "missing_evidence_action": "HOLD",
    "raw_standard_text_allowed": false,
    "internal_path_allowed": false
  }
}
```

저장 허용 필드:

```text
request_id
trace_id
tenant_id
organization_id
cohort_id
course_id
module_id
binding_id
user_id_hash
question_intent
query_summary
query_hash
risk_flags
evidence_ids
hold_reason
answer_status
created_at
```

저장 금지 필드:

```text
raw query
raw prompt
full free-text answer
internal path
local route
secret
paid standard raw text
```


### 7.4 Bridge.resolve_terms Response

```json
{
  "trace_id": "TRC-YYYYMMDD-HHMMSS-XXXX",
  "status": "PASS",
  "normalized_terms": [
    {
      "input": "rework approval",
      "term_ko": "수리 승인",
      "term_en": "rework approval",
      "graph_node_id": "GND-risk-repair-without-approval-v1",
      "confidence": "high"
    }
  ],
  "hold_reason": null
}
```

### 7.5 Bridge.retrieve_evidence Response

```json
{
  "trace_id": "TRC-YYYYMMDD-HHMMSS-XXXX",
  "status": "PASS",
  "evidence_bundle": {
    "evidence_ids": ["EVD-YYYYMMDD-HHMMSS-0001"],
    "library_ids": ["LIB-ref-pcb-rework-approval-v1"],
    "graph_node_ids": ["GND-risk-repair-without-approval-v1"],
    "summary": "Evidence summary for education-safe answer.",
    "source_labels": [
      {
        "document_name": "Document Name",
        "revision": "Rev",
        "year": "YYYY",
        "section_label": "Section label"
      }
    ]
  },
  "policy": {
    "raw_text_export_allowed": false,
    "student_visible": true,
    "internal_path_leak_count": 0
  },
  "hold_reason": null
}
```

### 7.6 Bridge.check_policy Response

```json
{
  "trace_id": "TRC-YYYYMMDD-HHMMSS-XXXX",
  "policy_result": "PASS | HOLD | FAIL",
  "hold_reason": "HOLD_COPYRIGHT | HOLD_PERMISSION | HOLD_NO_EVIDENCE | null",
  "output_constraints": {
    "allow_summary": true,
    "allow_short_reference_label": true,
    "allow_raw_standard_text": false,
    "allow_internal_path": false,
    "allow_instructor_guide_raw": false
  },
  "blocked_fields": ["raw_text", "internal_path", "secret", "raw_prompt"]
}
```

### 7.7 Bridge.explain_trace Response

```json
{
  "trace_id": "TRC-YYYYMMDD-HHMMSS-XXXX",
  "request_id": "REQ-YYYYMMDD-HHMMSS-XXXX",
  "course_id": "CRS-QUALI-SKILLUP-202605",
  "module_id": "MOD-QUALI-PCB-REWORK-V1",
  "binding_id": "BND-...",
  "evidence_ids": ["EVD-YYYYMMDD-HHMMSS-0001"],
  "library_ids": ["LIB-ref-pcb-rework-approval-v1"],
  "policy_result": "PASS",
  "hold_reason": null,
  "visible_trace_summary": "This answer used approved Library evidence and student-safe policy."
}
```

### 7.8 Bridge Error Codes

| Code | 의미 | 처리 |
|---|---|---|
| BRG-001 INDEX_MISSING | library_index 또는 bridge_trace_index 없음 | Bridge start HOLD |
| BRG-002 CONTRACT_INVALID | request schema invalid | 요청 거절 |
| BRG-003 BINDING_NOT_FOUND | course_library_binding 없음 | Skillup course active 금지 |
| BRG-004 EVIDENCE_NOT_FOUND | evidence_id 없음 | HOLD_NO_EVIDENCE |
| BRG-010 POLICY_DENIED | policy 위반 | HOLD 또는 BLOCK |
| BRG-011 RAW_EXPORT_BLOCKED | 원문 출력 시도 차단 | 안전 요약으로 전환 또는 HOLD |
| BRG-012 PATH_LEAK_BLOCKED | internal path 출력 차단 | 출력 필드 제거 |
| BRG-020 TRACE_WRITE_FAILED | trace 저장 실패 | 답변 PASS 금지 |
| BRG-030 HEALTH_HOLD | Bridge health 불량 | Bridge 사용 중지 |

### 7.9 Bridge Gate

| Gate | PASS 기준 | ProofPack |
|---|---|---|
| B-G1 Contract | request/response schema valid | reports/proofpacks/bridge/bridge_contract_schema_YYYYMMDD.md |
| B-G2 Index | library_index, bridge_trace_index 존재 | reports/proofpacks/bridge/bridge_index_health_YYYYMMDD.md |
| B-G3 Policy | paid standard raw export 차단 | reports/proofpacks/bridge/policy_guard_validation_YYYYMMDD.md |
| B-G4 Trace | trace_id 생성과 조회 PASS | reports/proofpacks/bridge/trace_smoke_YYYYMMDD.md |
| B-G5 Boundary | DB direct access 0, path leak 0 | reports/proofpacks/bridge/boundary_scan_YYYYMMDD.md |
| B-G6 E2E | Skillup 질문 1개 evidence 답변 PASS | reports/proofpacks/bridge/skillup_e2e_bridge_YYYYMMDD.md |

### 7.10 Bridge Definition of Done

```text
BRIDGE_CONTRACT_SCHEMA_PASS=True
BRIDGE_INDEX_HEALTH_PASS=True
BRIDGE_DB_DIRECT_ACCESS_COUNT=0
BRIDGE_DB_WRITE_COUNT=0
RAW_STANDARD_TEXT_EXPORT_COUNT=0
INTERNAL_PATH_LEAK_COUNT=0
EVIDENCE_REQUIRED_ENFORCED=True
TRACE_ID_CREATED=True
TRACE_EXPLAIN_PASS=True
FAIL_CLOSED_ON_POLICY_DENIED=True
```

---

### 7.11 Bridge Logging and Redaction Gate

Bridge는 원문 질의 저장을 금지한다. 원문 query는 request 처리 중 메모리에서만 사용하고, response 완료 또는 HOLD 처리 후 폐기한다.

검증 기준:

```text
RAW_QUERY_STORED_COUNT=0
RAW_PROMPT_STORED_COUNT=0
QUERY_SUMMARY_EXISTS=True
QUERY_HASH_EXISTS=True
RISK_FLAGS_EXISTS=True
TRACE_ID_EXISTS=True
```

| 항목 | 저장 여부 | 이유 |
|---|---:|---|
| query 원문 | 금지 | 개인정보·기밀·유료 표준 요청 노출 위험 |
| query_summary | 허용 | 리뷰와 통계에 필요한 최소 요약 |
| query_hash | 허용 | 중복 질의 분석과 감사 추적 |
| risk_flags | 필수 | 정책 차단과 HOLD 근거 |
| evidence_ids | 필수 | 답변 근거 추적 |
| full answer text | 기본 금지 | 교육 품질 검토 시 별도 승인 필요 |

---


## 8. QLIB Skillup Education Module

### 8.1 모듈 선언

Skillup은 도서관 정본 지식을 교육 효과로 변환하는 교육 실행 모듈이다. 단순 챗봇이나 자료 뷰어가 아니다.

### 8.2 책임

| 책임 | 설명 |
|---|---|
| Course | 교육 과정과 cohort 운영 |
| Module | 단원, 표준 교육모듈, 학습목표 관리 |
| Binding | course, module, library scope, standard pack 연결 |
| Question | 학생 질문 접수와 intent 분류 |
| Answer | evidence 기반 교육 답변 생성 |
| HOLD | 근거 부족, 권한 부족, 정책 위반 시 안전 중지 |
| Review | 답변 품질과 근거 매칭 검토 |
| Assessment | 실습 평가 기준과 점수 초안 관리 |
| Logs | login, qa, review, assessment, approval, permission event 저장 |
| Analytics | 집계 통계와 개선 후보 생성 |

### 8.3 금지

1. Library DB 또는 Warehouse DB를 직접 조회하지 않는다.
2. evidence 없는 답변을 단정하지 않는다.
3. 강사용 해설서 원문을 학생에게 노출하지 않는다.
4. 질문 로그를 자동으로 교재나 도서관 정본에 반영하지 않는다.
5. 개인 식별 가능한 학습 데이터를 마케팅에 직접 사용하지 않는다.
6. Bridge trace 없는 정상 답변을 PASS로 기록하지 않는다.

### 8.4 Module Manifest

```yaml
module_manifest:
  schema_version: 1
  contract_version: 1.0.0
  tenant_context:
    tenant_id: global | TEN-customer-or-service
    organization_id: global | ORGUNIT-company
  module_id: MOD-QUALI-PCB-REWORK-V1
  module_family: QUALI | IPC | ECSS | NASA | INTERNAL
  module_title: PCB Rework and Repair
  module_version: v1
  status: draft | active | deprecated
  owner: QLIB
  learning_objectives:
    - objective_id: OBJ-001
      title: "승인 없는 수리 위험 이해"
      linked_library_scope:
        library_ids:
          - LIB-ref-pcb-rework-approval-v1
        graph_node_ids:
          - GND-risk-repair-without-approval-v1
        evidence_ids:
          - EVD-YYYYMMDD-HHMMSS-0001
  standard_pack_refs:
    - SPK-QUALI-REWORK-V1
  required_library_scope:
    library_ids:
      - LIB-ref-pcb-rework-approval-v1
    graph_node_ids:
      - GND-risk-repair-without-approval-v1
    evidence_ids:
      - EVD-YYYYMMDD-HHMMSS-0001
  evidence_policy:
    evidence_required: true
    missing_evidence_action: HOLD
    raw_standard_text_allowed_for_student: false
    internal_path_allowed: false
  assessment_map:
    - assessment_id: ASM-REWORK-001
      objective_id: OBJ-001
      evidence_required: true
  telemetry_policy:
    event_logging_required: true
    learning_analytics_allowed: true
    marketing_use_default: aggregated_only
```

### 8.5 Course Library Binding

```yaml
course_library_binding:
  schema_version: 1
  contract_version: 1.0.0
  tenant_context:
    tenant_id: TEN-customer-or-service
    organization_id: ORGUNIT-company
    cohort_id: COH-202605-ipc-basic
  idempotency_key: IDEMP-sha256
  revision: 1
  binding_id: BND-CRS-QUALI-SKILLUP-202605-MOD-QUALI-PCB-REWORK-V1-V1
  course_id: CRS-QUALI-SKILLUP-202605
  module_id: MOD-QUALI-PCB-REWORK-V1
  binding_version: v1
  active_from: 2026-05-11
  active_until: 2026-12-31
  deprecated_after: null
  backward_compatibility: compatible
  standard_pack_ids:
    - SPK-QUALI-REWORK-V1
  library_scope:
    allowed_library_ids:
      - LIB-ref-pcb-rework-approval-v1
    allowed_graph_node_ids:
      - GND-risk-repair-without-approval-v1
    allowed_evidence_ids:
      - EVD-YYYYMMDD-HHMMSS-0001
    allowed_tailoring_pack_ids:
      - TPK-QUALI-SKILLUP-V1
  answer_policy:
    evidence_required: true
    missing_evidence_action: HOLD
    unsupported_judgement_action: HOLD
    raw_path_leak_action: BLOCK
    raw_standard_text_action: BLOCK
  role_policy:
    student:
      evidence_depth: student_safe
      instructor_guide_raw_allowed: false
    instructor:
      evidence_depth: instructor_safe
      instructor_guide_raw_allowed: true
    reviewer:
      evidence_depth: review_trace
    admin:
      evidence_depth: audit_trace
```

과정은 binding이 없으면 active 상태가 될 수 없다.

### 8.6 질문 답변 흐름

```text
학생 질문
  -> course_id, module_id, binding_id 확인
  -> 사용자 권한 확인
  -> question_intent와 risk_flag 분류
  -> Bridge.resolve_terms
  -> Bridge.retrieve_evidence
  -> Bridge.check_policy
  -> evidence 있으면 교육용 답변 생성
  -> evidence 없으면 HOLD
  -> raw standard text 요청이면 BLOCK 또는 안전 요약
  -> qa_logs에 request_id, trace_id, evidence_id summary 저장
  -> analytics_event 생성
  -> 학생 화면 표시
```

### 8.7 Answer Response 기준

```yaml
skillup_answer:
  request_id: REQ-YYYYMMDD-HHMMSS-XXXX
  trace_id: TRC-YYYYMMDD-HHMMSS-XXXX
  course_id: CRS-...
  module_id: MOD-...
  status: ANSWER | HOLD | BLOCK
  answer:
    student_safe_summary: string
    key_points:
      - string
    cannot_answer_reason: string | null
  evidence:
    evidence_ids:
      - EVD-YYYYMMDD-HHMMSS-XXXX
    visible_source_labels:
      - document_name: string
        revision: string
        year: string
        section_label: string
  policy:
    raw_text_export_count: 0
    internal_path_leak_count: 0
    instructor_guide_raw_leak_count: 0
  logs:
    qa_log_event_id: EVT-YYYYMMDD-HHMMSS-XXXX
```

### 8.8 역할 권한

| 역할 | 허용 | 금지 |
|---|---|---|
| 미승인 회원 | 승인 대기 상태 확인 | 교재, 질문, 복습, 평가, 해설서, 관리자 기능 |
| 학생 | 승인 과정 교재 보기, 질문, 복습, 평가 기준 확인 | 해설서 원문, 관리자 화면, 전체 로그, raw 표준 원문 |
| 강사 | 학생 기능, 강사용 해설, 질문 로그, 평가 초안 | 계정 승인, 권한 회수, 자료 판본 변경 |
| 검토자 | 답변 품질, 근거 매칭, HOLD 점검 | 학생 개인정보 불필요 열람, 계정 승인 |
| 관리자 | 계정 발급, 과정 배정, 세션, 권한 회수, 판본 확인 | 로그 삭제, 승인 없는 자료 교체 |

권한 변경은 항상 event로 남긴다.

```yaml
permission_event:
  event_id: EVT-YYYYMMDD-HHMMSS-XXXX
  actor_id: string
  target_user_id_hash: USH-hash
  before_role: string
  after_role: string
  reason: string
  changed_at: datetime
```

### 8.9 Skillup UI 기준

학생 첫 화면:

```text
과정명
남은 사용 기간
강의 세션 상태
단원 카드
질문하기
복습하기
실습 평가 기준
내 질문 기록
```

강사 첫 화면:

```text
오늘 과정
세션 코드 상태
단원별 교재
단원별 강사용 해설
학생 질문 최근 목록
확인 필요 질문 목록
실습 평가 기준과 점수 초안
```

검토자 첫 화면:

```text
HOLD 질문
근거 부족 답변
정책 차단 내역
Evidence 매칭 검토
Trace 조회
```

관리자 첫 화면:

```text
사용자 승인
과정 배정
역할 권한
세션 관리
자료 판본 확인
로그 조회
Release Board 상태
```

### 8.10 Skillup Gate

| Gate | PASS 기준 | ProofPack |
|---|---|---|
| S-G1 Module | module_manifest valid | reports/proofpacks/skillup/module_manifest_validation_YYYYMMDD.md |
| S-G2 Binding | course_library_binding valid | reports/proofpacks/skillup/course_binding_validation_YYYYMMDD.md |
| S-G3 Bridge | Bridge health PASS | reports/proofpacks/skillup/bridge_health_YYYYMMDD.md |
| S-G4 Answer | evidence answer/HOLD flow PASS | reports/proofpacks/skillup/answer_flow_validation_YYYYMMDD.md |
| S-G5 Role | role access matrix PASS | reports/proofpacks/skillup/role_access_validation_YYYYMMDD.md |
| S-G6 Logs | qa/event logs with trace_id | reports/proofpacks/skillup/log_trace_validation_YYYYMMDD.md |
| S-G7 UI | customer view no leak | reports/proofpacks/skillup/customer_view_validation_YYYYMMDD.md |
| S-G8 Backup | backup and restore dry-run | reports/proofpacks/skillup/backup_restore_YYYYMMDD.md |

### 8.11 Skillup Definition of Done

```text
MODULE_MANIFEST_PASS=True
COURSE_LIBRARY_BINDING_PASS=True
STANDARD_PACK_LINK_PASS=True
BRIDGE_BOUNDARY_PASS=True
ANSWER_EVIDENCE_PASS=True
EVIDENCE_ID_USED_IN_ANSWER=True
HOLD_POLICY_PASS=True
ROLE_ACCESS_PASS=True
QA_LOG_TRACE_PASS=True
ANALYTICS_GOVERNANCE_PASS=True
CUSTOMER_VIEW_PASS=True
BACKUP_RESTORE_PASS=True
INTEGRATED_RELEASE_BOARD_UPDATED=True
```

---

### 8.12 AI Model, Cost, Rate Limit, and Fallback Policy

Skillup은 AI 답변 모듈이므로 모델 라우팅, 비용 상한, 고위험 질문 승격 비율, rate limit, fallback을 문서 단계에서 고정한다. 모델명은 구현 시점의 사용 가능 모델로 매핑하되, 정책 역할명은 아래와 같이 유지한다.

```yaml
ai_policy:
  schema_version: 1
  contract_version: 1.0.0
  default_router: low_cost_classifier
  model_routes:
    intent_classification:
      model_role: classifier_small
      max_latency_ms: 1500
    student_answer:
      model_role: answer_standard
      evidence_required: true
      max_latency_ms: 8000
    high_risk_escalation:
      model_role: reviewer_grade
      escalation_rate_limit_percent: 5
      reviewer_approval_required: true
    reviewer_analysis:
      model_role: reviewer_grade
      admin_approval_required: false
    premium_deep_analysis:
      model_role: expert_grade
      admin_approval_required: true
  monthly_budget:
    tenant_budget_cap: number
    cohort_budget_cap: number
    alert_at_percent: 80
    hard_stop_at_percent: 100
  rate_limit:
    per_user_per_minute: integer
    per_user_per_day: integer
    per_cohort_per_hour: integer
    burst_policy: reject_or_queue
  fallback:
    bridge_error: HOLD_BRIDGE_ERROR
    evidence_missing: HOLD_NO_EVIDENCE
    budget_exhausted: HOLD_COST_LIMIT
    rate_limit_exceeded: HOLD_RATE_LIMIT
    model_unavailable: safe_template_answer_or_HOLD
```

AI 운영 PASS 기준:

```text
AI_MODEL_ROUTE_DEFINED=True
MONTHLY_BUDGET_CAP_DEFINED=True
RATE_LIMIT_DEFINED=True
HIGH_RISK_ESCALATION_RATE_DEFINED=True
FALLBACK_POLICY_DEFINED=True
EVIDENCE_REQUIRED_FOR_STUDENT_ANSWER=True
```

금지:

1. Evidence 없는 고비용 답변 생성.
2. 학생 실시간 질문에 expert_grade 모델을 기본값으로 사용하는 구조.
3. 월 비용 상한 초과 후 계속 자동 응답하는 구조.
4. 고위험 질문을 reviewer escalation 없이 확정 답변하는 구조.
5. rate limit 초과 요청을 silent success로 처리하는 구조.

---


## 9. Usage & Analytics Governance

### 9.1 Analytics 선언

Analytics는 운영 로그를 정본으로 바꾸는 모듈이 아니다. Analytics는 사용 패턴, 교육 효과, 콘텐츠 개선 후보, 마케팅용 집계 데이터셋을 만드는 거버넌스 계층이다.

현재 MVP에서는 Skillup 내부에 포함할 수 있다. 운영 데이터가 누적되고 마케팅 활용이 시작되면 독립 문서와 독립 모듈로 분리한다.

```text
MVP 단계: Master + Skillup 내부 Analytics 포함
운영 데이터 발생 단계: Usage Analytics Contract 분리
상용화/마케팅 사용 단계: Analytics Governance Module 분리
```

### 9.2 로그 분류

| 로그 | 목적 | 원문 저장 | 분석 사용 | 마케팅 사용 |
|---|---|---:|---:|---:|
| login_event | 접근 감사 | 최소 | 가능 | 불가 |
| qa_event | 질문/답변 운영 | raw prompt 저장 금지 원칙 | 가명 집계 가능 | 원문 사용 불가 |
| answer_trace_event | evidence/trace 검증 | trace summary만 | 가능 | 불가 |
| review_event | 검토 이력 | 필요 필드만 | 가능 | 불가 |
| assessment_event | 교육 평가 | 최소 | 가능 | 집계 가능 |
| permission_event | 권한 변경 감사 | 필요 | 감사 전용 | 불가 |
| marketing_aggregate_event | 집계 마케팅 데이터 | 원문 없음 | 가능 | 가능 |

### 9.3 QA Log Schema

```yaml
qa_log:
  contract_version: 1.0.0
  event_id: EVT-YYYYMMDD-HHMMSS-XXXX
  request_id: REQ-YYYYMMDD-HHMMSS-XXXX
  trace_id: TRC-YYYYMMDD-HHMMSS-XXXX
  tenant_id: TEN-customer-or-service
  organization_id: ORGUNIT-company
  cohort_id: COH-202605-ipc-basic
  user_id_hash: USH-hash
  query_summary: string
  query_hash: sha256:<hex>
  raw_query_stored: false
  course_id: CRS-...
  module_id: MOD-...
  binding_id: BND-...
  question_intent: explain | compare | apply | assess | ask_standard_raw | troubleshooting
  risk_flags: []
  answer_status: ANSWER | HOLD | BLOCK
  evidence_ids:
    - EVD-YYYYMMDD-HHMMSS-XXXX
  hold_reason: string | null
  prompt_text_stored: false
  answer_text_stored: false
  created_at: datetime
```

### 9.4 Analytics Mart Schema

```yaml
analytics_mart_event:
  contract_version: 1.0.0
  event_id: EVT-YYYYMMDD-HHMMSS-XXXX
  event_type: learning_question | evidence_hold | review_needed | assessment_result | content_gap | marketing_aggregate
  tenant_id: TEN-customer-or-service | anonymized_multi_tenant
  organization_id: ORGUNIT-company | anonymized_group
  cohort_id: COH-202605-ipc-basic | anonymized_group
  user_segment: student | instructor | reviewer | admin | anonymized_group
  user_id_hash: null
  course_id: CRS-...
  module_id: MOD-...
  standard_pack_id: SPK-...
  question_intent: string
  answer_status: ANSWER | HOLD | BLOCK
  evidence_count: integer
  hold_reason: string | null
  learning_outcome_signal: low | medium | high | unknown
  consent_scope: learning_analytics | marketing_aggregate | none
  retention_class: short | medium | long
  created_at: datetime
```

마케팅 데이터셋 금지 필드:

```text
user_id_hash
raw prompt
free-text answer
trace detail
internal path
specific student record
sensitive source label
```

### 9.5 Consent Record Schema

```yaml
consent_record:
  consent_id: CST-YYYYMMDD-HHMMSS-XXXX
  user_id_hash: USH-hash
  consent_scope:
    learning_analytics: true
    marketing_aggregate: false
    personalized_feedback: true
  consent_version: v1
  granted_at: datetime
  revoked_at: datetime | null
  retention_policy_id: RET-v1
```

### 9.6 Retention Policy

| 데이터 | 기본 보존 | 삭제/익명화 기준 |
|---|---:|---|
| permission_event | 장기 | 감사 목적 범위 내 보존 |
| qa_event | 중기 | 원문 없음, trace summary만 보존 |
| answer_trace_event | 중기 | Library evidence 추적 목적 |
| analytics_mart_event | 중기 또는 장기 | 집계화 후 장기 보존 가능 |
| marketing_aggregate_event | 장기 가능 | 완전 익명 집계만 허용 |
| raw prompt | 저장 금지 원칙 | 저장 시 별도 승인과 짧은 보존 필요 |

### 9.7 Analytics Feedback Loop

```text
qa/event logs
  -> Analytics Mart
  -> content_gap / misconception / frequent_question 감지
  -> improvement_candidate 생성
  -> Warehouse에 analytics_improvement_candidate로 입고
  -> Review
  -> Approved for Library
  -> Library publish
```

자동 Library 반영은 금지한다.

### 9.8 Analytics Gate

| Gate | PASS 기준 | ProofPack |
|---|---|---|
| A-G1 Log | raw prompt 저장 없음 | reports/proofpacks/analytics/raw_prompt_scan_YYYYMMDD.md |
| A-G2 Consent | consent scope 반영 | reports/proofpacks/analytics/consent_validation_YYYYMMDD.md |
| A-G3 Pseudonym | user_id_hash 정책 PASS | reports/proofpacks/analytics/pseudonym_validation_YYYYMMDD.md |
| A-G4 Mart | analytics mart schema valid | reports/proofpacks/analytics/mart_schema_validation_YYYYMMDD.md |
| A-G5 Marketing | 익명 집계만 포함 | reports/proofpacks/analytics/marketing_dataset_validation_YYYYMMDD.md |
| A-G6 Feedback | Warehouse 후보로만 환류 | reports/proofpacks/analytics/feedback_loop_validation_YYYYMMDD.md |

---

## 10. 표준 교육모듈 확장 구조

Skillup은 하나의 교육모듈만을 위한 구조가 아니다. IPC, ECSS, NASA, 사내 표준 교육모듈이 같은 계약으로 확장되어야 한다.

### 10.1 Standard Pack Manifest

```yaml
standard_pack_manifest:
  schema_version: 1
  contract_version: 1.0.0
  tenant_context:
    tenant_id: global | TEN-customer-or-service
    organization_id: global | ORGUNIT-company
  standard_pack_id: SPK-IPC-SOLDERING-V1
  pack_family: IPC | ECSS | NASA | INTERNAL | QUALI
  pack_title: IPC Soldering Standards Pack
  pack_version: v1
  status: draft | approved | deprecated
  standard_node_ids:
    - IPC:J-STD-001@RevH
  library_ids:
    - LIB-ipc-jstd-001-revh-v1
  graph_node_ids:
    - GND-ipc-jstd-001-revh-v1
  evidence_ids:
    - EVD-YYYYMMDD-HHMMSS-XXXX
  tailoring_pack_ids:
    - TPK-IPC-SOLDERING-V1
  policy:
    paid_standard_pointer_only: true
    raw_export_allowed: false
    student_summary_allowed: true
```

### 10.2 교육모듈 확장 조건

| 교육모듈 | 필요한 연결 | 활성화 조건 |
|---|---|---|
| IPC | IPC standard pack, soldering/rework/harness domain | standard_pack approved, binding PASS, Bridge health PASS |
| ECSS | ECSS standard pack, space quality domain | standard_pack approved, tailoring approved, Bridge health PASS |
| NASA | NASA workmanship pack, space manufacturing domain | standard_pack approved, evidence PASS, role policy PASS |
| 사내 표준 | internal standard pack, company policy domain | source rights clear, visibility policy PASS |

### 10.3 Extension Gate

```text
STANDARD_PACK_EXISTS=True
LIBRARY_SCOPE_APPROVED=True
EVIDENCE_ID_EXISTS=True
TAILORING_PACK_APPROVED=True
COURSE_LIBRARY_BINDING_PASS=True
BRIDGE_HEALTH_PASS=True
RAW_EXPORT_POLICY_PASS=True
```

---

## 11. 통합 관리자 UI

통합 관리자 UI는 각 모듈 UI를 하나의 Shell로 묶되, 데이터와 권한은 분리한다.

### 11.1 Quali Console 정보 구조

```text
Quali Console
  1. Warehouse
     - Captured
     - Classified
     - Review
     - Approved
     - Promotion Dry-run
     - Promoted
     - Trace

  2. Library
     - Standard / Reference Search
     - Evidence Cards
     - Ontology Graph
     - Library Index
     - Bridge Trace Index
     - Tailoring Packs
     - Publish Status

  3. Education
     - Courses
     - Modules
     - Cohorts
     - Learners
     - Questions
     - HOLD
     - Review
     - Assessment

  4. Bridge
     - Health
     - Contract Test
     - Policy Blocks
     - Trace Search
     - Evidence Retrieval

  5. Analytics
     - Usage Overview
     - Learning Effectiveness
     - Content Gaps
     - Consent Status
     - Marketing Aggregate

  6. Operations
     - Release Board
     - ProofPack
     - Backup / Restore
     - Security Scan
     - Rollback
```

### 11.2 첫 화면 대시보드

필수 카드:

| 카드 | 표시 항목 |
|---|---|
| Pipeline Health | Warehouse, Library, Bridge, Skillup, Analytics gate 상태 |
| Recent Approvals | 최근 승인된 warehouse_item_id, promotion_trace_id |
| Evidence Health | evidence_id 수, missing evidence 수, raw export count |
| Bridge Blocks | policy denied, path leak block, raw export block |
| Skillup Activity | answer, HOLD, BLOCK, review_needed |
| Analytics Consent | 동의율, 분석 제외 대상 수 |
| Release Board | PASS/HOLD/FAIL, ProofPack missing count |
| Risk Alerts | copyright, secret, PII, no evidence, no backup |

### 11.3 UI 상태 라벨

| 내부 상태 | 사용자 표시 | 의미 |
|---|---|---|
| PASS | 목표 | 기준 충족 |
| HOLD | 보류 | 추가 근거 또는 검토 필요 |
| FAIL | 결함 | 기준 미충족, 시정 필요 |

UI는 색상이나 아이콘만으로 상태를 전달하지 않고 반드시 텍스트 라벨을 같이 표시한다.

### 11.4 UI 금지

```text
raw path 표시 금지
secret 표시 금지
route/port/internal endpoint 표시 금지
raw prompt 표시 금지
paid standard raw text 표시 금지
학생에게 instructor guide raw 표시 금지
승인 전 Warehouse 항목을 Library 정본처럼 표시 금지
```

---

## 12. 통합 Release Board와 ProofPack Plan

PASS/HOLD는 선언이 아니라 증거 파일로 판단한다. 모든 release gate는 ProofPack 파일명을 가져야 한다.

### 12.1 Release Board Schema

```yaml
release_board:
  release_id: REL-YYYYMMDD-XXXX
  release_scope:
    warehouse: true
    library: true
    bridge: true
    skillup: true
    analytics: true
    ui: true
  status: PASS | HOLD | FAIL
  gates:
    - gate_id: W-G6
      gate_name: WAREHOUSE_PROMOTION_PASS
      status: PASS | HOLD | FAIL
      proofpack: reports/proofpacks/warehouse/promotion_trace_PTR-*.md
    - gate_id: L-G6
      gate_name: LIBRARY_INDEX_PASS
      status: PASS | HOLD | FAIL
      proofpack: reports/proofpacks/library/library_index_validation_YYYYMMDD.md
    - gate_id: B-G1
      gate_name: BRIDGE_CONTRACT_PASS
      status: PASS | HOLD | FAIL
      proofpack: reports/proofpacks/bridge/bridge_contract_schema_YYYYMMDD.md
    - gate_id: S-G4
      gate_name: SKILLUP_E2E_PASS
      status: PASS | HOLD | FAIL
      proofpack: reports/proofpacks/skillup/answer_flow_validation_YYYYMMDD.md
    - gate_id: A-G5
      gate_name: ANALYTICS_PRIVACY_PASS
      status: PASS | HOLD | FAIL
      proofpack: reports/proofpacks/analytics/marketing_dataset_validation_YYYYMMDD.md
  final_decision:
    decision: PASS | HOLD | FAIL
    decided_by: string
    decided_at: datetime
    rollback_plan: string
```

### 12.2 필수 ProofPack 목록

| 영역 | ProofPack | 목적 |
|---|---|---|
| Warehouse | reports/proofpacks/warehouse/warehouse_manifest_validation_YYYYMMDD.md | manifest 검증 |
| Warehouse | reports/proofpacks/warehouse/promotion_trace_PTR-*.md | 승격 추적 |
| Library | reports/proofpacks/library/language_ssot_validation_YYYYMMDD.md | 언어 SSOT 검증 |
| Library | reports/proofpacks/library/id_alignment_validation_YYYYMMDD.md | ID 정렬 검증 |
| Library | reports/proofpacks/library/evidence_pointer_validation_YYYYMMDD.md | evidence 검증 |
| Library | reports/proofpacks/library/library_index_validation_YYYYMMDD.md | index 검증 |
| Bridge | reports/proofpacks/bridge/bridge_contract_schema_YYYYMMDD.md | Bridge 계약 검증 |
| Bridge | reports/proofpacks/bridge/policy_guard_validation_YYYYMMDD.md | 정책 차단 검증 |
| Bridge | reports/proofpacks/bridge/trace_smoke_YYYYMMDD.md | trace 검증 |
| Skillup | reports/proofpacks/skillup/course_binding_validation_YYYYMMDD.md | course binding 검증 |
| Skillup | reports/proofpacks/skillup/answer_flow_validation_YYYYMMDD.md | evidence 답변 검증 |
| Skillup | reports/proofpacks/skillup/customer_view_validation_YYYYMMDD.md | 학생 화면 leak 검증 |
| Analytics | reports/proofpacks/analytics/consent_validation_YYYYMMDD.md | 동의 검증 |
| Analytics | reports/proofpacks/analytics/marketing_dataset_validation_YYYYMMDD.md | 마케팅 데이터 검증 |
| Operations | reports/proofpacks/ops/backup_restore_dryrun_YYYYMMDD.md | 백업·복원 검증 |
| Operations | reports/proofpacks/ops/rollback_plan_validation_YYYYMMDD.md | 롤백 검증 |
| Release | reports/proofpacks/release/release_board_REL-*.md | 최종 릴리즈 판정 |

### 12.3 Release Board PASS 기준

```text
WAREHOUSE_GATES_PASS=True
LIBRARY_GATES_PASS=True
BRIDGE_GATES_PASS=True
SKILLUP_GATES_PASS=True
ANALYTICS_GATES_PASS=True
BACKUP_RESTORE_PASS=True
ROLLBACK_PLAN_EXISTS=True
PROOFPACK_MISSING_COUNT=0
RAW_EXPORT_COUNT=0
INTERNAL_PATH_LEAK_COUNT=0
SECRET_LEAK_COUNT=0
PII_POLICY_VIOLATION_COUNT=0
FINAL_APPROVAL_EXISTS=True
```

### 12.4 Release Board HOLD 기준

아래 중 하나라도 발생하면 전체 release는 HOLD다.

```text
ProofPack 없음
Evidence 없음
Bridge trace 없음
course binding 없음
library_index 없음
bridge_trace_index 없음
backup restore dry-run 없음
paid standard raw text 노출
internal path 노출
secret 노출
raw prompt 마케팅 데이터 포함
동의 없는 마케팅 활용
```

---

### 12.5 운영 관측성, SLO, Alert Threshold

ProofPack은 배포 전 증거이고, 관측성은 배포 후 안전장치다. 운영 중 장애 기준은 다음으로 고정한다.

```yaml
observability_policy:
  schema_version: 1
  contract_version: 1.0.0
  slo:
    bridge_availability_monthly: 99.5
    skillup_answer_success_rate: 95
    evidence_retrieval_success_rate: 98
    p95_answer_latency_ms: 12000
    p95_bridge_latency_ms: 3000
    raw_leak_incidents: 0
    paid_standard_raw_export_incidents: 0
  alert_thresholds:
    bridge_5xx_rate_5min: 2
    hold_rate_spike_30min: 30
    evidence_missing_rate_30min: 10
    policy_denied_spike_30min: 10
    budget_usage_percent: 80
    rate_limit_reject_rate_10min: 20
  emergency_hold:
    trigger_raw_leak: true
    trigger_paid_standard_raw_export: true
    trigger_cross_tenant_access: true
    trigger_secret_leak: true
    trigger_restore_failure: true
```

필수 운영 지표:

| 영역 | 지표 | HOLD 기준 |
|---|---|---|
| Bridge | error rate, latency, evidence hit | 5xx 급증 또는 evidence hit 급락 |
| Skillup | answer status, HOLD rate, escalation rate | HOLD 비율 이상 급증 |
| Library | index freshness, evidence count, graph validation | index stale 또는 evidence 누락 |
| Warehouse | promotion queue, review backlog | 승인 병목 장기화 |
| Analytics | raw prompt scan, consent violation | 위반 1건 이상 |
| License | entitlement expiry, denied access | 만료 권한 사용 시도 |

### 12.6 Incident Severity and Post-Incident Review

```yaml
incident_record:
  incident_id: INC-YYYYMMDD-HHMMSS-XXXX
  severity: SEV1 | SEV2 | SEV3 | SEV4
  detected_at: datetime
  detected_by: monitor | user | admin | reviewer
  affected_tenant_ids: []
  affected_modules: warehouse | library | bridge | skillup | analytics | ui | release
  trigger: string
  emergency_hold_applied: true | false
  customer_visible: true | false
  root_cause: string | pending
  mitigation: string
  rollback_executed: true | false
  post_incident_review_due: date
  owner: string
```

| Severity | 기준 | 즉시 조치 |
|---|---|---|
| SEV1 | raw standard text, secret, cross-tenant data, 개인정보 중대 노출 | Emergency HOLD, 관련 기능 중지, rollback 검토 |
| SEV2 | Bridge 전면 장애, 교육 답변 대량 실패, backup/restore 실패 | Release HOLD, 장애 공지, hotfix 또는 rollback |
| SEV3 | 특정 과정 또는 특정 tenant 장애 | 해당 scope HOLD, 우회 안내 |
| SEV4 | 문서/표시/경미한 UI 오류 | 일반 수정 backlog |

Post-Incident Review 필수 항목:

```text
timeline
impact_scope
root_cause
what_detected_it
what_should_have_detected_it
proofpack_gap
contract_gap
schema_or_test_patch
owner
due_date
```

---


## 13. 검증 체계

### 13.1 PASS 단계 용어

| 단계 | 의미 | 필요한 증거 |
|---|---|---|
| Scenario PASS | 문서상 흐름이 모순 없음 | 시나리오 표와 기대 결과 |
| Contract Test PASS | JSON/YAML schema와 request/response가 검증됨 | contract test proofpack |
| Implementation PASS | 실제 코드, DB, API, UI가 실행됨 | test output, screenshots, logs |
| Operational PASS | backup, restore, rollback, release board까지 증거가 있음 | ops proofpack, release approval |

현재 문서의 100점은 Scenario PASS와 문서·설계 정합성 기준이다. Implementation PASS나 Operational PASS가 아니다.

### 13.2 End-to-End 시나리오

#### Scenario A. 전문가 암묵지가 도서관 정본이 되고 스킬업 답변에 쓰이는 경우

| 단계 | 기대 결과 |
|---|---|
| 전문가가 PCB Rework 승인 절차 노하우 제공 | source_id 생성 |
| Warehouse raw pointer와 raw_hash 저장 | raw immutable |
| provenance와 sensitivity 분류 | provenance 존재 |
| reviewer note와 confidence 입력 | reviewed 상태 |
| approved_for_library 처리 | approval record 존재 |
| promotion dry-run | Library write 없음 |
| Library Reference Card 생성 | library_id 생성 |
| ontology node와 graph relation 생성 | graph_node_id 생성 |
| Evidence pointer 생성 | evidence_id 생성, raw 원문 노출 없음 |
| course binding에 library scope 연결 | binding_id 생성 |
| 학생 질문 | request_id 생성 |
| Bridge evidence 조회 | trace_id 생성 |
| 답변 표시 | 학생 안전 요약, raw leak 0 |
| qa log와 analytics event 저장 | PII 최소화 |

#### Scenario B. 유료 표준 원문 장문 요청

| 단계 | 기대 결과 |
|---|---|
| 학생이 원문 전체 요청 | risk_flag 생성 |
| Bridge.check_policy | raw text export denied |
| Skillup 답변 | HOLD 또는 안전 요약 |
| qa log 저장 | raw prompt 저장 없음 |
| trace 조회 | trace_id 존재 |

#### Scenario C. IPC 교육모듈 추가

| 단계 | 기대 결과 |
|---|---|
| IPC standard pack 작성 | standard_pack_id 생성 |
| IPC module manifest 작성 | module_id 생성 |
| Library scope와 tailoring 승인 | approved 상태 |
| course/library binding 작성 | binding_id 생성 |
| Bridge health 확인 | PASS |
| 교육 과정 active 전환 | Gate PASS 필요 |

#### Scenario D. 학습 로그의 마케팅 활용

| 단계 | 기대 결과 |
|---|---|
| qa/event log 생성 | 운영 로그 저장 |
| Analytics Mart 변환 | 집계/가명/익명화 |
| 동의 상태 확인 | consent 반영 |
| marketing dataset 생성 | 익명 집계만 포함 |
| 개인 질문 원문 사용 시도 | 차단 |

#### Scenario E. 통합 관리자 UI에서 병목 확인

| 단계 | 기대 결과 |
|---|---|
| Console 접속 | Release Board 표시 |
| Warehouse hold 증가 | hold reason 표시 |
| Library evidence missing | evidence gap 표시 |
| Bridge policy block | block reason 표시 |
| Skillup HOLD 증가 | module/course 기준 표시 |
| Analytics consent issue | marketing 제외 표시 |

### 13.3 Contract Test 목록

| Test | 목적 |
|---|---|
| CT-ID-001 | 모든 ID regex 검증 |
| CT-ID-002 | standard_node_id/library_id/graph_node_id/evidence_id 매핑 검증 |
| CT-LANG-001 | canonical_lang=EN 고정 검증 |
| CT-LANG-002 | source_lang BCP47 검증 |
| CT-WH-001 | warehouse_item schema 검증 |
| CT-WH-002 | promotion_trace schema 검증 |
| CT-LIB-001 | Standard Card schema 검증 |
| CT-LIB-002 | Evidence Pointer schema 검증 |
| CT-BRG-001 | Bridge request/response schema 검증 |
| CT-BRG-002 | policy denied fail-closed 검증 |
| CT-SKL-001 | module_manifest schema 검증 |
| CT-SKL-002 | course_library_binding schema 검증 |
| CT-ANA-001 | consent record schema 검증 |
| CT-ANA-002 | marketing dataset forbidden fields 검증 |

### 13.4 Implementation Test 목록

| Test | 목적 |
|---|---|
| IT-E2E-001 | Warehouse approved item -> Library evidence -> Skillup answer |
| IT-E2E-002 | paid standard raw request -> BLOCK/HOLD |
| IT-E2E-003 | no evidence -> HOLD |
| IT-E2E-004 | no binding -> course active 금지 |
| IT-E2E-005 | path leak scan -> 0 |
| IT-E2E-006 | raw prompt stored scan -> 0 |
| IT-E2E-007 | role matrix access test |
| IT-E2E-008 | Analytics feedback candidate -> Warehouse 입고 |

### 13.5 Operational Test 목록

| Test | 목적 |
|---|---|
| OT-001 | backup scope 생성 |
| OT-002 | restore dry-run |
| OT-003 | rollback plan 검증 |
| OT-004 | release board proofpack 존재 검증 |
| OT-005 | final approval 기록 |
| OT-006 | post-deploy health check |
| OT-007 | incident HOLD 전환 |

---

## 14. 보안·저작권·개인정보 정책

### 14.1 저작권 정책

```text
유료 표준 원문 장문 출력 금지
원문 복사 대신 Evidence Pointer 사용
표준명, 개정, 연도, 조항명 수준 참조 허용
학생 화면에는 교육용 요약만 표시
policy 위반 시 Bridge fail-closed
```

### 14.2 개인정보 정책

```text
개인 식별 원문 최소 수집
user_id_hash 사용
raw prompt 저장 금지 원칙
마케팅 데이터는 익명 집계만 허용
동의 없는 마케팅 활용 금지
보존기간 경과 시 삭제 또는 익명화
```

### 14.3 비밀·내부 경로 정책

금지 출력:

```text
local path
internal route
port
secret
API key
raw prompt
raw standard text
raw instructor guide
private tacit knowledge
```

검증 기준:

```text
INTERNAL_PATH_LEAK_COUNT=0
SECRET_LEAK_COUNT=0
RAW_PROMPT_OUTPUT_COUNT=0
RAW_STANDARD_TEXT_EXPORT_COUNT=0
```

### 14.4 유료 표준 License Entitlement 정책

pointer-only 정책은 필수지만, 실제 운영에서는 누가 어떤 표준에 접근할 권한이 있는지도 기록해야 한다.

```yaml
license_entitlement:
  schema_version: 1
  contract_version: 1.0.0
  license_entitlement_id: LIC-YYYYMMDD-XXXX
  tenant_id: TEN-customer-or-service
  organization_id: ORGUNIT-company
  licensed_org: string
  standard_node_id: ORG:DOC_CODE@EDITION_OR_REV
  allowed_roles:
    - instructor
    - reviewer
    - admin
  allowed_use:
    student_safe_summary: true
    instructor_reference: true
    raw_text_export: false
    long_quote_export: false
    offline_export: false
  valid_from: date
  expiry_date: date
  source_contract_ref: string
  checked_at: datetime
  checked_by: string
  status: active | expired | suspended | unknown
```

권한 검사 규칙:

1. 유료 표준 evidence는 license_entitlement_id 없이는 student 답변에 사용하지 않는다.
2. student에게는 원문이 아니라 교육용 요약과 표준명·개정·연도·조항명 수준의 label만 제공한다.
3. instructor, reviewer, admin도 raw export는 기본 금지다.
4. expiry_date가 지난 entitlement는 자동 HOLD다.
5. licensed_org와 tenant_id가 불일치하면 cross-tenant license violation으로 HOLD한다.

PASS 기준:

```text
LICENSE_ENTITLEMENT_EXISTS_FOR_PAID_STANDARD=True
LICENSE_EXPIRY_VALID=True
LICENSED_ORG_SCOPE_MATCH=True
RAW_EXPORT_FOR_PAID_STANDARD_COUNT=0
CROSS_TENANT_LICENSE_VIOLATION_COUNT=0
```

### 14.5 역할 기반 정보 노출

| 정보 | 학생 | 강사 | 검토자 | 관리자 |
|---|---:|---:|---:|---:|
| 학생 안전 요약 | 허용 | 허용 | 허용 | 허용 |
| Evidence label | 허용 | 허용 | 허용 | 허용 |
| Evidence trace summary | 제한 | 허용 | 허용 | 허용 |
| raw path | 금지 | 금지 | 금지 | 금지 |
| 유료 표준 원문 | 금지 | 금지 | 금지 | 금지 |
| 강사용 해설 원문 | 금지 | 허용 | 허용 | 허용 |
| 전체 qa log | 본인 범위 | 과정 범위 | 검토 범위 | 운영 범위 |
| 마케팅 집계 | 금지 | 제한 | 제한 | 집계만 허용 |

---

## 15. 구현 로드맵

구현 로드맵은 두 가지 트랙을 구분한다. 전체 플랫폼을 처음부터 만든다면 Warehouse Promotion이 Skillup E2E보다 앞선다. 단, 이미 Library seed 데이터가 있고 교육 답변 흐름을 먼저 검증하려는 경우에는 Seeded Library Track으로 Skillup E2E를 선검증할 수 있다.

### 15.1 로드맵 전제

```text
Greenfield Track
  P0 Contract
  P1 Warehouse Promotion
  P2 Library Evidence / Index
  P3 Bridge
  P4 Skillup E2E
  P5 Analytics
  P6 Release / Operations

Seeded Library Track
  P0 Contract
  P1 Library Evidence / Bridge
  P2 Skillup E2E
  P3 Warehouse Promotion
  P4 Analytics
  P5 Release / Operations
```

본 문서의 기본 로드맵은 Greenfield Track이다. Seeded Library Track을 사용할 수 있는 조건은 다음이다.

```text
LIBRARY_SEED_DATA_EXISTS=True
LIBRARY_INDEX_EXISTS=True
EVIDENCE_POINTER_EXISTS=True
BRIDGE_TRACE_INDEX_EXISTS=True
WAREHOUSE_PROMOTION_NOT_REQUIRED_FOR_INITIAL_E2E=True
```

### 15.2 P0 — 계약과 경계 고정

| 작업 | 산출물 |
|---|---|
| Shared Contracts 분리 | schemas/shared_contracts/*.yaml |
| ID regex validator | tests/contracts/test_id_contract.py |
| Tenant isolation schema | schemas/shared_contracts/tenant_scope.schema.yaml |
| Contract versioning schema | schemas/shared_contracts/contract_meta.schema.yaml |
| Idempotency validator | tests/contracts/test_idempotency.py |
| Language SSOT validator | tests/contracts/test_language_ssot.py |
| Bridge Contract schema | schemas/bridge/*.json |
| Runtime boundary scan | reports/proofpacks/bridge/boundary_scan_YYYYMMDD.md |

### 15.3 P1 — Warehouse Promotion Foundation

| 작업 | 산출물 |
|---|---|
| warehouse item 입고 | warehouse_items.jsonl |
| tenant_scope 포함 item 검증 | warehouse tenant validation proofpack |
| review/approval state machine | review_state_machine proofpack |
| idempotent approve/promote | idempotency replay proofpack |
| promotion dry-run | promotion dry-run proofpack |
| promotion_trace 생성 | promotion_trace proofpack |

### 15.4 P2 — Library Evidence와 Index 생성

| 작업 | 산출물 |
|---|---|
| Library card/evidence 생성 | promotion_trace proofpack |
| evidence_id 생성기 | Library Evidence service |
| license entitlement 연결 | license entitlement proofpack |
| library_index 생성 | data/library/exports/indexes/library_index.json |
| bridge_trace_index 생성 | data/library/exports/indexes/bridge_trace_index.json |
| paid standard policy block | policy guard proofpack |

### 15.5 P3 — Bridge 연결

| 작업 | 산출물 |
|---|---|
| Bridge.retrieve_evidence 구현 | Bridge contract proofpack |
| transient query 처리 | raw_query_scan proofpack |
| Bridge.check_policy 구현 | policy_guard_validation |
| Bridge.explain_trace 구현 | trace_smoke proofpack |
| tenant/binding scope 검사 | bridge_scope_validation |

### 15.6 P4 — Skillup E2E

| 작업 | 산출물 |
|---|---|
| module_manifest 검증 | skillup module proofpack |
| course_library_binding 검증 | binding proofpack |
| rate limit / AI policy 적용 | ai_policy_validation proofpack |
| 질문 1개 E2E | answer_flow_validation |
| HOLD 1개 E2E | hold_policy_validation |
| role access 검증 | role_access_validation |

### 15.7 P5 — Analytics Governance

| 작업 | 산출물 |
|---|---|
| qa log 정책 적용 | raw_prompt_scan proofpack |
| tenant/cohort 집계 검증 | analytics_tenant_isolation proofpack |
| consent record | consent validation proofpack |
| analytics mart | mart schema validation |
| marketing aggregate validator | marketing dataset proofpack |
| feedback candidate loop | feedback loop proofpack |

### 15.8 P6 — Release Board와 운영 검증

| 작업 | 산출물 |
|---|---|
| 통합 Release Board 구현 | release_board_REL-*.md |
| SLO/alert threshold 설정 | observability proofpack |
| incident drill | incident_drill_INC-*.md |
| emergency HOLD drill | emergency_hold_validation |
| backup scope 검증 | backup proofpack |
| restore dry-run | restore proofpack |
| rollback plan | rollback plan proofpack |
| final approval | release approval record |

---

## 16. 개발 작업 루프

모든 구현 작업은 작은 Task 단위로 수행한다.

### 16.1 Task Template

```markdown
### Task Txx: 이름

- 한 줄 정의:
- 소속 모듈: Warehouse | Library | Bridge | Skillup | Analytics | UI | Ops
- 관련 계약: ID | Language | Evidence | Bridge | Promotion | Analytics
- 입력:
- 출력:
- 변경 파일:
- 제약:
  - 보안:
  - 저작권:
  - 개인정보:
  - 성능:
- 테스트:
- ProofPack:
- 완료 기준:
```

### 16.2 구현 루프

```text
Task 정의
  -> 설계/계약 확인
  -> 구현
  -> Contract Test
  -> Implementation Test
  -> ProofPack 생성
  -> Release Board 반영
  -> 기록
```

### 16.3 코딩 시작 금지 조건

```text
계약 schema 없음
ID 규칙 불명확
Evidence 정책 없음
Bridge 경계 불명확
raw export 정책 없음
개인정보 저장 정책 없음
테스트 기준 없음
ProofPack 위치 없음
```

---

## 17. 최종 정합성 매트릭스

| 항목 | 기준 | 판정 기준 |
|---|---|---|
| 상호 정합성 | Warehouse -> Library -> Bridge -> Skillup -> Analytics 흐름 | lifecycle 모순 0 |
| 데이터 정합성 | source_id부터 trace_id까지 연결 | orphan ID 0 |
| Evidence 정합성 | 답변에는 evidence_id가 있어야 함 | normal answer without evidence 0 |
| Language SSOT | canonical_lang=EN, source_lang BCP47 | violation 0 |
| 관리 정합성 | PASS/HOLD/FAIL와 proofpack 연결 | proofpack missing 0 |
| 기능 정합성 | 시나리오 A~E 구현 | E2E PASS |
| UI 정합성 | Console에서 병목 확인 | raw leak 0 |
| 보안 정합성 | secret/path/raw prompt 차단 | leak 0 |
| 개인정보 정합성 | consent/retention 반영 | violation 0 |
| 저작권 정합성 | pointer_only for paid standard | raw export 0 |
| 배포 정합성 | backup/restore/rollback/final approval | operational PASS |

---

## 18. 최종 Acceptance Checklist

### 18.1 문서·설계 Acceptance

```text
[ ] L0/L1/L2/L3 문서 구조가 명확하다.
[ ] Shared Contracts가 독립적으로 정의되어 있다.
[ ] Bridge Contract가 request/response/error code까지 정의되어 있다.
[ ] evidence_id가 1급 ID로 반영되어 있다.
[ ] standard_node_id/library_id/graph_node_id/evidence_id 역할이 분리되어 있다.
[ ] Language SSOT가 canonical_lang=EN, source_lang=BCP47로 정렬되어 있다.
[ ] 검수 점수 100은 문서·설계 기준임이 명시되어 있다.
[ ] 실제 구현 검수는 NOT_VALIDATED로 분리되어 있다.
[ ] ProofPack 파일명이 Release Board에 직접 연결되어 있다.
```

### 18.2 구현 Acceptance

```text
[ ] ID validator PASS
[ ] Language SSOT validator PASS
[ ] Warehouse item schema PASS
[ ] Promotion trace schema PASS
[ ] Library card schema PASS
[ ] Evidence pointer schema PASS
[ ] Bridge schema PASS
[ ] Skillup module/binding schema PASS
[ ] Analytics consent/mart schema PASS
[ ] E2E Scenario A PASS
[ ] E2E Scenario B PASS
[ ] E2E Scenario C PASS
[ ] E2E Scenario D PASS
[ ] E2E Scenario E PASS
```

### 18.3 운영 Acceptance

```text
[ ] Warehouse ProofPack PASS
[ ] Library ProofPack PASS
[ ] Bridge ProofPack PASS
[ ] Skillup ProofPack PASS
[ ] Analytics ProofPack PASS
[ ] Backup ProofPack PASS
[ ] Restore dry-run PASS
[ ] Rollback plan PASS
[ ] Security scan PASS
[ ] Privacy scan PASS
[ ] Copyright policy scan PASS
[ ] Release Board PASS
[ ] Final approval exists
```

---

## 19. v1.2 추가 보완 반영 매트릭스

| 번호 | 보완 요구 | 반영 위치 | PASS 기준 |
|---:|---|---|---|
| 1 | 단일 MD와 모듈형 문서 유지 원칙의 긴장 해소 | 1.1 | L0/L1 동기화 ProofPack 존재 |
| 2 | 레이어 의존 규칙 보정 | 3.3 | Application은 Integration/Contract/Core로 의존, Analytics는 이벤트만 수신 |
| 3 | 구현 로드맵 순서 보정 | 15.1~15.8 | Greenfield와 Seeded Library Track 구분 |
| 4 | Bridge query 원문 처리 | 7.3, 7.11, 9.3 | RAW_QUERY_STORED_COUNT=0 |
| 5 | 멀티기관/고객사/테넌트 경계 | 4.7, 8.5, 9.3, 9.4 | tenant_id, organization_id, cohort_id 검증 |
| 6 | schema migration / contract versioning | 4.8 | contract_version, migration_policy, deprecated_after 존재 |
| 7 | idempotency와 중복 승격 방지 | 4.9, 5.5, 5.6, 8.5 | 중복 promote/publish 0건 |
| 8 | AI 비용/모델/Rate Limit 정책 | 8.12 | 비용 상한, rate limit, fallback 존재 |
| 9 | 유료 표준 라이선스 권한 기록 | 4.10, 6.4, 6.6, 14.4 | license_entitlement_id 검증 |
| 10 | 운영 관측/사고 대응 기준 | 12.5, 12.6, 15.8 | SLO, alert, incident, emergency HOLD 기준 존재 |

---

## 20. 최종 판정

현재 이 문서는 다음 상태로 고정한다.

```text
GUIDEBOOK_COHERENCE_SCORE=100
GUIDEBOOK_SET_STATUS=PASS
GUIDEBOOK_VERSION=v1.2
GUIDEBOOK_BASIS=DOCUMENT_AND_DESIGN_COHERENCE
CONTRACT_TEST_STATUS=NOT_VALIDATED
IMPLEMENTATION_STATUS=NOT_VALIDATED
OPERATIONAL_STATUS=NOT_VALIDATED
DEPLOYMENT_STATUS=HOLD_UNTIL_IMPLEMENTATION_PROOFPACK_PASS
```

QLIB의 정답은 다음이다.

```text
One Governance
  - 공통 ID
  - 공통 언어 SSOT
  - 공통 Evidence
  - 공통 Bridge
  - 공통 Trace
  - 공통 Approval
  - 공통 Release Board

Modular Implementation
  - Warehouse
  - Library Core
  - Bridge
  - Skillup Education
  - Usage & Analytics
  - Integrated Console

Evidence-Based Operation
  - PASS는 증거 파일로만 인정
  - HOLD는 실패가 아니라 안전 중지
  - 정본은 Library Core
  - 개선 후보는 Warehouse로 환류
  - 교육모듈은 Bridge 계약으로 확장
```

이 문서는 개발 착수 기준서로 사용할 수 있다. 운영 배포 선언서는 아니다. 배포 가능 상태는 코드, DB, API, UI, 테스트, 백업, 복원, 롤백, Release Board, Final Approval 증거가 모두 붙은 뒤에만 선언한다.

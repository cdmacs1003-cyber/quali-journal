# QLIB_FINAL_WORKFLOW_AND_RELEASE_GUARDRAIL_ADDENDUM_20260702

문서 상태: ADDENDUM_CANDIDATE_FOR_FINAL_WORKFLOW_AND_RELEASE_GUARDRAIL  
작성일: 2026-07-02 KST  
대상 원본: `QLIB_FINAL_DEVELOPMENT_SCHEDULE_REPORT_20260522_FINAL.md`, `QLIB_COMPLETE_DEVELOPMENT_GUIDEBOOK_20260511_v1_2.md`  
변경 방식: 원본 직접 수정 금지. 본 문서는 post-R9ZNW259 실제 베타 상태와 제품 완료·배포까지의 최종 작업 흐름을 반영하는 Addendum으로 적용한다.  
권장 repo 저장 위치: `H:\a\퀄리저널_track_a_clean_standalone\docs\governance\addenda\QLIB_FINAL_WORKFLOW_AND_RELEASE_GUARDRAIL_ADDENDUM_20260702.md`  
권장 장기기억 보관 위치: `H:\장기기억\docs\codex\2026\07\governance_addenda\QLIB_FINAL_WORKFLOW_AND_RELEASE_GUARDRAIL_ADDENDUM_20260702.md`

---

## 0. Addendum 판정

```text
ADDENDUM_DECISION=APPROVE_AS_FINAL_WORKFLOW_AND_RELEASE_GUARDRAIL_CANDIDATE
ORIGINAL_SCHEDULE_REPORT_DIRECT_MODIFICATION=FORBIDDEN
ORIGINAL_COMPLETE_GUIDEBOOK_DIRECT_MODIFICATION=FORBIDDEN
CHANGE_POLICY=ADDENDUM_ONLY_UNTIL_EXPLICIT_REPO_PLACEMENT_GATE
```

본 Addendum은 QLIB 최종 개발 완료와 배포까지의 작업 흐름을 고정한다.  
본 Addendum 자체는 구현 완료 선언이나 배포 승인서가 아니다.

---

## 1. 최종 개발 헌법 vFinal

```text
QLIB는 단일 앱이 아니라 Warehouse, Library Core, Bridge, Skillup Education, Usage & Analytics, Integrated UI, Operations가 연결된 지식 운영 제품군이다.

현재 베타는 Skillup 입출력 UI 초간단 세로 절단면이며, 전체 제품 완료가 아니다.

앞으로의 개발은 Warehouse → Library → Bridge → Skillup → Feedback → Analytics → Integrated UI → ProofPack → Release Board → Deployment 순서로 진행한다.

Skillup은 Library/Warehouse DB를 직접 조회하지 않고, Bridge 계약을 통해 Evidence와 Trace만 사용한다.

Evidence 없는 답변, Trace 없는 답변, ProofPack 없는 PASS, rollback 없는 배포, raw leak 가능 상태의 release는 금지한다.

모든 산출물은 Git commit, Recovery Package, ProofPack manifest, Handover record 중 하나로 물질화되어야 다음 작업의 기준이 된다.

최종 제품 완료는 2026-08-14 목표로 하되, 핵심 NOT_EXECUTED / NOT_VERIFIED가 0이고 Release Board가 APPROVE일 때만 선언한다.

정식 배포는 최종 완료와 별도이며, backup/restore, rollback, security, rights, legal/brand status, final approval이 모두 증거화된 뒤에만 가능하다.
```

초딩 버전:

```text
지금은 질문창 가게가 열렸다.
다음은 진짜 도서관 책장과 연결한다.
그다음 창고에서 책을 넣고, 대출창구를 만들고, 수업에 쓰고, 손님 반응을 기록한다.
마지막에는 시험지와 증거 파일을 다 모은 뒤에만 “완성”이라고 한다.
배포는 완성보다 더 뒤의 일이다.
```

---

## 2. 현재 베타 상태와 제품 완료 상태 분리

| 구분 | 현재 상태 | 판정 |
|---|---|---|
| Skillup 입출력 UI 초간단 버전 | 완료 with limits | `READY_WITH_LIMITS` |
| Bridge selected/local submit path | 제한 증거 있음 | `CLOSED_WITH_LIMITS` |
| safe_short_answer 구조 | 커밋됨 | `COMMITTED_WITH_LIMITS` |
| admin background call guard | 커밋됨 | `COMMITTED_WITH_LIMITS` |
| 퀄리도서관 최소 evidence 연동 | 미실행 | `NOT_EXECUTED` |
| 퀄리창고 → 도서관 → Skillup 전체 E2E | 미실행 | `NOT_EXECUTED` |
| Analytics/Feedback governance | 미실행 또는 부분 | `NOT_VERIFIED` |
| Integrated UI Shell | 미완료 | `NOT_EXECUTED` |
| Release/deploy/prod readiness | 승인 없음 | `NOT_GRANTED` |

현재 베타는 “전체 제품 완료”가 아니라 다음 세로 절단면이다.

```text
Skillup 질문 입력
→ Bridge/route submit
→ safe_short_answer 또는 HOLD 안내
→ 사용자 화면 표시
→ 피드백 버튼
→ raw leak 방어
```

---

## 3. Phase 0~9 최종 작업 흐름

### Phase 0 — 현재 베타 세로 절단면 닫기

목표:

```text
스킬업 질문 → Bridge → safe_short_answer → UI 표시 → 피드백 버튼
```

완료 조건:

```text
[ ] “솔더링이란?” 질문에 safe_short_answer 표시
[ ] raw JSON 미노출
[ ] 관리자 API 401 소음 없음
[ ] token/key/path/secret 미노출
[ ] 피드백 버튼 작동
[ ] 결과는 PASS_WITH_LIMITS로만 표기
```

### Phase 1 — 퀄리도서관 최소 연동

목표:

```text
로컬 낱말카드가 아니라 Library Core의 승인된 Evidence/Reference에서 safe_short_answer를 가져온다.
```

완료 조건:

```text
[ ] Library seed evidence 존재
[ ] evidence_id 존재
[ ] bridge_trace_id 존재
[ ] Bridge.retrieve_evidence 계약 통과
[ ] Skillup 직접 DB 조회 없음
[ ] 원문 장문 노출 없음
```

### Phase 2 — 퀄리창고 → 도서관 승격 루프

목표:

```text
Warehouse 후보 자료가 review/approval/dry-run/promotion trace를 거쳐 Library Core로 승격된다.
```

완료 조건:

```text
[ ] warehouse_item 생성
[ ] raw_hash / provenance / rights / sensitivity 확인
[ ] review / approval 기록
[ ] promotion dry-run
[ ] promotion trace
[ ] Library card / evidence pointer 생성
[ ] ProofPack 봉인
```

### Phase 3 — Bridge 정식 계약화

목표:

```text
Skillup이 Library/Warehouse를 직접 보지 않고 Bridge 계약으로만 답변한다.
```

완료 조건:

```text
[ ] Bridge.retrieve_evidence
[ ] Bridge.check_policy
[ ] Bridge.explain_trace
[ ] evidence 없는 답변 HOLD
[ ] raw standard text 요청 차단
[ ] trace 설명 가능
```

### Phase 4 — Skillup Education MVP 완성

목표:

```text
질문창이 아니라 실제 교육 수업 흐름으로 확장한다.
```

완료 조건:

```text
[ ] course_library_binding
[ ] module_manifest
[ ] course_id / module_id / cohort_id
[ ] 학생/강사/검토자/관리자 역할 분리
[ ] answer/HOLD/review flow
[ ] feedback queue
```

### Phase 5 — Feedback Queue + Analytics Governance

목표:

```text
사용자 질문, HOLD, 헷갈림, 틀림, 검토 필요를 다시 Warehouse 후보로 돌린다.
```

완료 조건:

```text
[ ] raw prompt 저장 금지
[ ] query_hash / query_summary 저장
[ ] consent 반영
[ ] feedback_queue_item 생성
[ ] 중복 feedback dedup
[ ] Analytics는 정본 직접 수정 금지
```

### Phase 6 — Integrated UI Shell

목표:

```text
Warehouse, Library, Bridge, Skillup, Analytics, Release Board 상태를 한 화면에서 본다.
```

완료 조건:

```text
[ ] Warehouse 상태
[ ] Library evidence/index 상태
[ ] Bridge health
[ ] Skillup answer/HOLD 상태
[ ] Feedback queue
[ ] Release Board
[ ] 위험 알림
[ ] raw/path/secret 미노출
```

### Phase 7 — Full E2E + Operations

목표:

```text
Warehouse → Library → Bridge → Skillup → Feedback → Analytics 전체 흐름을 실제로 검증한다.
```

완료 조건:

```text
[ ] 전체 E2E PASS
[ ] raw leak 0
[ ] evidence_id 누락 0
[ ] bridge_trace_id 누락 0
[ ] backup/restore dry-run PASS
[ ] rollback plan PASS
[ ] incident drill PASS
```

### Phase 8 — RC Freeze

목표:

```text
기능 추가 중지. P0/P1 결함과 증거 보강만 허용.
```

완료 조건:

```text
[ ] 신규 기능 추가 금지
[ ] P0/P1 0건
[ ] Release Board draft
[ ] ProofPack draft
[ ] rollback/handover 완성
```

### Phase 9 — Final Completion Declaration

목표:

```text
제품 개발 완료를 증거 기반으로 선언한다.
```

완료 조건:

```text
[ ] Warehouse gates remain PASS
[ ] Library evidence/index PASS
[ ] Bridge contract/policy/trace PASS
[ ] Skillup module/binding/answer/HOLD PASS
[ ] Analytics consent/raw prompt scan/feedback loop PASS
[ ] Integrated UI no leak PASS
[ ] Backup/restore dry-run PASS
[ ] Rollback plan PASS
[ ] Release Board APPROVE
[ ] ProofPack missing count = 0
[ ] Critical NOT_EXECUTED / NOT_VERIFIED = 0
```

---

## 4. Beta / Final / Deploy 판정 분리

| 판정 | 의미 | 허용 조건 |
|---|---|---|
| `BETA_MINIMAL_READY_WITH_LIMITS` | 제한된 베타 질문창 사용 가능 | UI, submit, safe answer/status, no leak, feedback 확인 |
| `TRACK_A_PASS` | Track A 목표 전체 통과 | Bridge/Skillup/F13 gates 별도 증거 필요 |
| `FINAL_COMPLETION_APPROVE` | 제품 개발 완료 선언 | Phase 0~9 필수 Gate와 Release Board APPROVE |
| `DEPLOY_APPROVE` | 운영/정식 배포 승인 | final completion + deployment guardrails 모두 충족 |

금지:

```text
BETA_MINIMAL_READY_WITH_LIMITS를 BETA_PASS로 확장 금지
Track A 일부 evidence를 F13_PASS로 확장 금지
UI 표시를 제품 완료로 확장 금지
commit 성공을 release readiness로 확장 금지
```

---

## 5. 배포 금지 조건

```text
DEPLOY_FORBIDDEN_UNTIL:
- Release Board APPROVE
- ProofPack missing count = 0
- rollback plan exists
- backup/restore dry-run PASS
- raw leak 0
- secret/internal path leak 0
- DB migration proof exists if DB changed
- legal/trademark/canonical brand status explicitly recorded
- final approver recorded
```

배포 전 필수 증거:

| 영역 | 증거 |
|---|---|
| Release | release_board_REL-*.md/json |
| ProofPack | manifest + hashes |
| Rollback | rollback_plan.md + dry-run result |
| Backup/restore | backup manifest + restore dry-run PASS |
| Security | raw leak, secret, internal path, token scan |
| Rights | paid standard raw export count 0 |
| DB | migration plan/dry-run/rollback if DB changed |
| Brand/legal | candidate/canonical/legal/trademark state recorded |
| Approval | final approver, scope, timestamp |

---

## 6. No broad PASS rule

```text
NO_BROAD_PASS_WITHOUT_PROOFPACK=true
NO_BETA_PASS_WITHOUT_BETA_RELEASE_BOARD=true
NO_TRACK_A_PASS_WITHOUT_TRACK_A_GATE_RESULTS=true
NO_F13_PASS_WITHOUT_F13_PROOFPACK=true
NO_RELEASE_READY_WITHOUT_RELEASE_BOARD_APPROVE=true
NO_PRODUCTION_READY_WITHOUT_DEPLOYMENT_APPROVAL=true
```

상태 표현:

| 상태 | 의미 |
|---|---|
| `PASS` | 실행했고 증거가 있음 |
| `PASS_WITH_LIMITS` | 제한 범위와 한계가 명시됨 |
| `READY_WITH_LIMITS` | 제한 조건으로 다음 단계 진행 가능 |
| `COMMITTED_WITH_LIMITS` | 코드/문서가 commit 되었으나 runtime/live 검증 분리 |
| `NOT_EXECUTED` | 실행하지 않음 |
| `NOT_VERIFIED` | 검증하지 않음 |
| `NOT_GRANTED` | 승인/판정 권한 없음 |
| `HOLD` | 안전 중지 |
| `REVIEW_REQUIRED` | 사람 검토 필요 |
| `REJECT` | 중대 기준 불충족 |

---

## 7. Codex 작업 패킷 규칙

향후 Codex 작업은 아래 규칙을 따른다.

```text
1. COMMON → PROJECT MEMORY → POST_R9ZNW259 ADDENDUM → AGENTS 순서로 읽는다.
2. repository state gate를 먼저 실행한다.
3. untracked docs/brand는 계속 CANDIDATE_OUT_OF_SCOPE_PRESERVE_NOT_CANONICAL로 보존한다.
4. 같은 risk class의 review-only loop를 만들지 않는다.
5. runtime, HTTP, browser, DB, external provider, deploy는 별도 승인으로 분리한다.
6. 완료 보고서는 H:\장기기억\docs\codex\2026\07\에 저장한다.
7. PASS는 ProofPack 또는 completion report 증거 안에서만 주장한다.
```

---

## 8. 최종 작업 카드 목록

| 순서 | 작업 카드 | 목적 | 상태 |
|---:|---|---|---|
| 0 | Human beta submit review / safe glossary or seed validation | 현재 세로 절단면 닫기 | NEXT |
| 1 | Library minimal evidence seed integration | safe answer를 Library evidence 기반으로 전환 | TODO |
| 2 | Warehouse promotion reuse gate | 창고 후보 → Library 승격 재사용 | TODO |
| 3 | Bridge contract proofpack gate | Evidence/Trace/Policy 정식화 | TODO |
| 4 | Skillup course/module/binding MVP | 교육 모듈화 | TODO |
| 5 | Feedback Queue + Analytics governance | 피드백 환류 | TODO |
| 6 | Integrated UI Shell | 운영 콘솔 | TODO |
| 7 | Full E2E + Operations proofpack | 전체 흐름 검증 | TODO |
| 8 | RC Freeze | 기능 동결 | TODO |
| 9 | Final Completion Declaration | 제품 완료 선언 | TODO |
| 10 | Production Deployment Gate | 정식 배포 | TODO, separate approval |

---

## 9. Release Board 최종 승인 체크리스트

```text
[ ] Warehouse gates PASS
[ ] Library evidence/index PASS
[ ] Bridge contract/policy/trace PASS
[ ] Skillup module/binding/answer/HOLD PASS
[ ] Feedback loop PASS
[ ] Analytics privacy/consent/raw prompt scan PASS
[ ] Integrated UI no leak PASS
[ ] Raw leak 0
[ ] Secret/internal path leak 0
[ ] Evidence missing count 0
[ ] Bridge trace missing count 0
[ ] ProofPack missing count 0
[ ] Backup/restore dry-run PASS
[ ] Rollback plan PASS
[ ] Incident drill PASS
[ ] Legal/brand/canonical status recorded
[ ] Final approver recorded
```

---

## 10. Final recommendation

```text
FINAL_RECOMMENDATION=APPROVE_AS_FINAL_WORKFLOW_AND_RELEASE_GUARDRAIL_ADDENDUM_CANDIDATE
NEXT_ACTION=PLACE_IN_REPO_AND_LONG_MEMORY_WITH_EXPLICIT_COMMIT_GATE
DO_NOT_EDIT_ORIGINAL_SCHEDULE_OR_GUIDEBOOK_YET=true
```

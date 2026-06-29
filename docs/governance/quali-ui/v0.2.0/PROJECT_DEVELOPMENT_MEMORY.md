# PROJECT_DEVELOPMENT_MEMORY.md

> 파일명: `PROJECT_DEVELOPMENT_MEMORY.md`  
> 프로젝트: Quali UI System  
> 버전: v0.2.0  
> 상태: Active  
> 기준일: 2026-06-29  
> 목적: 퀄리 UI의 결정·예외·HOLD·컴포넌트·토큰 기록 장부

---

## 1. Memory 운영 원칙

이 문서는 프로젝트의 판례집이다.  
새로운 결정을 반복 적용해야 할 때 이 문서에 기록한다.

Memory는 헌법과 가드레일을 대체하지 않는다.  
헌법·가드레일과 충돌하면 Memory 항목이 수정 대상이다.

---

## 2. Memory 기록 상태 값

| 상태 | 의미 |
|---|---|
| ACTIVE | 현재 적용 중 |
| HOLD | 확인 필요 |
| DEPRECATED | 더 이상 권장하지 않음 |
| SUPERSEDED | 새 결정으로 대체됨 |
| ARCHIVED | 보관용 기록 |

### 2.1 공식 UI 상태 값

| 상태 | 의미 | 사용 조건 |
|---|---|---|
| DRAFT | 초안 | 검토 전 산출물 |
| INFO | 참고 정보 | 판정이 아닌 보조 정보 |
| HOLD | 확인 필요 | 이유와 다음 조치 필수 |
| FAIL | 기준 미달 또는 차단 | 복구 조건 필수 |
| REVIEWED | 검토 완료 | READY 또는 PASS를 대체하지 않음 |
| READY | 제출 또는 사용 가능 | Output Gate 통과 필요 |
| PASS | 기준 충족 | 근거 연결 필수 |
| ARCHIVED | 보관 | 비활성 기록 |

---

## 3. 공식 제품명 결정

| 결정 ID | 날짜 | 결정 | 이유 | 상태 |
|---|---|---|---|---|
| QMEM-NAME-001 | 2026-06-29 | `퀄리창고`를 공식 제품명으로 고정 | 원본 자료 저장·버전 추적 역할을 명확히 함 | ACTIVE |
| QMEM-NAME-002 | 2026-06-29 | `퀄리도서관`을 공식 제품명으로 고정 | 표준 지식화·교육·용어집 역할을 명확히 함 | ACTIVE |
| QMEM-NAME-003 | 2026-06-29 | `브릿지`를 공식 제품명으로 고정 | 고객 소통과 표준 지식 연결 역할을 명확히 함 | ACTIVE |
| QMEM-NAME-004 | 2026-06-29 | `스킬업`을 공식 교육 모듈명으로 사용 | 교육·훈련·평가 흐름과 연결 | ACTIVE |

영문 보조명은 코드·API·국제 자료에서만 사용한다.

---

## 4. 핵심 결정 로그

| 결정 ID | 날짜 | 영역 | 결정 | 이유 | 상태 |
|---|---|---|---|---|---|
| QMEM-001 | 2026-06-29 | Identity | 퀄리 UI는 표준·품질·교육·검증 중심 고신뢰 작업대 UI | 일반 SaaS 화면과 구분 | ACTIVE |
| QMEM-002 | 2026-06-29 | Flow | 모든 화면은 입력 → 기준 → 판정 → 근거 → 조치 흐름 적용 | 추적성 확보 | ACTIVE |
| QMEM-003 | 2026-06-29 | Color | Navy/Blue/Gray/White + PASS/HOLD/FAIL로 제한 | 의미 중심 색상 체계 | ACTIVE |
| QMEM-004 | 2026-06-29 | Typography | Noto Sans KR + ui-monospace 조합 고정 | 한국어 가독성, 표준번호 식별 | ACTIVE |
| QMEM-005 | 2026-06-29 | Layout | 표준·조항·판정·근거는 테이블·패널 우선 | 문서성·비교성 강화 | ACTIVE |
| QMEM-006 | 2026-06-29 | Accessibility | 상태는 색상+텍스트 동시 표시 | 색상 단독 전달 금지 | ACTIVE |
| QMEM-007 | 2026-06-29 | Modularity | 모든 제품은 같은 토큰과 컴포넌트 공유 | 시리즈 정합성 확보 | ACTIVE |
| QMEM-008 | 2026-06-29 | Components | 13개 핵심 컴포넌트 우선 채택 | 반복 개발 단축 | ACTIVE |
| QMEM-009 | 2026-06-29 | Workflow | 새 재사용 결정은 Memory 기록 | 드리프트 방지 | ACTIVE |
| QMEM-010 | 2026-06-29 | Naming | `ipc610`은 검색어, 화면 표기는 `IPC-A-610` 권장 | 표준명 정합성 | ACTIVE |
| QMEM-011 | 2026-06-29 | Status | `REVIEWED`를 공식 UI 상태 값으로 승격 | 검토 완료와 제출 가능 상태를 구분 | ACTIVE |
| QMEM-012 | 2026-06-29 | Governance | 문서 변경 후 검증과 `SHA256SUMS.txt` 갱신을 의무화 | 무결성 드리프트 방지 | ACTIVE |
| QMEM-013 | 2026-06-29 | Evidence | Evidence Schema 필수 필드 채택 | 근거 재사용성과 검증 가능성 확보 | ACTIVE |
| QMEM-014 | 2026-06-29 | Typography | 한글 대형 제목 자간은 `letter-spacing: 0`을 기본값으로 사용 | 과도한 자간 압축으로 인한 가독성 저하 방지 | ACTIVE |

---

## 5. 승인된 디자인 토큰

### 색상

| 토큰 | 값 | 용도 | 상태 |
|---|---:|---|---|
| `--q-navy-900` | `#061F4A` | 헤더, 신뢰 배너 | ACTIVE |
| `--q-blue-700` | `#0B3D91` | 링크, 표준 강조 | ACTIVE |
| `--q-blue-500` | `#105BD8` | 주요 액션 | ACTIVE |
| `--q-text-900` | `#212121` | 본문 | ACTIVE |
| `--q-text-700` | `#323A45` | 보조 제목 | ACTIVE |
| `--q-text-500` | `#64748B` | 보조 설명 | ACTIVE |
| `--q-bg` | `#F8FAFC` | 앱 배경 | ACTIVE |
| `--q-panel` | `#FFFFFF` | 패널 | ACTIVE |
| `--q-line` | `#D9E2EC` | 경계선 | ACTIVE |
| `--q-pass` | `#15803D` | PASS | ACTIVE |
| `--q-hold` | `#92400E` | HOLD | ACTIVE |
| `--q-fail` | `#B91C1C` | FAIL | ACTIVE |
| `--q-focus` | `#105BD8` | 포커스 링 | ACTIVE |

### 폰트

| 토큰 | 값 | 상태 |
|---|---|---|
| `--q-font-main` | `"Noto Sans KR", system-ui, sans-serif` | ACTIVE |
| `--q-font-code` | `ui-monospace, Menlo, Consolas, monospace` | ACTIVE |

---

## 6. 승인된 공통 컴포넌트

| 컴포넌트 ID | 이름 | 목적 | 상태 |
|---|---|---|---|
| QCMP-001 | Trust Banner | 공식성·신뢰성 표시 | ACTIVE |
| QCMP-002 | Standard Context Bar | 표준번호·개정·상태 표시 | ACTIVE |
| QCMP-003 | Standard Code Badge | 표준번호·개정·Class 짧은 식별 | ACTIVE |
| QCMP-004 | Workbench Search | 검색·필터 | ACTIVE |
| QCMP-005 | Module Card | 제품·모듈 진입 | ACTIVE |
| QCMP-006 | Status Badge | 상태 표시 | ACTIVE |
| QCMP-007 | Evidence Box | 근거 표시 | ACTIVE |
| QCMP-008 | Trace Table | 입력→기준→판정→근거→조치 표시 | ACTIVE |
| QCMP-009 | Beginner Box | 초보자 설명 | ACTIVE |
| QCMP-010 | Expert Box | 전문가 근거 | ACTIVE |
| QCMP-011 | Action Checklist | 다음 행동 | ACTIVE |
| QCMP-012 | Output Gate | 출력 전 검사 | ACTIVE |
| QCMP-013 | Line Break Gate | 한글 제목·핵심 문장 행갈이 검사 | ACTIVE |

---

## 6.1 승인된 운영 산출물

| 산출물 ID | 경로 | 목적 | 상태 |
|---|---|---|---|
| QOPS-001 | `schemas/tokens.schema.json` | 승인 토큰 기준 | ACTIVE |
| QOPS-002 | `schemas/status.schema.json` | 공식 UI 상태 값 기준 | ACTIVE |
| QOPS-003 | `schemas/evidence.schema.json` | Evidence 필수 필드 기준 | ACTIVE |
| QOPS-004 | `schemas/component-contract.schema.json` | 공통 컴포넌트 계약 기준 | ACTIVE |
| QOPS-005 | `scripts/check-hash.mjs` | 해시 무결성 검사 | ACTIVE |
| QOPS-006 | `scripts/check-doc-sync.mjs` | 문서 정합성 검사 | ACTIVE |
| QOPS-007 | `templates/preflight.md` | 작업 전 확인 양식 | ACTIVE |
| QOPS-008 | `templates/verification-report.md` | 검증 보고 양식 | ACTIVE |

---

## 7. 금지 패턴 기록

| 패턴 ID | 날짜 | 금지 내용 | 이유 | 상태 |
|---|---|---|---|---|
| QBAN-001 | 2026-06-29 | 과한 그라데이션 | 표준형 신뢰 저하 | ACTIVE |
| QBAN-002 | 2026-06-29 | 이모지 중심 네비게이션 | 업무형 UI 부적합 | ACTIVE |
| QBAN-003 | 2026-06-29 | 색상만으로 상태 표현 | 접근성 실패 | ACTIVE |
| QBAN-004 | 2026-06-29 | 표준번호 없는 판정 카드 | 근거 추적 불가 | ACTIVE |
| QBAN-005 | 2026-06-29 | 중복 컴포넌트 생성 | 유지보수 비용 증가 | ACTIVE |
| QBAN-006 | 2026-06-29 | 페이지별 임의 HEX | 시리즈 정합성 붕괴 | ACTIVE |

---

## 8. 현재 HOLD 항목

| HOLD ID | 날짜 | 항목 | 설명 | 소유자 | 해소 조건 | 목표일 | 차단 수준 | 상태 |
|---|---|---|---|---|---|---|---|---|
| QHOLD-001 | 2026-06-29 | 로고 시스템 | 단색/역상/아이콘 버전 확정 필요 | TBD | 로고 사용 규칙 승인 | TBD | Medium | HOLD |
| QHOLD-002 | 2026-06-29 | 배포 플랫폼 | 정적 HTML, 웹앱, 문서 출력 경로 확정 필요 | TBD | 1차 배포 경로 결정 | TBD | Medium | HOLD |
| QHOLD-003 | 2026-06-29 | CI 도구 | GitHub Actions 등 확정 필요 | TBD | 자동 검증 실행 위치 결정 | TBD | Medium | HOLD |
| QHOLD-004 | 2026-06-29 | 표준 메타데이터 스키마 상세 | version, revision, status, evidence_level 최종 확정 필요 | TBD | Evidence Schema와 메타데이터 스키마 연결 승인 | TBD | High | HOLD |
| QHOLD-005 | 2026-06-29 | 고객별 브랜드 정책 | 고객사 로고/색상 허용 범위 확정 필요 | TBD | 고객별 로고·색상 예외 정책 승인 | TBD | Medium | HOLD |
| QHOLD-006 | 2026-06-29 | Playwright 시각 검증 | Line Break Gate의 실제 렌더링 검증 경로 확정 필요 | TBD | 데스크톱·모바일 스크린샷 기준 승인 | TBD | Low | HOLD |

공식 제품명은 HOLD가 아니다. `퀄리창고`, `퀄리도서관`, `브릿지`는 고정이다.

---

## 9. 새 Memory 기록 양식

```text
결정 ID:
날짜:
영역:
결정:
이유:
영향 범위:
소유자:
해소 조건:
목표일:
차단 수준:
상태:
관련 파일:
```

---

## 10. 변경 이력

### v0.2.0 — 2026-06-29

- 공식 UI 상태 값 표 추가
- `REVIEWED` 공식 상태 승격 결정 기록
- 검증·해시·Evidence Schema 운영 결정 기록
- `--q-focus` 토큰과 Line Break Gate 컴포넌트 기록 추가
- HOLD 항목에 소유자, 해소 조건, 목표일, 차단 수준 필드 추가
- 승인된 운영 산출물 목록 추가

### v0.1.0 — 2026-06-29

- 초기 Memory 생성
- 공식 제품명 고정 반영
- 핵심 결정 10개 등록
- 토큰, 컴포넌트, 금지 패턴, HOLD 항목 등록

---

## HOLD 항목 관리 스키마 v0.2.0

v0.2.0부터 모든 HOLD 항목은 단순 보류 메모가 아니라 해소 가능한 관리 항목으로 기록한다.

| 필드 | 의미 | 필수 여부 |
|---|---|---|
| HOLD ID | HOLD 식별자 | 필수 |
| 항목 | 보류 항목명 | 필수 |
| owner / 소유자 | 해소 책임자 또는 담당 역할 | 필수 |
| resolution_condition / 해소 조건 | HOLD를 닫기 위한 조건 | 필수 |
| target_date / 목표일 | 해소 목표일 또는 다음 검토일 | 권장 |
| blocking_level / 차단 수준 | NONE, LOW, MEDIUM, HIGH, BLOCKER | 필수 |
| status / 상태 | HOLD, REVIEWED, READY, ARCHIVED | 필수 |

### HOLD 기록 양식

```text
HOLD ID:
항목:
owner / 소유자:
resolution_condition / 해소 조건:
target_date / 목표일:
blocking_level / 차단 수준:
status / 상태:
관련 파일:
메모:


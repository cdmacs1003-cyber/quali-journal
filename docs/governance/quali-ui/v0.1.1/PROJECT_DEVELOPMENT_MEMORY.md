# PROJECT_DEVELOPMENT_MEMORY.md

> 파일명: `PROJECT_DEVELOPMENT_MEMORY.md`  
> 프로젝트: Quali UI System  
> 버전: v0.1.1  
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

## 2. 상태 값

| 상태 | 의미 |
|---|---|
| ACTIVE | 현재 적용 중 |
| HOLD | 확인 필요 |
| DEPRECATED | 더 이상 권장하지 않음 |
| SUPERSEDED | 새 결정으로 대체됨 |
| ARCHIVED | 보관용 기록 |

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
| QMEM-008 | 2026-06-29 | Components | 10개 핵심 컴포넌트 우선 채택 | 반복 개발 단축 | ACTIVE |
| QMEM-009 | 2026-06-29 | Workflow | 새 재사용 결정은 Memory 기록 | 드리프트 방지 | ACTIVE |
| QMEM-010 | 2026-06-29 | Naming | `ipc610`은 검색어, 화면 표기는 `IPC-A-610` 권장 | 표준명 정합성 | ACTIVE |

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
| QCMP-003 | Workbench Search | 검색·필터 | ACTIVE |
| QCMP-004 | Module Card | 제품·모듈 진입 | ACTIVE |
| QCMP-005 | Status Badge | 상태 표시 | ACTIVE |
| QCMP-006 | Evidence Box | 근거 표시 | ACTIVE |
| QCMP-007 | Trace Table | 입력→기준→판정→근거→조치 표시 | ACTIVE |
| QCMP-008 | Beginner Box | 초보자 설명 | ACTIVE |
| QCMP-009 | Expert Box | 전문가 근거 | ACTIVE |
| QCMP-010 | Output Gate | 출력 전 검사 | ACTIVE |
| QCMP-011 | Action Checklist | 다음 행동 | ACTIVE |

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

| HOLD ID | 날짜 | 항목 | 설명 | 상태 |
|---|---|---|---|---|
| QHOLD-001 | 2026-06-29 | 로고 시스템 | 단색/역상/아이콘 버전 확정 필요 | HOLD |
| QHOLD-002 | 2026-06-29 | 배포 플랫폼 | 정적 HTML, 웹앱, 문서 출력 경로 확정 필요 | HOLD |
| QHOLD-003 | 2026-06-29 | CI 도구 | GitHub Actions 등 확정 필요 | HOLD |
| QHOLD-004 | 2026-06-29 | 표준 메타데이터 스키마 상세 | version, revision, status, evidence_level 최종 확정 필요 | HOLD |
| QHOLD-005 | 2026-06-29 | 고객별 브랜드 정책 | 고객사 로고/색상 허용 범위 확정 필요 | HOLD |

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
상태:
관련 파일:
```

---

## 10. 변경 이력

### v0.1.0 — 2026-06-29

- 초기 Memory 생성
- 공식 제품명 고정 반영
- 핵심 결정 10개 등록
- 토큰, 컴포넌트, 금지 패턴, HOLD 항목 등록

# Quali UI Governance Advancement Strategy

> 파일명: `QUALI_UI_GOVERNANCE_ADVANCEMENT_STRATEGY.md`  
> 프로젝트: Quali UI System  
> 기준일: 2026-06-29  
> 목적: Quali UI 거버넌스 팩의 고도화 전략 제안
> 적용 상태: v0.2.0 문서 정합성 패치 반영

---

## 1. Executive Summary

현재 Quali UI 거버넌스 팩은 방향성이 좋은 v0.1 단계다.

헌법, 가드레일, AGENTS, 워크플로, 메모리, 가이드북, Skill이 모두 같은 핵심 방향을 공유한다.

핵심 방향은 다음과 같다.

```text
입력 → 기준 → 판정 → 근거 → 조치
```

이 흐름은 Quali UI를 일반적인 SaaS 화면이 아니라, 표준·품질·교육·검증·근거 추적을 위한 고신뢰 작업대 UI로 구분하게 만든다.

다음 고도화 목표는 문서를 더 많이 만드는 것이 아니라, 문서가 실제 작업을 통제하고 검증하게 만드는 것이다.

```text
v0.2: 문서 정합성
v0.3: 자동 검증
v1.0: 어떤 에이전트가 작업해도 같은 품질로 나오는 Quali UI 운영체계
```

---

## 2. Current Strengths

### 2.1 명확한 최상위 정체성

`00_QUALI_UI_CONSTITUTION.md`는 Quali UI의 정체성을 명확하게 정의한다.

Quali UI는 보기 좋은 화면보다 신뢰 가능한 화면을 우선한다.

핵심 가치도 명확하다.

- 쉬움
- 표준성
- 신뢰성
- 전문성
- 간결성
- 여백
- 모듈성
- 추적성
- 접근성
- 재사용성

### 2.2 강한 정보 흐름

모든 주요 화면과 산출물이 아래 흐름을 따르도록 되어 있다.

```text
입력 → 기준 → 판정 → 근거 → 조치
```

이 흐름은 제품, 화면, 보고서, 교육 UI, 고객 응대 UI에 모두 적용할 수 있는 좋은 뼈대다.

### 2.3 초보자와 전문가를 동시에 고려

`Beginner Box`와 `Expert Box / Evidence Box`를 분리한 구조는 매우 좋다.

초보자는 바로 이해하고, 전문가는 표준번호·개정·조항·근거를 확인할 수 있다.

### 2.4 실행자용 압축본이 존재

`.agents/skills/quali-standard-ui/SKILL.md`는 Codex나 다른 에이전트가 Quali UI 작업을 시작할 때 읽기 좋은 실행 압축본 역할을 한다.

### 2.5 Memory가 판례집 역할을 시작함

`PROJECT_DEVELOPMENT_MEMORY.md`는 공식 제품명, 토큰, 컴포넌트, 금지 패턴, HOLD 항목을 기록하고 있다.

이는 향후 의사결정 드리프트를 막는 좋은 기반이다.

---

## 3. Key Risks

### 3.1 문서 간 상태 값 충돌

`REVIEWED` 상태가 Guardrails와 Guidebook에는 존재하지만 Constitution의 허용 상태 목록에는 없다.

헌법 우선순위상 이는 정리해야 하는 충돌이다.

결정이 필요하다.

```text
선택 A: REVIEWED를 헌법의 공식 상태로 승격
선택 B: REVIEWED를 READY 또는 INFO로 흡수
선택 C: REVIEWED를 내부 검토 메타데이터로만 사용
```

### 3.2 SHA256SUMS 불일치

초기 분석 당시 `SHA256SUMS.txt` 기준으로 일부 파일의 해시가 실제 파일과 맞지 않았다.

확인된 불일치 파일은 다음과 같다.

- `00_QUALI_UI_GUARDRAILS.md`
- `PROJECT_DEVELOPMENT_GUIDEBOOK.md`
- `.agents/skills/quali-standard-ui/SKILL.md`

이는 문서가 변경되었지만 무결성 기록이 갱신되지 않았다는 뜻이다. v0.2.0 패치에서는 해시 갱신과 `check:hash` 검증을 운영 절차에 포함한다.

### 3.3 자동 검증 부재

`COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`에는 토큰 검사와 대비 검사 스크립트 예시가 있지만, 실제 실행 가능한 `scripts/`와 `package.json` 게이트는 아직 없다.

현재 구조는 사람이 체크리스트를 성실히 읽는다는 전제에 많이 의존한다.

### 3.4 HOLD 운영 정보 부족

현재 HOLD 항목은 있으나 다음 정보가 부족하다.

- 소유자
- 해소 조건
- 목표일
- 차단 수준
- 관련 결정 ID

HOLD가 운영 대기열이 되려면 이 필드들이 필요하다.

### 3.5 Single Source of Truth 부족

토큰, 상태, 컴포넌트, 출력 게이트, Line Break Gate가 여러 문서에 반복된다.

반복 자체는 상기 장치로 유용하지만, 기준 데이터가 여러 곳에 흩어지면 장기적으로 드리프트가 생길 수 있다.

---

## 4. Advancement Strategy

## 4.1 v0.2 목표: 문서 정합성

v0.2의 목표는 모든 문서가 같은 기준을 말하도록 정리하는 것이다.

### 핵심 작업

1. 상태 값 정합성 정리
2. 토큰 목록 정합성 정리
3. 컴포넌트 목록 정합성 정리
4. HOLD 항목 운영 필드 확장
5. SHA256SUMS 갱신 절차 정의
6. 문서 우선순위와 변경 절차 명문화

### 권장 산출물

```text
docs-sync-report.md
status-decision-record.md
updated SHA256SUMS.txt
expanded PROJECT_DEVELOPMENT_MEMORY.md
```

### v0.2 완료 조건

- Constitution, Guardrails, Guidebook, Memory, Skill의 상태 값이 일치한다.
- 승인 토큰 목록이 문서 간 충돌하지 않는다.
- 승인 컴포넌트 목록이 문서 간 충돌하지 않는다.
- 모든 HOLD 항목에 소유자와 해소 조건이 있다.
- SHA256SUMS 검증이 PASS한다.

---

## 4.2 v0.3 목표: 자동 검증

v0.3의 목표는 문서 기준을 실제 검사로 내리는 것이다.

### 핵심 작업

아래 검사를 실제 스크립트로 만든다.

```text
check:hash
check:tokens
check:contrast
check:a11y
check:korean-linebreak
check:doc-sync
check:evidence-schema
```

### 권장 디렉터리

```text
scripts/
  check-hash.mjs
  check-token-usage.mjs
  check-contrast.mjs
  check-doc-sync.mjs
  check-korean-linebreak.mjs
  check-evidence-schema.mjs

schemas/
  tokens.schema.json
  status.schema.json
  evidence.schema.json
  component-contract.schema.json

templates/
  preflight.md
  verification-report.md
  memory-entry.md
  output-gate.md
```

### v0.3 완료 조건

- `npm test` 또는 동일한 단일 명령으로 기본 게이트가 실행된다.
- 임의 HEX 색상 추가가 자동으로 잡힌다.
- 상태 값 불일치가 자동으로 잡힌다.
- Evidence 필수 필드 누락이 자동으로 잡힌다.
- SHA256SUMS 불일치가 자동으로 잡힌다.
- Line Break Gate 일부가 정적 검사로 잡힌다.

---

## 4.3 v1.0 목표: Quali UI 운영체계

v1.0의 목표는 어떤 사람이나 에이전트가 작업해도 같은 품질이 나오도록 만드는 것이다.

### 핵심 작업

1. 제품별 기준 화면 템플릿 확정
2. 컴포넌트 계약 확정
3. Evidence Schema 확정
4. Output Gate 자동화
5. Line Break Gate 시각 검증
6. Playwright 기반 화면 검증
7. Governance Dashboard 구축

### v1.0 완료 조건

- 퀄리창고, 퀄리도서관, 브릿지, 스킬업 기준 화면 템플릿이 존재한다.
- 모든 템플릿은 승인 토큰과 컴포넌트를 사용한다.
- 모든 주요 산출물은 Evidence Schema를 통과한다.
- 모든 출력물은 Output Gate 결과를 포함한다.
- 주요 화면은 Playwright 스크린샷 검증을 통과한다.
- 새 에이전트가 작업해도 동일한 preflight, verify, final report 형식을 따른다.

---

## 5. Recommended Immediate Fixes

### 5.1 REVIEWED 상태 결정

우선 `REVIEWED`를 공식 상태로 둘지 결정해야 한다.

추천은 공식 상태로 승격하는 것이다.

이유:

- 검토 완료와 제출 가능은 다르다.
- `REVIEWED`와 `READY`는 서로 다른 운영 의미를 갖는다.

권장 상태 체계:

| 상태 | 의미 |
|---|---|
| DRAFT | 초안 |
| INFO | 참고 정보 |
| HOLD | 확인 필요 |
| FAIL | 기준 미달 또는 차단 |
| REVIEWED | 검토 완료 |
| READY | 제출 또는 사용 가능 |
| PASS | 기준 충족 |
| ARCHIVED | 보관 |

### 5.2 SHA256SUMS 갱신

문서 변경 후 반드시 해시를 갱신해야 한다.

권장 규칙:

```text
문서 수정 → 검증 → SHA256SUMS 갱신 → 최종 보고
```

### 5.3 HOLD 표 확장

현재 HOLD 표를 아래처럼 확장한다.

| HOLD ID | 항목 | 설명 | 소유자 | 해소 조건 | 목표일 | 차단 수준 | 상태 |
|---|---|---|---|---|---|---|---|
| QHOLD-001 | 로고 시스템 | 단색/역상/아이콘 버전 확정 필요 | TBD | 로고 사용 규칙 승인 | TBD | Medium | HOLD |

### 5.4 Evidence Schema 추가

Quali UI의 핵심은 근거 추적이다.

따라서 Evidence는 공통 스키마가 필요하다.

권장 필드:

```json
{
  "standard_id": "IPC-A-610",
  "revision": "H",
  "clause": "TBD",
  "source_file": "TBD",
  "page": "TBD",
  "evidence_level": "direct | summarized | inferred | pending",
  "verification_status": "PASS | HOLD | FAIL",
  "limitation": "TBD",
  "next_action": "TBD"
}
```

### 5.5 Skill을 실행 패키지로 확장

현재 Skill은 instruction-first 구조다.

다음 단계에서는 Skill 안에 다음을 추가하는 것이 좋다.

```text
scripts/
templates/
fixtures/
examples/
```

Skill이 단순 지침이 아니라, 반복 가능한 실행 패키지가 되어야 한다.

---

## 6. Governance Model

### 6.1 문서 계층

권장 우선순위는 현재 구조를 유지한다.

```text
Constitution
> Guardrails
> AGENTS
> Guidebook
> Workflow
> Codex Final
> Memory
> Skill
```

단, Skill은 실행 압축본이므로 변경 시 상위 문서와 동기화 검사가 필요하다.

### 6.2 변경 등급

| 등급 | 예시 | 필요 조치 |
|---|---|---|
| PATCH | 오탈자, 설명 보강 | 변경 이력 |
| MINOR | 새 컴포넌트, 새 검사 추가 | Memory 기록, Guidebook 반영 |
| MAJOR | 상태 체계, 제품명, 핵심 흐름 변경 | Constitution 반영, 전체 문서 동기화 |

### 6.3 릴리즈 판정

| 판정 | 의미 |
|---|---|
| PASS | 사용 가능 |
| HOLD | 확인 필요 항목 존재 |
| FAIL | 헌법 또는 가드레일 위반 |

---

## 7. Final Recommendation

Quali UI 거버넌스 팩은 이미 좋은 v0.1 기반을 갖고 있다.

다음 단계는 문서를 더 길게 만드는 것이 아니다.

다음 단계는 문서의 기준을 다음 네 가지로 내리는 것이다.

```text
스키마
템플릿
검증 스크립트
자동 게이트
```

최종 목표는 다음 문장으로 정리할 수 있다.

> Quali UI는 어떤 사람이 만들었는지가 아니라, 어떤 기준을 통과했는지로 신뢰되는 UI 시스템이 되어야 한다.

---

## 8. v0.2.0 적용 기록

2026-06-29 기준으로 아래 항목을 거버넌스 팩에 반영한다.

- `REVIEWED`를 공식 UI 상태 값으로 승격
- HOLD 항목에 소유자, 해소 조건, 목표일, 차단 수준 필드 추가
- Evidence Schema 필수 필드 정의
- 스키마, 검증 스크립트, 반복 템플릿 추가
- 문서 정합성, Evidence Schema, 해시 검증 명령 추가
- 한글 제목 자간 정책을 `letter-spacing: 0`으로 정리
- 문서 변경 후 `SHA256SUMS.txt` 갱신을 완료 조건에 포함

# COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md

> 파일명: `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`  
> 프로젝트: Quali UI System  
> 버전: v0.1.1  
> 상태: Active  
> 기준일: 2026-06-29  
> 목적: Codex 및 자동화 코드 작업자의 최종 실행 절차

---

## 1. 강제 원칙

Codex는 아래 원칙을 따른다.

1. 먼저 읽고, 그 다음 수정한다.
2. 헌법과 가드레일을 코드보다 우선한다.
3. 기존 토큰과 컴포넌트를 우선한다.
4. 새 기능보다 추적성과 검증을 우선한다.
5. 검증하지 않은 PASS를 만들지 않는다.
6. 반복 가능한 결정은 Memory에 기록한다.
7. 공식 제품명은 변경하지 않는다.

---

## 2. 필수 선행 읽기

코드 편집 전 반드시 읽는다.

```text
00_QUALI_UI_CONSTITUTION.md
00_QUALI_UI_GUARDRAILS.md
AGENTS.md
PROJECT_DEVELOPMENT_GUIDEBOOK.md
PROJECT_DEVELOPMENT_MEMORY.md
COMMON_DEVELOPMENT_WORKFLOW.md
```

---

## 3. 편집 전 계획

파일 수정 전 아래 계획을 먼저 작성한다.

```text
작업 목표:
대상 제품:
영향 파일:
재사용 컴포넌트:
새 컴포넌트 필요 여부:
상태 구조:
접근성 리스크:
검증 방법:
Memory 갱신 여부:
HOLD 예상:
```

계획 없이 파일 수정 금지.

---

## 4. 파일 수정 규칙

### 4.1 우선순위

1. 토큰 파일
2. 공통 컴포넌트
3. 페이지 조합 파일
4. 모듈 전용 스타일
5. 신규 파일

### 4.2 금지

- 임의 HEX 하드코딩
- 임의 폰트 추가
- 중복 컴포넌트 생성
- `style=""` 남발
- `button`의 `type` 생략
- `aria-label` 없는 아이콘 버튼
- 색상만으로 상태 표시
- 기존 컴포넌트 무시
- 모바일 순서 미검증

---

## 5. 토큰 검증

### 5.1 승인 색상만 사용

```css
--q-navy-900
--q-blue-700
--q-blue-500
--q-text-900
--q-text-700
--q-text-500
--q-bg
--q-panel
--q-line
--q-pass
--q-hold
--q-fail
--q-focus
```

### 5.2 예시 스크립트: `scripts/check-token-usage.mjs`

```js
import fs from "node:fs";
import path from "node:path";

const roots = ["src", "app", "components", "docs"];
const allowed = new Set([
  "#061F4A", "#0B3D91", "#105BD8", "#212121", "#323A45", "#64748B",
  "#F8FAFC", "#FFFFFF", "#D9E2EC", "#15803D", "#92400E", "#B91C1C"
].map(x => x.toLowerCase()));

function walk(dir, files = []) {
  if (!fs.existsSync(dir)) return files;
  for (const item of fs.readdirSync(dir)) {
    const full = path.join(dir, item);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) walk(full, files);
    else files.push(full);
  }
  return files;
}

const files = roots.flatMap(root => walk(root)).filter(f => /\.(css|scss|html|tsx|jsx|ts|js)$/.test(f));
const hexPattern = /#[0-9a-fA-F]{6}/g;
let failed = false;

for (const file of files) {
  const text = fs.readFileSync(file, "utf8");
  const matches = text.match(hexPattern) || [];
  for (const hex of matches) {
    if (!allowed.has(hex.toLowerCase())) {
      console.error(`[FAIL] Unapproved HEX ${hex} in ${file}`);
      failed = true;
    }
  }
}

if (failed) process.exit(1);
console.log("[PASS] Quali token usage check complete");
```

---

## 6. 대비 검증

### 예시 스크립트: `scripts/check-contrast.mjs`

```js
const pairs = [
  ["#212121", "#FFFFFF", 4.5, "body on panel"],
  ["#323A45", "#FFFFFF", 4.5, "secondary heading on panel"],
  ["#105BD8", "#FFFFFF", 4.5, "action blue text on white"],
  ["#FFFFFF", "#105BD8", 4.5, "button text on action blue"],
  ["#FFFFFF", "#15803D", 4.5, "PASS badge"],
  ["#FFFFFF", "#92400E", 4.5, "HOLD badge"],
  ["#FFFFFF", "#B91C1C", 4.5, "FAIL badge"]
];

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)].map(v => v / 255);
}
function lin(v) { return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }
function lum(hex) { const [r,g,b] = hexToRgb(hex).map(lin); return 0.2126*r + 0.7152*g + 0.0722*b; }
function contrast(a, b) { const x=lum(a), y=lum(b); const [L1,L2]=x>y?[x,y]:[y,x]; return (L1+0.05)/(L2+0.05); }

let failed = false;
for (const [fg, bg, min, label] of pairs) {
  const value = contrast(fg, bg);
  if (value < min) {
    console.error(`[FAIL] ${label}: ${value.toFixed(3)} < ${min}`);
    failed = true;
  } else {
    console.log(`[PASS] ${label}: ${value.toFixed(3)}`);
  }
}
if (failed) process.exit(1);
```

---

## 7. 접근성 체크리스트

- [ ] 모든 form input에 label 존재
- [ ] 아이콘 버튼에 aria-label 존재
- [ ] 상태는 색상과 텍스트로 동시 표시
- [ ] focus-visible 제거 금지
- [ ] 테이블 caption 존재
- [ ] th scope 또는 적절한 header 구조 존재
- [ ] 에러 메시지는 텍스트로 제공
- [ ] 비활성 상태는 시각+시맨틱 모두 반영
- [ ] 모바일에서 콘텐츠 순서가 논리적

---

## 8. 권장 `package.json` 스크립트

프로젝트 도구가 확정되면 조정한다.

```json
{
  "scripts": {
    "lint:md": "markdownlint-cli2 "**/*.md"",
    "lint:css": "stylelint "src/**/*.css"",
    "lint:js": "eslint "src/**/*.{js,ts,jsx,tsx}"",
    "format:check": "prettier --check .",
    "test:tokens": "node scripts/check-token-usage.mjs",
    "test:contrast": "node scripts/check-contrast.mjs",
    "test": "npm run lint:md && npm run format:check && npm run test:tokens && npm run test:contrast"
  }
}
```

---

## 9. 최종 보고 형식

```text
요약:
변경 파일:
재사용 컴포넌트:
새 컴포넌트:
토큰 준수:
접근성:
검증 명령:
검증 결과:
Memory 반영:
HOLD:
다음 행동:
```

---

## 10. FAIL 조건

아래 중 하나라도 해당하면 FAIL 또는 HOLD로 보고한다.

- 헌법 위반
- 가드레일 위반
- 근거 없는 PASS
- 임의 색상 추가
- 접근성 검증 누락
- 모바일 붕괴
- Memory 기록 누락
- 공식 제품명 변경

---

## 11. 변경 이력

### v0.1.0 — 2026-06-29

- Codex 최종 실행 절차 생성
- 토큰·대비 검증 스크립트 예시 추가
- 공식 제품명 고정 반영
- 접근성 체크리스트 추가

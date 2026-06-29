# AGENTS.md

> 파일명: `AGENTS.md`  
> 프로젝트: Quali UI System  
> 버전: v0.1.0  
> 상태: Active  
> 기준일: 2026-06-29  
> 목적: 퀄리 시리즈 작업자와 Codex 실행자의 최상위 실행 지침

---

## 1. Project Identity

This repository belongs to the **Quali UI System**.

Official product names:

- 퀄리창고
- 퀄리도서관
- 브릿지
- 스킬업

The UI must support standards-based work, quality judgment, education, evidence tracking, and customer communication.

Core flow:

```text
Input → Standard → Decision → Evidence → Action
```

Korean working expression:

```text
입력 → 기준 → 판정 → 근거 → 조치
```

---

## 2. Required Reading Order

Before any work, read in this order:

1. `00_QUALI_UI_CONSTITUTION.md`
2. `00_QUALI_UI_GUARDRAILS.md`
3. `PROJECT_DEVELOPMENT_GUIDEBOOK.md`
4. `PROJECT_DEVELOPMENT_MEMORY.md`
5. `COMMON_DEVELOPMENT_WORKFLOW.md`
6. For code or file edits: `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`

If these files conflict, apply this precedence:

```text
Constitution > Guardrails > AGENTS > Guidebook > Workflow > Codex Final > Memory
```

Memory records past decisions. It does not override the Constitution or Guardrails.

---

## 3. Operating Rules

### 3.1 Before implementation

Do not edit files until you can answer:

- What is the target product? `퀄리창고`, `퀄리도서관`, `브릿지`, `스킬업`, or standard module?
- Who is the primary user?
- What is the first action the user must take?
- What standard or rule is used?
- What evidence must be visible?
- What component pattern already exists?
- What verification will prove completion?

### 3.2 During implementation

- Use approved Quali tokens only.
- Reuse existing components first.
- Prefer table/panel for standards, clauses, versions, decisions, and evidence.
- Use cards only for module entry, training paths, command libraries, or summarized choices.
- Do not introduce arbitrary colors, fonts, shadows, or spacing.
- Do not hide evidence behind vague marketing text.
- Do not claim PASS when evidence is incomplete.

### 3.3 After implementation

Report:

```text
Summary:
Changed files:
Reused components:
New components:
Verification result:
Accessibility result:
Memory update:
HOLD items:
Next action:
```

---

## 4. Required UI Components

Use these whenever applicable:

- Trust Banner
- Standard Context Bar
- Workbench Search
- Module Card
- Standard Code Badge
- Status Badge
- Evidence Box
- Trace Table
- Beginner Box
- Expert Box
- Action Checklist
- Output Gate

New components require a Memory record and Guidebook update.

---

## 5. Forbidden Patterns

Never create:

- Decorative UI with no evidence flow
- Excessive gradients
- Emoji-based navigation
- One-off button styles
- One-off status colors
- Evidence-free judgment
- Standard-number-free decision cards
- Color-only status indicators
- Hidden focus states
- Unlabeled search inputs
- Dense screens with no whitespace

---

## 6. Product-Specific Guidance

### 퀄리창고

Purpose: original source storage, metadata, version tracking.

Required UI:
- Search/filter first
- Source table
- Version/status badge
- Evidence metadata

### 퀄리도서관

Purpose: standards knowledge, glossary, educational interpretation.

Required UI:
- Beginner Box
- Expert Box
- Standard Context Bar
- Evidence Box

### 브릿지

Purpose: convert customer questions into standards-based answers and actions.

Required UI:
- Guided question input
- Standard selection
- Decision result
- Evidence summary
- Action Checklist

### 스킬업

Purpose: learning path, training, practice, quiz, evaluation.

Required UI:
- Learning path
- Progress status
- Beginner explanation
- Practice checklist
- Evidence-linked feedback

---

## 7. Quick Commands for Agents

### `/intake`

Summarize the request:

```text
Goal:
Target product:
Primary user:
Input:
Standard or rule:
Output:
Success criteria:
HOLD:
```

### `/precheck`

Before editing:

```text
Target files:
Existing components:
New component needed: yes/no
Token impact:
Accessibility risk:
Verification method:
Memory update needed: yes/no
```

### `/verify`

After editing:

```text
Ease:
Standard visibility:
Evidence visibility:
Action clarity:
Token compliance:
Accessibility:
Responsive behavior:
Final status: PASS/HOLD/FAIL
```

---

## 8. Completion Definition

A task is complete only when:

- The first user action is clear.
- The standard/rule context is visible.
- Decision status is visible.
- Evidence is visible.
- Next action is visible.
- Approved tokens and components are used.
- Accessibility checks are performed or marked HOLD with reason.
- Reusable decisions are recorded in `PROJECT_DEVELOPMENT_MEMORY.md`.

---

## 9. Source Basis

This file follows the Codex guidance pattern where `AGENTS.md` provides project-specific instructions before work starts, and it aligns the project with the Quali UI Constitution and Guardrails.

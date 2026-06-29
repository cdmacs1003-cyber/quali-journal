---
name: quali-standard-ui
description: Use this skill when creating or modifying Quali UI, 퀄리창고, 퀄리도서관, 브릿지, 스킬업, standard-based dashboards, evidence-driven HTML, PASS/HOLD/FAIL panels, standard search UI, education screens, customer bridge UI, or design-token-based components.
---

# Quali Standard UI Skill

> 파일 위치: `.agents/skills/quali-standard-ui/SKILL.md`  
> 버전: v0.1.0  
> 상태: Active  
> 기준일: 2026-06-29

---

## Purpose

Create and modify Quali Series UI using the approved Constitution, Guardrails, design tokens, components, and workflow.

The skill must produce UI that is easy for beginners and credible for experts.

Core flow:

```text
Input → Standard → Decision → Evidence → Action
```

Korean working expression:

```text
입력 → 기준 → 판정 → 근거 → 조치
```

---

## Official Product Names

Use these names exactly.

- 퀄리창고
- 퀄리도서관
- 브릿지
- 스킬업

Do not rename these products.  
English helper names may appear only in code, IDs, API, or international-facing documentation.

---

## When to Use

Use this skill for:

- Quali UI system work
- 퀄리창고 UI
- 퀄리도서관 UI
- 브릿지 UI
- 스킬업 UI
- IPC-A-610 UI
- J-STD-001 UI
- IPC/WHMA-A-620 UI
- ECSS UI
- NASA standards UI
- Evidence-based report UI
- Standard search UI
- PASS/HOLD/FAIL decision UI
- HTML educational guide UI
- Design-token-based refactoring

Do not use this skill for unrelated creative landing pages, entertainment pages, or decorative-only design tasks.

---

## Required Reading

Before starting, read:

1. `00_QUALI_UI_CONSTITUTION.md`
2. `00_QUALI_UI_GUARDRAILS.md`
3. `AGENTS.md`
4. `PROJECT_DEVELOPMENT_GUIDEBOOK.md`
5. `PROJECT_DEVELOPMENT_MEMORY.md`
6. `COMMON_DEVELOPMENT_WORKFLOW.md`

For coding or file edits, also read:

7. `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`

---

## Core Design Rules

1. Show the first action within 10 seconds.
2. Show the standard/rule context.
3. Show the decision status.
4. Show the evidence.
5. Show the next action.
6. Use approved tokens only.
7. Reuse existing components first.
8. Use text + color for status.
9. Separate beginner explanation and expert evidence.
10. Prefer tables for standards, clauses, versions, decisions, and traceability.

---

## Approved Tokens

Use these tokens only.

```css
:root {
  --q-navy-900: #061F4A;
  --q-blue-700: #0B3D91;
  --q-blue-500: #105BD8;
  --q-text-900: #212121;
  --q-text-700: #323A45;
  --q-text-500: #64748B;
  --q-bg: #F8FAFC;
  --q-panel: #FFFFFF;
  --q-line: #D9E2EC;
  --q-pass: #15803D;
  --q-hold: #92400E;
  --q-fail: #B91C1C;
  --q-focus: #105BD8;
  --q-font-main: "Noto Sans KR", system-ui, sans-serif;
  --q-font-code: ui-monospace, Menlo, Consolas, monospace;
}
```

---

## Approved Components

Use these first:

- Trust Banner
- Standard Context Bar
- Workbench Search
- Module Card
- Status Badge
- Evidence Box
- Trace Table
- Beginner Box
- Expert Box
- Action Checklist
- Output Gate

Create a new component only when the existing components cannot solve the task and the new pattern is reusable across at least two modules.

---

## Output Patterns

### Standard Detail

```text
Trust Banner
Standard Context Bar
Beginner Box
Expert Box
Evidence Box
Trace Table
Action Checklist
```

### Customer Bridge

```text
Trust Banner
Question Input
Standard Selector
Decision Result
Evidence Summary
Action Checklist
Output Gate
```

### Training Page

```text
Learning Goal
Beginner Explanation
Expert Reference
Practice Task
Quiz
Feedback
Progress Status
```

### Warehouse Page

```text
Trust Banner
Search / Filter
Source Table
Metadata Panel
Version History
Action Checklist
```

---

## Preflight Checklist

Before implementation, answer:

```text
Target product:
Primary user:
First user action:
Input:
Standard/rule:
Decision status:
Evidence source:
Next action:
Existing components:
New component needed:
Verification:
HOLD:
```

---

## Verification Checklist

Before final output, check:

- [ ] Beginner can identify the first action.
- [ ] Standard/rule context is visible.
- [ ] Decision status is visible.
- [ ] Evidence is visible.
- [ ] Next action is visible.
- [ ] Approved tokens are used.
- [ ] Existing components are reused.
- [ ] Status uses text + color.
- [ ] Focus and labels are present.
- [ ] Mobile order remains logical.
- [ ] Reusable decisions are recorded in Memory.

---

## Final Response Format

```text
Summary:
Changed files:
Reused components:
New components:
Verification:
Accessibility:
Memory update:
HOLD:
Next action:
```

---

## Failure Conditions

Return HOLD or FAIL if:

- Evidence is missing.
- Status is color-only.
- Approved tokens are bypassed.
- Official product names are changed.
- User cannot identify the next action.
- Accessibility is not checked.
- New reusable decision is not recorded.

---

## Notes

This skill is instruction-first. Optional scripts, references, and assets may be added later only after Guidebook and Memory updates.

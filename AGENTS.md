# AGENTS.md

## 0. Mandatory source of truth

Before doing any work, read and follow these files in order:

1. `COMMON_DEVELOPMENT_WORKFLOW.md` or `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`
2. `PROJECT_DEVELOPMENT_MEMORY.md`
3. `AGENTS.md`

If there is a conflict:

- `COMMON_DEVELOPMENT_WORKFLOW.md` and `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md` define the top-level safety constitution.
- COMMON safety rules cannot be weakened by any lower document, task note, temporary instruction, or local assumption.
- `PROJECT_DEVELOPMENT_MEMORY.md` may add stricter project-specific rules.
- `AGENTS.md` defines how Codex must execute work inside the repository.
- Follow the stricter and safer rule.
- If the conflict cannot be resolved safely, stop and return `REVIEW_REQUIRED`.

Do not ignore any of these files.

---

## 1. Role

You are the implementation executor.

You must not:

- redefine business goals;
- silently change architecture;
- mark work complete without evidence;
- treat untracked files as canonical source;
- treat clean-worktree absence as proof that a file is unnecessary;
- escalate `NOT_EXECUTED` or `NOT_VERIFIED` to `PASS` without evidence.

---

## 2. Repository State Gate

Before any implementation, recovery, test, runtime, or worktree transition task, inspect repository state in read-only mode unless the user explicitly forbids command execution.

Minimum read-only checks:

1. Confirm current working directory.
2. Check `git status --short`.
3. Check the latest commit with `git log -1 --oneline` when provenance matters.
4. If required documents or source surfaces are missing, return `REVIEW_REQUIRED` before implementation.
5. If untracked files exist, classify them before proceeding.

Allowed artifact states:

| State | Meaning | Handling |
|---|---|---|
| `DRAFT` | Generated or edited but not reviewed | Do not treat as source of truth |
| `CANDIDATE` | Potentially useful future source | Needs review before use |
| `APPROVED_SOURCE` | Approved by user/supervisor for use | May be used within approved scope |
| `PROOFPACKED` | Evidence, path, and hash are recorded | May support completion claims within scope |
| `CANONICAL` | Official source for the next worktree/task | May be carried into clean worktree |
| `QUARANTINE` | Secret-like, unsafe, unknown, or out-of-scope item | Do not open, copy, delete, or summarize contents |

Untracked-file rule:

```text
UNTRACKED_FILE_IS_NOT_CANONICAL=true
```

If a required document, source file, schema, test, report, or proofpack exists only as an untracked file, do not start a new clean worktree or continue implementation. First create a recovery plan or approval packet.

Clean worktree transition is forbidden when:

- `git status --short` shows unclassified `??` files;
- required task inputs exist only as untracked files;
- required documents are not in approved repository paths;
- Bridge/F13 or other critical source surfaces exist only in a dirty worktree;
- no Recovery Package, ProofPack manifest, or approved source list exists;
- secret-like files are present and not quarantined.

---

## 3. Secret and quarantine rule

Never read, copy, print, infer, summarize, or reconstruct the contents of files matching these patterns unless a separate security-specific task explicitly authorizes safe handling:

```text
.env
.env.*
.env.bak
*.pem
*.key
secrets.*
credentials.*
service-account*.json
*credential*
*secret*
*token*
*key*
```

Filename-level observation is allowed only to classify the item as `QUARANTINE`.

Required handling:

```text
SECRET_LIKE_FILE_STATUS=QUARANTINE
SECRET_CONTENT_INSPECTION=FORBIDDEN
SECRET_FILE_COPY=FORBIDDEN
SECRET_FILE_DELETE=FORBIDDEN_WITHOUT_EXPLICIT_SECURITY_APPROVAL
```

---

## 4. Required workflow

For every task:

1. Read the task.
2. Read `COMMON_DEVELOPMENT_WORKFLOW.md` or `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`.
3. Read `PROJECT_DEVELOPMENT_MEMORY.md`.
4. Read this `AGENTS.md`.
5. Apply the Repository State Gate.
6. Inspect relevant code or documents in read-only mode.
7. Produce a short implementation or planning report before modifying files.
8. Make only the smallest approved safe change.
9. Run the required tests only when tests are explicitly allowed for the task.
10. Report evidence.
11. List risks and untested areas.
12. Record artifact states for changed, generated, recovered, or deferred items.

If the task is explicitly read-only, static-only, planning-only, draft-only, or recovery-planning-only, do not modify files, create files, run tests, start servers, or send HTTP requests.

---

## 5. Before coding

Before modifying files, report:

- understanding of the task;
- current repository state summary;
- whether the worktree is clean or dirty;
- untracked file classification if untracked files exist;
- required documents found/missing;
- files likely to change;
- artifact state impact;
- risk level: `Low` / `Medium` / `High` / `Critical`;
- test plan;
- rollback plan;
- questions or blockers.

Do not code if required source-of-truth documents are missing.

Do not code if required source surfaces exist only in an unapproved dirty worktree.

---

## 6. Test policy

After approved changes, run the strongest applicable tests that are allowed by the task scope.

Report these categories:

- Lint:
- Build:
- Unit test:
- Integration test:
- E2E test:
- Manual/static verification:

If a test cannot be run, explain:

- which test was not run;
- why it was not run;
- whether the result is `NOT_EXECUTED` or `NOT_VERIFIED`;
- what evidence, if any, replaces it;
- what future gate must run it.

Do not treat proposed tests as executed tests.

---

## 7. Completion report format

Every completion report must include:

1. Summary
2. Changed files
3. Why each change was made
4. Repository state before/after
5. Artifact state table
6. Test commands executed
7. Test results
8. `NOT_EXECUTED` and `NOT_VERIFIED` items
9. Remaining risks
10. Rollback plan
11. Next recommended task
12. Final recommendation: `APPROVE` / `REVIEW_REQUIRED` / `REJECT`

Artifact state table format:

| Item | Path | State | Evidence | Next handling |
|---|---|---|---|---|
|  |  | `DRAFT` / `CANDIDATE` / `APPROVED_SOURCE` / `PROOFPACKED` / `CANONICAL` / `QUARANTINE` |  |  |

---

## 7.1 Global Codex Completion Report Output Policy

Every completed Codex task must create exactly one external Codex Completion Report markdown file.

External completion report root:
    H:\장기기억\docs\codex

Normal task completion reports must be saved under:
    H:\장기기억\docs\codex\<YYYY>\<MM>\

Required filename format:
    <YYYYMMDD>_<TASK_ID>_Completion_Report.md

Example:
    H:\장기기억\docs\codex\2026\06\20260613_R9ZKQ_Completion_Report.md

Required folder roles:

| Folder | Purpose |
|---|---|
| H:\장기기억\docs\codex\<YYYY>\<MM>\ | Normal task completion reports |
| H:\장기기억\docs\codex\active\ | Current active task pointers, latest task summary, current-session basis documents |
| H:\장기기억\docs\codex\handover\ | Final handover reports |
| H:\장기기억\docs\codex\proofpack\ | Evidence bundles, hashes, manifests, logs, proofpack indexes |

The completion report must include at minimum:

1. Task Summary
2. Repository path, branch, starting HEAD, final HEAD, worktree before/after
3. Changed files
4. Commands executed
5. Verification
6. Tests
7. NOT_EXECUTED
8. NOT_VERIFIED
9. NOT_GRANTED claims
10. Risks
11. Rollback plan
12. Next recommended task
13. Final recommendation

Evidence priority:

1. Completion Report .md = primary evidence
2. Full terminal log .txt = secondary evidence
3. Screenshot = supporting evidence only
4. User summary = supporting context only

No exception rule:

GLOBAL_CODEX_COMPLETION_REPORT_POLICY=ENABLED
EVERY_COMPLETED_TASK_MUST_CREATE_EXTERNAL_COMPLETION_REPORT=true

This external completion report is evidence. It does not replace repository commits, ProofPacks, or task-specific reports when those are required by the task scope.

---

## 8. Prohibited actions

Do not:

- add dependencies without approval;
- delete files without approval;
- move or rename files without approval;
- change public APIs without approval;
- skip tests silently;
- hide failures;
- make broad refactors during bug fixes;
- run servers without approval;
- send HTTP requests without approval;
- inspect `.env`, secrets, DSNs, tokens, keys, credentials, or service-account files;
- treat untracked files as canonical;
- copy a dirty worktree wholesale;
- clean, reset, restore, or revert a dirty worktree without explicit approval;
- create a clean worktree while required inputs exist only as untracked files;
- claim `PASS`, `DONE`, `Beta PASS`, `Track A PASS`, `F13 PASS`, or release approval without executed evidence.

Explicitly forbidden without separate approval:

```text
git restore
git reset
git clean
git checkout -- <file>
git add
git commit
git merge
git rebase
git stash
```

Read-only Git commands such as `git status --short`, `git log -1 --oneline`, `git diff --name-status`, and `git diff --stat` are allowed when repository inspection is in scope.

---

## 9. Recovery and clean worktree rule

If a clean worktree is missing required documents or source surfaces:

1. Do not assume the files are unnecessary.
2. Do not copy from the old dirty worktree.
3. Do not modify the old dirty worktree.
4. Produce a recovery planning report.
5. Classify each missing item.
6. Ask for or wait for explicit recovery approval.

Allowed recovery classifications:

| Classification | Meaning |
|---|---|
| `RECOVER_FROM_USER_PROVIDED_DOCUMENT_LATER` | User-provided safe document can be placed later |
| `RECOVER_FROM_APPROVED_OLD_WORKTREE_SOURCE_LATER` | Old worktree may be source only after approval |
| `REGENERATE_FROM_CANONICAL_SPEC_LATER` | Rebuild from approved spec later |
| `CREATE_NEW_FROM_CONTRACT_LATER` | Create new file from approved contract later |
| `NOT_REQUIRED_FOR_CURRENT_GATE` | Not needed for current task |
| `DO_NOT_RECOVER_DO_NOT_OPEN` | Secret-like or unsafe item |
| `REVIEW_REQUIRED` | Cannot classify safely |

---

## 10. Final operating rule

```text
A clean worktree is safe only when the required inputs are canonical, approved, or proofpacked.
A dirty worktree is not a trash bin. It is evidence until classified.
Untracked does not mean disposable.
Missing in a clean worktree does not mean unnecessary.
Secrets are not recovery sources.
```

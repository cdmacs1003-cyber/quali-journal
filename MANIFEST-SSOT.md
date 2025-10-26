# MANIFEST-SSOT (Draft)

> Single Source of Truth for `cdmacs1003-cyber/quali-journal`

**Purpose**: Keep a human-readable index of project files. The machine index lives at `/_inventory/inventory.json`.

## Fields
- **path**: relative path in repo
- **role**: short description of the file's responsibility
- **component**: admin | orchestrator | ci | docs | tools | data | other
- **anchors**: list of unique strings used for idempotent patches
- **owner**: who reviews / approves changes
- **dod**: link to definition of done for this part

## Seed Entries
| path | role | component | anchors | owner | dod |
|---|---|---|---|---|---|
| /admin/index.html | Admin UI (runtime, token, KPI, export) | admin | clearAdminTokenJJ, runQa, btn-qa | @cdmacs1003-cyber | docs/DoD.md#admin-ui |

> This document will be expanded by the inventory workflow PR.

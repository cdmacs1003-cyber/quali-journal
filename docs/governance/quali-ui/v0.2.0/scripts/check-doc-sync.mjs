import { readFileSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const skillFile = "../../../../.agents/skills/quali-standard-ui-v0-2-0/SKILL.md";

const statuses = ["DRAFT", "INFO", "HOLD", "FAIL", "REVIEWED", "READY", "PASS", "ARCHIVED"];
const tokens = [
  "--q-navy-900",
  "--q-blue-700",
  "--q-blue-500",
  "--q-text-900",
  "--q-text-700",
  "--q-text-500",
  "--q-bg",
  "--q-panel",
  "--q-line",
  "--q-pass",
  "--q-hold",
  "--q-fail",
  "--q-focus"
];
const components = [
  "Trust Banner",
  "Standard Context Bar",
  "Standard Code Badge",
  "Workbench Search",
  "Module Card",
  "Status Badge",
  "Evidence Box",
  "Trace Table",
  "Beginner Box",
  "Expert Box",
  "Action Checklist",
  "Output Gate",
  "Line Break Gate"
];

const statusFiles = [
  "00_QUALI_UI_CONSTITUTION.md",
  "00_QUALI_UI_GUARDRAILS.md",
  "PROJECT_DEVELOPMENT_GUIDEBOOK.md",
  "PROJECT_DEVELOPMENT_MEMORY.md",
  skillFile
];
const tokenFiles = [
  "00_QUALI_UI_GUARDRAILS.md",
  "PROJECT_DEVELOPMENT_GUIDEBOOK.md",
  "PROJECT_DEVELOPMENT_MEMORY.md",
  skillFile
];
const componentFiles = [
  "PROJECT_DEVELOPMENT_GUIDEBOOK.md",
  "PROJECT_DEVELOPMENT_MEMORY.md",
  skillFile
];

function read(relativeFile) {
  return readFileSync(path.join(root, relativeFile), "utf8");
}

function requireTerms(files, terms, label) {
  let failed = false;
  for (const file of files) {
    const text = read(file);
    for (const term of terms) {
      if (!text.includes(term)) {
        console.error(`[FAIL] Missing ${label} "${term}" in ${file}`);
        failed = true;
      }
    }
  }
  return failed;
}

let failed = false;
failed = requireTerms(statusFiles, statuses, "status") || failed;
failed = requireTerms(tokenFiles, tokens, "token") || failed;
failed = requireTerms(componentFiles, components, "component") || failed;

for (const file of tokenFiles) {
  const text = read(file);
  if (text.includes("letter-spacing: -")) {
    console.error(`[FAIL] Negative letter-spacing remains in ${file}`);
    failed = true;
  }
}

if (failed) process.exit(1);
console.log("[PASS] Quali document sync check complete");

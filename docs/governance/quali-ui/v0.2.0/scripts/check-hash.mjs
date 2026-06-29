import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const sumsPath = path.join(root, "SHA256SUMS.txt");

if (!existsSync(sumsPath)) {
  console.error("[FAIL] SHA256SUMS.txt not found");
  process.exit(1);
}

const lines = readFileSync(sumsPath, "utf8").replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean);
let failed = false;

for (const line of lines) {
  const match = line.match(/^([a-fA-F0-9]{64})\s+(.+)$/);
  if (!match) {
    console.error(`[FAIL] Invalid SHA256SUMS line: ${line}`);
    failed = true;
    continue;
  }

  const [, expectedRaw, relativeFile] = match;
  const expected = expectedRaw.toLowerCase();
  const target = path.join(root, relativeFile);

  if (!existsSync(target)) {
    console.error(`[FAIL] Missing file listed in SHA256SUMS: ${relativeFile}`);
    failed = true;
    continue;
  }

  const actual = createHash("sha256").update(readFileSync(target)).digest("hex");
  if (actual !== expected) {
    console.error(`[FAIL] Hash mismatch: ${relativeFile}`);
    failed = true;
  } else {
    console.log(`[PASS] ${relativeFile}`);
  }
}

if (failed) process.exit(1);
console.log("[PASS] SHA256SUMS integrity check complete");

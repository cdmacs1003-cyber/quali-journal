import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const ignoredDirectories = new Set([".git", "node_modules"]);
const allowedHex = new Set([
  "#061f4a",
  "#0b3d91",
  "#105bd8",
  "#212121",
  "#323a45",
  "#64748b",
  "#f8fafc",
  "#ffffff",
  "#d9e2ec",
  "#15803d",
  "#92400e",
  "#b91c1c"
]);
const scannedExtensions = new Set([
  ".css",
  ".scss",
  ".html",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".mjs",
  ".json",
  ".md"
]);

function walk(directory, files = []) {
  if (!existsSync(directory)) return files;
  for (const entry of readdirSync(directory)) {
    if (ignoredDirectories.has(entry)) continue;
    const fullPath = path.join(directory, entry);
    const stat = statSync(fullPath);
    if (stat.isDirectory()) {
      walk(fullPath, files);
    } else if (scannedExtensions.has(path.extname(entry))) {
      files.push(fullPath);
    }
  }
  return files;
}

const hexPattern = /#[0-9a-fA-F]{6}/g;
let failed = false;

for (const file of walk(root)) {
  const text = readFileSync(file, "utf8");
  const matches = text.match(hexPattern) || [];
  for (const hex of matches) {
    if (!allowedHex.has(hex.toLowerCase())) {
      console.error(`[FAIL] Unapproved HEX ${hex} in ${path.relative(root, file)}`);
      failed = true;
    }
  }
}

if (failed) process.exit(1);
console.log("[PASS] Quali token usage check complete");

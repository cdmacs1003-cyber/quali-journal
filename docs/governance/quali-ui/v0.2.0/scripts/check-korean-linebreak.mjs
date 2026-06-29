import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const uiRoots = ["src", "app", "components", "pages", "public", "docs/html"];
const extensions = new Set([".html", ".jsx", ".tsx", ".vue", ".svelte"]);
const forbiddenPatterns = [
  /문장으\s*[\r\n]+\s*로/,
  /리\s*[\r\n]+\s*스크/,
  /시각요\s*[\r\n]+\s*약/,
  /은\s*[\r\n]+\s*<\/span>/,
  /는\s*[\r\n]+\s*<\/span>/,
  /이\s*[\r\n]+\s*<\/span>/,
  /가\s*[\r\n]+\s*<\/span>/,
  /을\s*[\r\n]+\s*<\/span>/,
  /를\s*[\r\n]+\s*<\/span>/,
  /으로\s*[\r\n]+\s*<\/span>/
];

function walk(directory, files = []) {
  if (!existsSync(directory)) return files;
  for (const entry of readdirSync(directory)) {
    const fullPath = path.join(directory, entry);
    const stat = statSync(fullPath);
    if (stat.isDirectory()) {
      walk(fullPath, files);
    } else if (extensions.has(path.extname(entry))) {
      files.push(fullPath);
    }
  }
  return files;
}

let failed = false;
const files = uiRoots.flatMap((directory) => walk(path.join(root, directory)));

for (const file of files) {
  const text = readFileSync(file, "utf8");
  for (const pattern of forbiddenPatterns) {
    if (pattern.test(text)) {
      console.error(`[FAIL] Korean line-break risk in ${path.relative(root, file)}`);
      failed = true;
    }
  }
}

if (failed) process.exit(1);
console.log(`[PASS] Quali Korean line-break static check complete (${files.length} UI files scanned)`);

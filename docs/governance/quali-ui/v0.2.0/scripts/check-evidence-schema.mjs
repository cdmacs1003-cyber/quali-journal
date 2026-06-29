import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const schemaPath = path.join(root, "schemas", "evidence.schema.json");
const requiredFields = [
  "standard_id",
  "revision",
  "clause",
  "source_file",
  "page",
  "evidence_level",
  "verification_status",
  "limitation",
  "next_action"
];

if (!existsSync(schemaPath)) {
  console.error("[FAIL] schemas/evidence.schema.json not found");
  process.exit(1);
}

let failed = false;
const schema = JSON.parse(readFileSync(schemaPath, "utf8"));
const schemaRequired = new Set(schema.required || []);

for (const field of requiredFields) {
  if (!schemaRequired.has(field)) {
    console.error(`[FAIL] Evidence schema missing required field: ${field}`);
    failed = true;
  }
  if (!schema.properties || !schema.properties[field]) {
    console.error(`[FAIL] Evidence schema missing property definition: ${field}`);
    failed = true;
  }
}

function walk(directory, files = []) {
  if (!existsSync(directory)) return files;
  for (const entry of readdirSync(directory)) {
    const fullPath = path.join(directory, entry);
    const stat = statSync(fullPath);
    if (stat.isDirectory()) {
      walk(fullPath, files);
    } else if (entry.endsWith(".json")) {
      files.push(fullPath);
    }
  }
  return files;
}

const exampleRoot = path.join(root, "examples", "evidence");
for (const file of walk(exampleRoot)) {
  const data = JSON.parse(readFileSync(file, "utf8"));
  for (const field of requiredFields) {
    if (!(field in data)) {
      console.error(`[FAIL] Evidence example ${path.relative(root, file)} missing field: ${field}`);
      failed = true;
    }
  }
}

if (failed) process.exit(1);
console.log("[PASS] Quali evidence schema check complete");

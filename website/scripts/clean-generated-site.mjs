import { rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(websiteDir, "..");

const generatedPaths = [
  "docs/funding.json",
  "docs/.well-known",
  "docs/index.html",
  "docs/CNAME",
  "docs/install.sh",
  "docs/install.ps1",
  "docs/mesh-llm-logo.svg",
  "docs/assets",
  "docs/catalog",
  "docs/docs",
  "docs/pagefind",
  "website/src/assets/site.generated.css",
];

for (const relativePath of generatedPaths) {
  const target = path.resolve(repoRoot, relativePath);
  if (!target.startsWith(`${repoRoot}${path.sep}`)) {
    throw new Error(`Refusing to remove path outside repo root: ${target}`);
  }
  await rm(target, { force: true, recursive: true });
}

console.log("Removed generated website artifacts from docs/ and website/src/assets/.");

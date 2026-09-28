import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(scriptDir, "..");

// The star count is the one number on the site that changes without this
// repository changing, so every page must render it from
// `site.githubStarsFallback` \u2014 filled in at build time by the GitHub fetch in
// src/_data/site.js and refreshed in the browser by src/assets/github-stars.js
// \u2014 rather than spelling a number into the page.
//
// Regression guard: the Exo comparison shipped "Mesh is newer (1.1k stars)"
// while the repository was already at 3.4k, and nothing ever updated it.
const templatedStars = "{{ site.githubStarsFallback }}";

const readSource = (relativePath) => readFileSync(path.resolve(websiteDir, relativePath), "utf8");

// The nav is the surface src/assets/github-stars.js refreshes in the browser.
const nav = readSource("src/_includes/nav.njk");
assert.ok(
  nav.includes(`<span data-github-stars>${templatedStars}</span>`),
  "nav star count must render from site.githubStarsFallback",
);
assert.ok(
  nav.includes(`${templatedStars} stars`),
  "nav aria-label must render the star count from site.githubStarsFallback",
);

// Our own count in docs prose must be templated too. Counts for other projects
// ("45k+ stars" for exo) are static claims about someone else and are out of
// scope, so only lines that talk about Mesh are checked.
const exoComparison = readSource("src/docs/pages/exo-comparison.md");
const meshStarLines = exoComparison
  .split("\n")
  .filter((line) => /\bmesh\b/i.test(line) && /\([^)]*stars?[^)]*\)/i.test(line));

assert.ok(meshStarLines.length > 0, "expected the Exo comparison to state a Mesh star count");
for (const line of meshStarLines) {
  assert.ok(
    line.includes(templatedStars),
    `Mesh star count must come from ${templatedStars}: ${line.trim()}`,
  );
}

console.log("Website data sources render live values (nav + Exo comparison checked).");

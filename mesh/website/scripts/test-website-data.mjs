import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import plugins from "../src/_data/plugins.js";

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

// Discovery and installation share one release list, including the cases where
// a repository slug is not the installed plugin identifier.
const releasedPlugins = plugins.groups.flatMap((group) => group.plugins);
assert.equal(plugins.count, releasedPlugins.length);
assert.ok(plugins.count > 0, "the released plugin directory must not be empty");
assert.match(plugins.checkedOn, /^\d{4}-\d{2}-\d{2}$/);
assert.match(plugins.hostVersion, /^\d+\.\d+\.\d+$/);
for (const field of ["name", "repository"]) {
  assert.equal(new Set(releasedPlugins.map((plugin) => plugin[field])).size, plugins.count, `plugin ${field} values must be unique`);
}
assert.equal(new Set(plugins.groups.map((group) => group.id)).size, plugins.groups.length);
for (const group of plugins.groups) {
  assert.match(group.id, /^[a-z][a-z-]*$/);
  assert.ok(group.plugins.length > 0, `${group.id} must contain published plugins`);
}
for (const plugin of releasedPlugins) {
  assert.match(plugin.name, /^[a-z0-9]+(?:-[a-z0-9]+)*$/);
  assert.match(plugin.repository, /^[a-z0-9]+(?:-[a-z0-9]+)*-plugin$/);
  assert.match(plugin.version, /^v\d+\.\d+\.\d+$/);
  for (const field of ["title", "summary", "platforms"]) {
    assert.ok(plugin[field]?.trim(), `${plugin.name} needs ${field}`);
  }
}
assert.equal(releasedPlugins.find((plugin) => plugin.repository === "ln-payments-plugin")?.name, "pay-lightning");
assert.equal(releasedPlugins.find((plugin) => plugin.repository === "model-price-plugin")?.name, "model-prices");

const directory = readSource("src/plugins/index.njk");
const pluginGuide = readSource("src/docs/pages/plugins.md");
for (const source of [directory, pluginGuide]) {
  assert.ok(source.includes("plugins.groups"), "directory and guide must consume the shared plugin release data");
  assert.ok(source.includes("plugins.count"), "plugin totals must not be hard-coded");
  assert.ok(source.includes("{{ plugin.repository }}"), "repository links must use the canonical release data");
}
assert.ok(pluginGuide.includes("### Install a downloaded archive"), "archive-install links need a destination");
const hero = readSource("src/_includes/sections/hero.njk");
assert.match(hero, /id="hero-plugins-title">Extend Mesh<\/h2>/);
assert.match(hero, /href="\/plugins\/">Browse plugins/);
assert.ok(hero.includes("{{ plugins.count }} published plugins"));
assert.ok(hero.includes('href="/docs/pages/sdk/"'), "the Embed entry point must remain available");
assert.ok(readSource("src/_data/docs.js").includes('["Plugin directory", "/plugins/"]'));

console.log("Website data sources checked: GitHub stars, plugin releases, installation guide and Extend entry point.");

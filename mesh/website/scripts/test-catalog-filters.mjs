import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(scriptDir, "..");
const readSource = (relativePath) => readFileSync(path.resolve(websiteDir, relativePath), "utf8");

const catalogPage = readSource("src/catalog/index.njk");
const homeSection = readSource("src/_includes/sections/catalog-home.njk");

// #1955: an option selects a row only when the option's `.value` and the row's
// `data-*` attribute are the same string once the HTML parser has had them.
// `<option value="&le; 8 GB">` is parsed as markup, so the browser hands
// `applyFilters()` `≤ 8 GB`. A `data-*` attribute written through `escapeHtml()`
// keeps its ampersand escaped, so the browser hands back the literal
// `&le; 8 GB`, and the `===` comparison in `applyFilters()` matches nothing.
//
// `attributeValue` is the one decode step the parser performs. The test renders
// rows through the page's own `renderRows()` and reads the attributes back the
// way `getAttribute()` does, so every assertion is about shipped behaviour.

const NAMED_ENTITIES = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: "\u00a0",
  le: "\u2264",
  ge: "\u2265",
  middot: "\u00b7",
  ndash: "\u2013",
  mdash: "\u2014",
};

// Non-recursive, like the parser: `&amp;le;` decodes to `&le;` and stops there.
function attributeValue(raw) {
  return String(raw).replace(/&(#x[0-9a-f]+|#[0-9]+|[a-z]+);/gi, (whole, body) => {
    if (body[0] === "#") {
      const codePoint = /^#x/i.test(body)
        ? Number.parseInt(body.slice(2), 16)
        : Number.parseInt(body.slice(1), 10);
      return Number.isNaN(codePoint) ? whole : String.fromCodePoint(codePoint);
    }
    const named = NAMED_ENTITIES[body.toLowerCase()];
    return named === undefined ? whole : named;
  });
}

const catalogScript = catalogPage.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(catalogScript, "src/catalog/index.njk must carry the catalog script inline");

// Run the shipped script against a stub DOM. `fetch` never settles, so the only
// thing that happens is whatever the script's own functions do when called.
const context = vm.createContext({ console });
vm.runInContext(
  `
  var __elements = {};
  function __element(id) {
    if (!__elements[id]) {
      __elements[id] = {
        id: id, innerHTML: "", textContent: "", value: "", style: {},
        addEventListener: function () {}, querySelector: function () { return null; },
        querySelectorAll: function () { return []; },
      };
    }
    return __elements[id];
  }
  var document = { getElementById: __element, addEventListener: function () {} };
  var fetch = function () { return new Promise(function () {}); };
  var navigator = { clipboard: { writeText: function () { return new Promise(function () {}); } } };
  `,
  context,
);
vm.runInContext(catalogScript[1], context, { filename: "catalog/index.njk:<script>" });

const renderRows = vm.runInContext("renderRows", context);
const displaySize = vm.runInContext("displaySize", context);
const parseSizeInGb = vm.runInContext("parseSizeInGb", context);
const catalogRows = vm.runInContext("__element('catalog-rows')", context);

// Rows in the shape `meshllm/catalog` actually returns, one per interesting
// bucket edge. `61 layers` exercises the size override the page keeps for
// Kimi-K2-Thinking; the vision row is the only one with `mmproj_json`.
const FIXTURE_ROWS = [
  { name: "Qwen2.5-0.5B-Instruct-Q4_K_M", size: "491 MB", source_model_repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF", source_file: "Q4_K_M/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", mmproj_json: "", layer_package_available: false },
  { name: "Qwen3-14B-Q4_K_M", size: "9.0GB", source_model_repo: "unsloth/Qwen3-14B-GGUF", source_file: "Q4_K_M/Qwen3-14B-Q4_K_M.gguf", mmproj_json: "", layer_package_available: false },
  { name: "Qwen3-32B-Q4_K_M", size: "19.8GB", source_model_repo: "unsloth/Qwen3-32B-GGUF", source_file: "Q4_K_M/Qwen3-32B-Q4_K_M.gguf", mmproj_json: "", layer_package_available: false },
  { name: "Llama-4-Scout-Q4_K_M", size: "22.5GB", source_model_repo: "unsloth/Llama-4-Scout-GGUF", source_file: "Q4_K_M/Llama-4-Scout-Q4_K_M.gguf", mmproj_json: "", layer_package_available: false },
  { name: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", size: "18.6GB", source_model_repo: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", source_file: "Q4_K_M/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", mmproj_json: "", layer_package_available: false },
  { name: "Qwen3.5-9B-Vision-Q4_K_M", size: "5.8GB", source_model_repo: "unsloth/Qwen3.5-9B-GGUF", source_file: "Q4_K_M/Qwen3.5-9B-GGUF-Q4_K_M.gguf", mmproj_json: '{"file":"mmproj-BF16.gguf","repo":"unsloth/Qwen3.5-9B-GGUF","revision":"main"}', layer_package_available: false },
  { name: "unsloth/Kimi-K2-Thinking-GGUF:UD-Q4_K_XL", size: "61 layers", source_model_repo: "unsloth/Kimi-K2-Thinking-GGUF", source_file: "UD-Q4_K_XL/Kimi-K2-Thinking-UD-Q4_K_XL-00001-of-00046.gguf", mmproj_json: "", layer_package_available: true },
];

renderRows(FIXTURE_ROWS.map((row) => ({ row })));

function renderedAttributes(name) {
  const values = [...catalogRows.innerHTML.matchAll(new RegExp(`\\b${name}="([^"]*)"`, "g"))].map(
    (match) => attributeValue(match[1]),
  );
  assert.equal(
    values.length,
    FIXTURE_ROWS.length,
    `renderRows() must write one ${name} attribute per row, found ${values.length}`,
  );
  return values;
}

function optionValues(source, selectId, surface) {
  const select = source.match(new RegExp(`<select id="${selectId}"[\\s\\S]*?</select>`));
  assert.ok(select, `${surface} must render <select id="${selectId}">`);
  return [...select[0].matchAll(/<option\b[^>]*\bvalue="([^"]*)"/g)]
    .map((match) => attributeValue(match[1]))
    .filter((value) => value !== "");
}

const catalogHardware = optionValues(catalogPage, "filter-hardware", "the catalog page");
const homeHardware = optionValues(homeSection, "filter-hardware", "the homepage catalog section");
const catalogCapabilities = optionValues(catalogPage, "filter-capability", "the catalog page");

// 1. Every bucket renderRows() writes into data-hardware must be one an <option>
//    actually offers. This is the double-escape failure from #1955: escapeHtml()
//    turns the `&le;` that guessHardware() returns into `&amp;le;`, so the
//    browser hands applyFilters() the literal "&le; 8 GB" while the option's
//    .value has already been decoded to "≤ 8 GB".
const renderedHardware = renderedAttributes("data-hardware");
for (const bucket of renderedHardware) {
  assert.ok(
    catalogHardware.includes(bucket),
    `renderRows() writes data-hardware=${JSON.stringify(bucket)}, which no <option> selects; the ` +
      `options decode to ${JSON.stringify(catalogHardware)} and applyFilters() compares them with ===`,
  );
}

// 1. Both surfaces must offer the same buckets, and the labels must name the
//    thresholds in use with no gap between them. "≤ 8 GB" then "11–20 GB"
//    leaves 8.1–10.9 GB unnameable, which is why four 9.0 GB models showed up
//    under "11–20 GB"; "40 GB+" above a 20 GB ceiling is why the 22.5 GB
//    Llama-4-Scout showed up under "40 GB+".
function bucketBounds(label) {
  const text = label.trim();
  const atMost = text.match(/^(?:≤|<=)\s*([\d.]+)\s*GB$/i);
  if (atMost) return { lower: 0, upper: Number(atMost[1]), upperInclusive: true };
  const range = text.match(/^([\d.]+)\s*(?:–|—|-)\s*([\d.]+)\s*GB$/i);
  if (range) return { lower: Number(range[1]), upper: Number(range[2]), upperInclusive: true };
  const plus = text.match(/^([\d.]+)\s*GB\+$/i);
  if (plus) return { lower: Number(plus[1]), upper: Number.POSITIVE_INFINITY, upperInclusive: false };
  return null;
}

assert.ok(catalogHardware.length > 0, "the hardware filter must offer at least one bucket");
assert.deepEqual(
  [...catalogHardware].sort(),
  [...homeHardware].sort(),
  "the catalog page and the homepage section must offer the same hardware buckets",
);

const buckets = catalogHardware.map((label) => {
  const bounds = bucketBounds(label);
  assert.ok(
    bounds,
    `hardware bucket ${JSON.stringify(label)} names neither a ceiling ("≤ N GB"), a range ` +
      `("N–M GB"), nor a floor ("N GB+")`,
  );
  return { label, ...bounds };
});
buckets.sort((a, b) => a.lower - b.lower);
assert.equal(buckets[0].lower, 0, `the smallest hardware bucket must start at 0 GB, not ${buckets[0].lower}`);
buckets.forEach((bucket, index) => {
  if (index === 0) return;
  const previous = buckets[index - 1];
  assert.equal(
    bucket.lower,
    previous.upper,
    `${JSON.stringify(previous.label)} and ${JSON.stringify(bucket.label)} do not meet: sizes above ` +
      `${previous.upper} GB and up to ${bucket.lower} GB belong to neither label`,
  );
});

// 2. Every bucket `renderRows()` writes must be selectable, and must be the one
//    whose own bounds contain the size the row displays.
FIXTURE_ROWS.forEach((row, index) => {
  const shown = displaySize(row);
  const sizeGb = parseSizeInGb(shown);
  assert.ok(!Number.isNaN(sizeGb), `${row.name}: displayed size ${JSON.stringify(shown)} must parse`);
  const matching = buckets.filter(
    (bucket) =>
      (sizeGb > bucket.lower || (sizeGb === bucket.lower && bucket.lower === 0)) &&
      (bucket.upperInclusive ? sizeGb <= bucket.upper : sizeGb < bucket.upper),
  );
  assert.equal(
    matching.length,
    1,
    `${shown} must fall in exactly one hardware bucket, found ` +
      JSON.stringify(matching.map((bucket) => bucket.label)),
  );
  assert.equal(
    renderedHardware[index],
    matching[0].label,
    `${row.name} displays ${shown} and renders data-hardware=${JSON.stringify(renderedHardware[index])}, ` +
      `but only ${JSON.stringify(matching[0].label)} covers that size — the filter compares the two with ===`,
  );
});

// 3. The homepage lists the same models from static markup, so a hand-written
//    `data-hardware` must be the bucket the catalog page gives that row's own
//    displayed size. This is what caught Llama-4-Scout at 22.5 GB in the middle
//    bucket, and the stale fallback row on the catalog page.
// Only authored markup carries a bucket a human chose. The catalog page's own
// renderRows() template also contains data-hardware, and its size is a
// placeholder rather than a size, so the script block is out of scope here.
// Of the rows that remain, the homepage section lists every model as static
// markup and the catalog page carries one fallback row for a failed fetch.
const authoredMarkup = (source) => source.replace(/<script>[\s\S]*?<\/script>/g, "");

function authoredRows(source, surface) {
  return authoredMarkup(source)
    .split(/(?=<tr\b)/)
    .filter((chunk) => /\bdata-hardware="/.test(chunk))
    .map((chunk) => {
      const size = chunk.match(/data-label="Size \/ quant">([^<]*)/);
      assert.ok(size, `${surface} has a row with data-hardware but no displayed size`);
      const name = chunk.match(/<strong>([^<]*)<\/strong>/);
      return {
        name: name ? attributeValue(name[1]) : attributeValue(size[1]),
        size: attributeValue(size[1]).trim(),
        hardware: attributeValue(chunk.match(/\bdata-hardware="([^"]*)"/)[1]),
      };
    });
}

const authoredHomeRows = authoredRows(homeSection, "the homepage catalog section");
const authoredCatalogRows = authoredRows(catalogPage, "the catalog page");
const authored = [...authoredHomeRows, ...authoredCatalogRows];
assert.ok(authored.length > 0, "expected hand-authored catalog rows to check");

for (const row of authored) {
  const sizeGb = parseSizeInGb(row.size);
  if (Number.isNaN(sizeGb)) continue;
  const matching = buckets.filter(
    (bucket) =>
      (sizeGb > bucket.lower || (sizeGb === bucket.lower && bucket.lower === 0)) &&
      (bucket.upperInclusive ? sizeGb <= bucket.upper : sizeGb < bucket.upper),
  );
  assert.equal(
    matching.length,
    1,
    `${row.name} displays ${row.size}, which must fall in exactly one hardware bucket`,
  );
  assert.equal(
    row.hardware,
    matching[0].label,
    `${row.name} displays ${row.size} but is hand-tagged ${JSON.stringify(row.hardware)}; that size ` +
      `belongs in ${JSON.stringify(matching[0].label)}`,
  );
}

// 4. `meshllm/catalog` has no `capabilities` column, so the page must derive the
//    tag from the row: `mmproj_json` marks the vision models, the model or repo
//    name marks the code ones.
// Keyed by model name so the expectation cannot drift out of step with the
// fixture order.
const expectedCapabilities = {
  "Qwen2.5-0.5B-Instruct-Q4_K_M": [],
  "Qwen3-14B-Q4_K_M": [],
  "Qwen3-32B-Q4_K_M": [],
  "Llama-4-Scout-Q4_K_M": [],
  "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M": ["Code"],
  "Qwen3.5-9B-Vision-Q4_K_M": ["Vision"],
  "unsloth/Kimi-K2-Thinking-GGUF:UD-Q4_K_XL": [],
};

const renderedCapabilities = renderedAttributes("data-capability");
renderedCapabilities.forEach((capability, index) => {
  const tags = capability === "" ? [] : capability.split(" ");
  assert.ok(
    capability === "" || /^\S+(?: \S+)*$/.test(capability),
    `${FIXTURE_ROWS[index].name}: data-capability must be space-separated tokens that ` +
      `applyFilters() can split, got ${JSON.stringify(capability)}`,
  );
  for (const tag of tags) {
    assert.ok(
      catalogCapabilities.includes(tag),
      `${FIXTURE_ROWS[index].name}: capability ${JSON.stringify(tag)} is written into ` +
        `data-capability but no <option> offers it; the options are ${JSON.stringify(catalogCapabilities)}`,
    );
  }
  const name = FIXTURE_ROWS[index].name;
  assert.ok(
    Object.hasOwn(expectedCapabilities, name),
    `the fixture row ${name} has no capability expectation`,
  );
  assert.deepEqual(tags, expectedCapabilities[name], `${name} must carry capability ` +
    `${JSON.stringify(expectedCapabilities[name])}, got ${JSON.stringify(capability)}`);
});

// The homepage's hand-written capability tags must be selectable too.
for (const chunk of homeSection.split(/(?=<tr\b)/).filter((c) => /\bdata-capability="/.test(c))) {
  const tags = attributeValue(chunk.match(/\bdata-capability="([^"]*)"/)[1]).split(" ").filter(Boolean);
  for (const tag of tags) {
    assert.ok(
      catalogCapabilities.includes(tag),
      `the homepage tags a row ${JSON.stringify(tag)}, which the capability filter offers no option for`,
    );
  }
}

console.log(
  `Catalog filters: ${catalogHardware.length} hardware buckets selectable and contiguous, ` +
    `${catalogCapabilities.length} capability options derivable, ${authored.length} authored rows ` +
    `bucketed by their own size (${authoredHomeRows.length} homepage models + ` +
    `${authoredCatalogRows.length} catalog fallback row).`,
);

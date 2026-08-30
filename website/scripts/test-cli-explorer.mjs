import assert from "node:assert/strict";
import { createReadStream, existsSync, mkdirSync, statSync } from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(websiteDir, "..");
const docsRoot = path.resolve(repoRoot, "docs");
const explorerPath = "/docs/pages/cli-explorer/";
const screenshotDirectory = path.resolve(websiteDir, "test-results/cli-explorer");

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
};

function serveDocs() {
  if (!existsSync(docsRoot)) {
    throw new Error("docs/ does not exist; run `just website-build` before the browser suite");
  }

  const server = http.createServer((request, response) => {
    try {
      const requestUrl = new URL(request.url ?? "/", "http://127.0.0.1");
      let relativePath = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, "");
      if (relativePath.length === 0 || relativePath.endsWith("/")) relativePath += "index.html";
      const target = path.resolve(docsRoot, relativePath);
      if (!target.startsWith(`${docsRoot}${path.sep}`) || !existsSync(target) || !statSync(target).isFile()) {
        response.writeHead(404).end("Not found");
        return;
      }
      response.writeHead(200, {
        "Cache-Control": "no-store",
        "Content-Type": mimeTypes[path.extname(target)] ?? "application/octet-stream",
      });
      createReadStream(target).pipe(response);
    } catch (error) {
      response.writeHead(400).end(error instanceof Error ? error.message : String(error));
    }
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        reject(new Error("unable to determine local server port"));
        return;
      }
      resolve({ server, url: `http://127.0.0.1:${address.port}` });
    });
  });
}

async function firstAvailable(page, selectors) {
  for (const selector of selectors) {
    const locator = page.locator(selector).first();
    if (await locator.count()) return locator;
  }
  return null;
}

async function requireVisible(locator, message) {
  assert.ok(locator, message);
  await locator.waitFor({ state: "visible" });
  return locator;
}

function nodeLocator(page) {
  return page.locator(
    '[role="treeitem"], [data-cli-node], g.node[role="button"], [data-node-kind]'
  );
}

async function visibleNodeCount(page) {
  return nodeLocator(page).evaluateAll((nodes) =>
    nodes.filter((node) => {
      const style = getComputedStyle(node);
      const box = node.getBoundingClientRect();
      return style.display !== "none" && style.visibility !== "hidden" && box.width > 0 && box.height > 0;
    }).length
  );
}

async function assertNoNodeCollisions(page) {
  const boxes = await nodeLocator(page).evaluateAll((nodes) =>
    nodes
      .map((node) => {
        const box = node.getBoundingClientRect();
        const style = getComputedStyle(node);
        return {
          display: style.display,
          visibility: style.visibility,
          x: box.x,
          y: box.y,
          width: box.width,
          height: box.height,
        };
      })
      .filter((box) => box.display !== "none" && box.visibility !== "hidden" && box.width > 1 && box.height > 1)
  );

  for (let left = 0; left < boxes.length; left += 1) {
    for (let right = left + 1; right < boxes.length; right += 1) {
      const a = boxes[left];
      const b = boxes[right];
      const overlapWidth = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
      const overlapHeight = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
      if (overlapWidth > 2 && overlapHeight > 2) {
        throw new Error(`CLI tree nodes overlap (${left} and ${right})`);
      }
    }
  }
}

async function assertNoPageOverflow(page) {
  const overflow = await page.evaluate(() => ({
    horizontal: document.documentElement.scrollWidth - window.innerWidth,
  }));
  assert.ok(overflow.horizontal <= 1, `page overflows horizontally by ${overflow.horizontal}px`);
}

async function visibleTreeItemData(page) {
  return page.locator('g.cli-explorer-node[role="treeitem"]').evaluateAll((nodes) =>
    nodes
      .map((node) => {
        const box = node.getBoundingClientRect();
        const style = getComputedStyle(node);
        return {
          path: node.getAttribute("data-path") || "",
          level: Number(node.getAttribute("aria-level") || 0),
          expanded: node.getAttribute("aria-expanded"),
          tabindex: node.getAttribute("tabindex"),
          opacity: Number(style.opacity),
          display: style.display,
          visibility: style.visibility,
          x: box.x,
          y: box.y,
          width: box.width,
          height: box.height,
        };
      })
      .filter((item) =>
        item.display !== "none"
        && item.visibility !== "hidden"
        && item.opacity > 0.05
        && item.width > 1
        && item.height > 1
      )
  );
}

async function activeTreePath(page) {
  return page.evaluate(() => document.activeElement?.getAttribute("data-path") || "");
}

function boxesOverlap(a, b) {
  const overlapWidth = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
  const overlapHeight = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
  return overlapWidth > 2 && overlapHeight > 2;
}

async function orientationSnapshot(page) {
  return page.evaluate(() => {
    const app = document.querySelector("[data-cli-explorer]");
    const orientation = app?.dataset.orientation || "";
    const parseTransform = (value) => {
      const match = value?.match(/translate\(\s*(-?[\d.]+)[, ]+(-?[\d.]+)\s*\)/);
      return match ? { x: Number(match[1]), y: Number(match[2]) } : null;
    };
    const nodes = [...document.querySelectorAll('g.cli-explorer-node[role="treeitem"]')]
      .filter((node) => {
        const style = getComputedStyle(node);
        const box = node.getBoundingClientRect();
        return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0.05 && box.width > 1 && box.height > 1;
      })
      .map((node) => ({
        path: node.getAttribute("data-path") || "",
        level: Number(node.getAttribute("aria-level") || 0),
        expanded: node.getAttribute("aria-expanded"),
        selected: node.getAttribute("aria-selected"),
        transform: parseTransform(node.getAttribute("transform")),
      }));
    const firstChild = nodes.find((node) => node.level === 2);
    const root = nodes.find((node) => node.level === 1);
    const buses = [...document.querySelectorAll(".cli-explorer-root-bus, .cli-explorer-branch-bus, .cli-explorer-vertical-bus")]
      .filter((bus) => Number(getComputedStyle(bus).opacity) > 0.05)
      .map((bus) => ({
        className: bus.getAttribute("class") || "",
        d: bus.getAttribute("d") || "",
        strokeWidth: Number.parseFloat(getComputedStyle(bus).strokeWidth),
      }));
    const links = [...document.querySelectorAll(".cli-explorer-link")]
      .filter((link) => Number(getComputedStyle(link).opacity) > 0.05)
      .map((link) => ({
        className: link.getAttribute("class") || "",
        d: link.getAttribute("d") || "",
        strokeWidth: Number.parseFloat(getComputedStyle(link).strokeWidth),
      }));
    return {
      orientation,
      root,
      firstChild,
      selectedPath: nodes.find((node) => node.selected === "true")?.path || "",
      buses,
      links,
    };
  });
}

async function assertOrientationControl(page, viewportName) {
  const group = page.locator('.cli-explorer-orientation[role="group"]').first();
  assert.ok(await group.isVisible(), `${viewportName}: orientation control group is not visible`);
  assert.match(await group.getAttribute("aria-label"), /orientation/i, `${viewportName}: orientation group lacks an accessible label`);
  const buttons = group.locator('[data-action="orientation"]');
  assert.equal(await buttons.count(), 2, `${viewportName}: orientation control must expose exactly two choices`);
  for (const [orientation, direction] of [["horizontal", "left to right"], ["vertical", "top to bottom"]]) {
    const button = group.locator(`[data-orientation="${orientation}"]`).first();
    assert.ok(await button.isVisible(), `${viewportName}: ${orientation} orientation button is not visible`);
    assert.match(await button.getAttribute("aria-label"), new RegExp(`${orientation}.*${direction}`, "i"), `${viewportName}: ${orientation} button label is ambiguous`);
    assert.match(await button.getAttribute("title"), new RegExp(`${orientation}.*${direction}`, "i"), `${viewportName}: ${orientation} button tooltip is ambiguous`);
  }
  const expected = viewportName === "mobile" ? "vertical" : "horizontal";
  assert.equal(await page.locator("[data-cli-explorer]").getAttribute("data-orientation"), expected, `${viewportName}: responsive orientation default is ${expected}`);
  assert.equal(await group.locator('[aria-pressed="true"]').count(), 1, `${viewportName}: orientation control has more than one pressed choice`);
  assert.equal(await group.locator(`[data-orientation="${expected}"]`).getAttribute("aria-pressed"), "true", `${viewportName}: ${expected} choice is not pressed by default`);
}

async function assertResponsiveOrientationDefault(page, viewportName) {
  const original = page.viewportSize();
  assert.ok(original, `${viewportName}: viewport size is unavailable for responsive orientation testing`);
  const narrow = viewportName === "desktop";
  await page.setViewportSize({
    width: narrow ? 390 : 1440,
    height: narrow ? 844 : 1000,
  });
  await page.waitForTimeout(120);
  const alternateExpected = narrow ? "vertical" : "horizontal";
  assert.equal(
    await page.locator("[data-cli-explorer]").getAttribute("data-orientation"),
    alternateExpected,
    `${viewportName}: responsive orientation did not follow the ${alternateExpected} default before user choice`
  );
  await page.setViewportSize(original);
  await page.waitForTimeout(120);
  const originalExpected = viewportName === "mobile" ? "vertical" : "horizontal";
  assert.equal(
    await page.locator("[data-cli-explorer]").getAttribute("data-orientation"),
    originalExpected,
    `${viewportName}: responsive orientation did not restore the ${originalExpected} default before user choice`
  );
}

async function assertExplicitOrientationAcrossResize(page, viewportName) {
  const initialOrientation = viewportName === "mobile" ? "vertical" : "horizontal";
  const alternateOrientation = initialOrientation === "vertical" ? "horizontal" : "vertical";
  const original = page.viewportSize();
  assert.ok(original, `${viewportName}: viewport size is unavailable for explicit orientation persistence testing`);
  await page.locator(`[data-action="orientation"][data-orientation="${alternateOrientation}"]`).first().click();
  await page.waitForTimeout(100);
  assert.equal(
    await page.locator("[data-cli-explorer]").getAttribute("data-orientation"),
    alternateOrientation,
    `${viewportName}: explicit orientation choice did not apply before resize`
  );
  await page.setViewportSize({
    width: viewportName === "mobile" ? 1440 : 390,
    height: viewportName === "mobile" ? 1000 : 844,
  });
  await page.waitForTimeout(120);
  assert.equal(
    await page.locator("[data-cli-explorer]").getAttribute("data-orientation"),
    alternateOrientation,
    `${viewportName}: explicit ${alternateOrientation} choice was lost across resize`
  );
  await page.setViewportSize(original);
  await page.waitForTimeout(120);
  assert.equal(
    await page.locator("[data-cli-explorer]").getAttribute("data-orientation"),
    alternateOrientation,
    `${viewportName}: explicit ${alternateOrientation} choice was lost after restoring viewport`
  );
  await page.locator(`[data-action="orientation"][data-orientation="${initialOrientation}"]`).first().click();
  await page.waitForTimeout(100);
  assert.equal(
    await page.locator("[data-cli-explorer]").getAttribute("data-orientation"),
    initialOrientation,
    `${viewportName}: failed to restore initial orientation after resize persistence check`
  );
}

async function assertOrientationBehavior(page, viewportName, search) {
  await assertOrientationControl(page, viewportName);
  const command = page.locator('g.cli-explorer-node.command[role="treeitem"]').first();
  const commandPath = await command.getAttribute("data-path");
  assert.ok(commandPath, `${viewportName}: orientation fixture command path is missing`);
  await command.focus();
  await page.keyboard.press("Enter");
  await page.waitForTimeout(80);
  const selectedBefore = await page.locator('g.cli-explorer-node[aria-selected="true"]').getAttribute("data-path");
  const expandedBefore = await command.getAttribute("aria-expanded");
  assert.equal(selectedBefore, commandPath, `${viewportName}: orientation fixture failed to select command`);
  assert.equal(expandedBefore, "true", `${viewportName}: orientation fixture command did not expand`);
  await search.fill("model");

  const initialOrientation = viewportName === "mobile" ? "vertical" : "horizontal";
  const alternateOrientation = initialOrientation === "vertical" ? "horizontal" : "vertical";
  const switchTo = page.locator(`[data-action="orientation"][data-orientation="${alternateOrientation}"]`).first();
  await switchTo.click();
  await page.waitForTimeout(100);
  const alternate = await orientationSnapshot(page);
  assert.equal(alternate.orientation, alternateOrientation, `${viewportName}: orientation did not switch to ${alternateOrientation}`);
  assert.equal(await switchTo.getAttribute("aria-pressed"), "true", `${viewportName}: ${alternateOrientation} pressed state did not update`);
  assert.equal(await search.inputValue(), "model", `${viewportName}: orientation switch discarded search text`);
  assert.equal(alternate.selectedPath, commandPath, `${viewportName}: orientation switch discarded selected command`);
  assert.equal(await page.locator(`g.cli-explorer-node[data-path="${commandPath}"]`).getAttribute("aria-expanded"), expandedBefore, `${viewportName}: orientation switch changed expansion state`);
  assert.ok(alternate.root?.transform && alternate.firstChild?.transform, `${viewportName}: ${alternateOrientation} lacks root/child transforms`);
  if (alternateOrientation === "vertical") {
    assert.ok(alternate.firstChild.transform.y > alternate.root.transform.y, `${viewportName}: vertical parent/child axis does not advance downward`);
    assert.ok(alternate.buses.some((bus) => bus.className.includes("vertical-bus") && /V[^H]*M[^H]*H/.test(bus.d)), `${viewportName}: vertical bus lacks stem/trunk grammar`);
    assert.ok(alternate.links.some((link) => /V/.test(link.d)), `${viewportName}: vertical child drops are missing`);
  } else {
    assert.ok(alternate.firstChild.transform.x > alternate.root.transform.x, `${viewportName}: horizontal parent/child axis does not advance rightward`);
    assert.ok(alternate.buses.some((bus) => /H[^M]*M[^V]*V/.test(bus.d) && bus.strokeWidth >= 1), `${viewportName}: horizontal bus lacks trunk/drop grammar`);
    assert.ok(alternate.links.some((link) => /H/.test(link.d)), `${viewportName}: horizontal child links are missing`);
  }
  await assertConnectorWeights(page, viewportName, alternateOrientation);
  await assertRoundedOrthogonalPaths(page, viewportName, alternateOrientation);
  await assertLinkGradients(page, viewportName, alternateOrientation);
  await assertNoPageOverflow(page);
  await assertNoNodeCollisions(page);

  const switchBack = page.locator(`[data-action="orientation"][data-orientation="${initialOrientation}"]`).first();
  await switchBack.click();
  await page.waitForTimeout(100);
  const restored = await orientationSnapshot(page);
  assert.equal(restored.orientation, initialOrientation, `${viewportName}: orientation did not switch back to ${initialOrientation}`);
  assert.equal(await switchBack.getAttribute("aria-pressed"), "true", `${viewportName}: ${initialOrientation} pressed state did not restore`);
  assert.equal(await search.inputValue(), "model", `${viewportName}: switching back discarded search text`);
  assert.equal(restored.selectedPath, commandPath, `${viewportName}: switching back discarded selected command`);
  assert.equal(await page.locator(`g.cli-explorer-node[data-path="${commandPath}"]`).getAttribute("aria-expanded"), expandedBefore, `${viewportName}: switching back changed expansion state`);
  await assertConnectorWeights(page, viewportName, initialOrientation);
  await assertRoundedOrthogonalPaths(page, viewportName, initialOrientation);
  await assertLinkGradients(page, viewportName, initialOrientation);
  await search.fill("");
  await page.waitForTimeout(80);
  await assertLinkGradients(page, viewportName, initialOrientation);
  await command.focus();
  await page.keyboard.press("Enter");
  await page.waitForTimeout(80);
  await assertLinkGradients(page, viewportName, initialOrientation);
  await command.focus();
  await page.keyboard.press("Enter");
  await page.waitForTimeout(80);
  await assertLinkGradients(page, viewportName, initialOrientation);
}

async function assertConnectorWeights(page, viewportName, expectedOrientation) {
  const metrics = await page.evaluate(() => {
    const app = document.querySelector("[data-cli-explorer]");
    const links = [...document.querySelectorAll(".cli-explorer-link")]
      .filter((link) => Number(getComputedStyle(link).opacity) > 0.05)
      .map((link) => ({
        className: link.getAttribute("class") || "",
        strokeWidth: Number.parseFloat(getComputedStyle(link).strokeWidth),
      }));
    const buses = [...document.querySelectorAll(".cli-explorer-root-bus, .cli-explorer-mobile-branch-bus, .cli-explorer-branch-bus, .cli-explorer-vertical-bus")]
      .filter((bus) => Number(getComputedStyle(bus).opacity) > 0.05)
      .map((bus) => ({
        className: bus.getAttribute("class") || "",
        strokeWidth: Number.parseFloat(getComputedStyle(bus).strokeWidth),
      }));
    return {
      orientation: app?.dataset.orientation || "",
      links,
      buses,
    };
  });
  assert.equal(metrics.orientation, expectedOrientation, `${viewportName}: connector measurement used the wrong orientation`);
  const connectors = [...metrics.links, ...metrics.buses];
  assert.ok(connectors.length > 0, `${viewportName}: no visible connectors to measure`);
  assert.ok(
    connectors.every(({ strokeWidth }) => Number.isFinite(strokeWidth) && Math.abs(strokeWidth - 2) <= 0.05),
    `${viewportName}: every visible connector must use a 2px non-scaling stroke (${connectors.map(({ className, strokeWidth }) => `${className}:${strokeWidth}`).join(", ")})`
  );
}

async function assertRoundedOrthogonalPaths(page, viewportName, expectedOrientation) {
  const snapshot = await orientationSnapshot(page);
  const pathData = [
    ...snapshot.links.map((link) => link.d),
    ...snapshot.buses.map((bus) => bus.d),
  ];
  assert.ok(pathData.length > 0, `${viewportName}: no connector paths to inspect for rounded elbows`);
  assert.ok(
    pathData.every((path) => path && !/NaN|Infinity/.test(path)),
    `${viewportName}: connector path data contains an invalid number`
  );
  assert.ok(pathData.some((path) => /Q|A/.test(path)), `${viewportName}: no connector path contains a rounded elbow command`);
  assert.ok(pathData.some((path) => /H/.test(path)), `${viewportName}: connector paths lost horizontal runs`);
  assert.ok(pathData.some((path) => /V/.test(path)), `${viewportName}: connector paths lost vertical runs`);
  if (expectedOrientation === "horizontal") {
    assert.ok(
      snapshot.links.some((link) => /Q/.test(link.d) && /H/.test(link.d)),
      `${viewportName}: horizontal outer child links lack rounded Q→H elbows`
    );
  } else {
    assert.ok(
      snapshot.links.some((link) => /Q/.test(link.d) && /V/.test(link.d)),
      `${viewportName}: vertical outer child links lack rounded Q→V elbows`
    );
  }
}

async function assertLinkGradients(page, viewportName, expectedOrientation) {
  const expectedColors = {
    command: "#94a3b8",
    option: "#f59e0b",
    positional: "#a78bfa",
  };
  const metrics = await page.evaluate(() => {
    const app = document.querySelector("[data-cli-explorer]");
    const gradients = new Map(
      [...document.querySelectorAll("defs linearGradient.cli-explorer-link-gradient")]
        .map((gradient) => [gradient.id, {
          id: gradient.id,
          units: gradient.getAttribute("gradientUnits") || "",
          x1: Number(gradient.getAttribute("x1")),
          y1: Number(gradient.getAttribute("y1")),
          x2: Number(gradient.getAttribute("x2")),
          y2: Number(gradient.getAttribute("y2")),
          targetKind: gradient.getAttribute("data-target-kind") || "",
          stops: [...gradient.querySelectorAll("stop")].map((stop) => ({
            offset: Number.parseFloat(stop.getAttribute("offset") || "NaN"),
            color: (stop.getAttribute("stop-color") || "").toLowerCase(),
            opacity: Number.parseFloat(stop.getAttribute("stop-opacity") || "NaN"),
          })),
        }])
    );
    const links = [...document.querySelectorAll(".cli-explorer-link")]
      .filter((link) => Number(getComputedStyle(link).opacity) > 0.05)
      .map((link) => ({
        id: link.getAttribute("data-gradient-id") || "",
        stroke: link.getAttribute("stroke") || "",
        gradient: gradients.get(link.getAttribute("data-gradient-id") || "") || null,
      }));
    return {
      orientation: app?.dataset.orientation || "",
      links,
      gradients: [...gradients.values()],
    };
  });
  assert.equal(metrics.orientation, expectedOrientation, `${viewportName}: gradient measurement used the wrong orientation`);
  assert.ok(metrics.links.length > 0, `${viewportName}: no visible links to measure gradients`);
  assert.equal(
    metrics.gradients.length,
    metrics.links.length,
    `${viewportName}: defs contain stale or duplicate gradients (${metrics.gradients.length} defs for ${metrics.links.length} links)`
  );
  const ids = new Set(metrics.gradients.map((gradient) => gradient.id));
  assert.equal(ids.size, metrics.gradients.length, `${viewportName}: gradient ids are duplicated`);
  for (const link of metrics.links) {
    assert.ok(link.id && link.gradient, `${viewportName}: visible link does not reference an existing gradient`);
    assert.equal(link.stroke, `url(#${link.id})`, `${viewportName}: visible link stroke is not bound to its gradient`);
    const gradient = link.gradient;
    assert.equal(gradient.units, "userSpaceOnUse", `${viewportName}: link gradient is not user-space anchored`);
    assert.equal(gradient.stops.length, 3, `${viewportName}: link gradient does not have source/blend/padding stops`);
    assert.ok(
      [gradient.x1, gradient.y1, gradient.x2, gradient.y2].every(Number.isFinite),
      `${viewportName}: link gradient coordinates are invalid`
    );
    assert.equal(gradient.stops[0].offset, 0, `${viewportName}: gradient source stop is not at 0%`);
    assert.equal(gradient.stops[0].color, "#60a5fa", `${viewportName}: gradient source stop is not bus blue`);
    const axisDistance = expectedOrientation === "horizontal"
      ? gradient.x2 - gradient.x1
      : gradient.y2 - gradient.y1;
    const crossDistance = expectedOrientation === "horizontal"
      ? Math.abs(gradient.y2 - gradient.y1)
      : Math.abs(gradient.x2 - gradient.x1);
    assert.ok(axisDistance > 0, `${viewportName}: ${expectedOrientation} gradient does not advance along its axis`);
    assert.equal(crossDistance, 0, `${viewportName}: ${expectedOrientation} gradient drifts across its axis`);
    assert.ok(Math.abs(axisDistance - 32) < 0.01, `${viewportName}: gradient blend distance is not 32 user units (${axisDistance})`);
    const targetStop = gradient.stops[1];
    const targetDistance = targetStop.offset / 100 * axisDistance;
    assert.ok(targetDistance >= 27 && targetDistance <= 32.5, `${viewportName}: semantic color is reached outside the 28–32 unit blend (${targetDistance})`);
    assert.equal(targetStop.color, expectedColors[gradient.targetKind], `${viewportName}: gradient target color does not match ${gradient.targetKind}`);
    assert.equal(gradient.stops[2].color, targetStop.color, `${viewportName}: gradient padding stop changed semantic color`);
  }
}

async function assertLegendDoesNotOverlapTree(page, viewportName, stateName) {
  if (viewportName !== "mobile") return;
  const legend = page.locator(".cli-explorer-legend").first();
  const legendBox = await legend.boundingBox();
  assert.ok(legendBox, `${viewportName}: legend bounds are unavailable in ${stateName}`);
  const nodes = await visibleTreeItemData(page);
  const collision = nodes.find((node) => boxesOverlap(legendBox, node));
  assert.equal(
    collision,
    undefined,
    `${viewportName}: legend overlaps visible treeitem ${collision?.path || "(unknown)"} in ${stateName}`
  );
}

async function assertInventoryContract(page, viewportName, search) {
  const payload = JSON.parse(await page.locator("#cli-inventory-data").textContent());
  const walk = (node, ancestors = []) => [
    { node, ancestors },
    ...(node.children || []).flatMap((child) => walk(child, [...ancestors, node.name])),
  ];
  const entries = walk(payload.root);
  const external = entries.find(({ node }) => node.external === true)?.node;
  assert.ok(external, `${viewportName}: inventory is missing the external plugin catch-all`);
  assert.match(external.name, /PLUGIN_COMMAND\s+\.\.\./, `${viewportName}: external catch-all has no pattern text`);

  const internalPlugin = entries.find(({ node, ancestors }) =>
    ancestors.includes("serve")
      || ancestors.includes("client")
      ? node.name === "--plugin" || node.long === "plugin"
      : false
  );
  assert.equal(internalPlugin, undefined, `${viewportName}: internal --plugin leaked into serve/client inventory`);

  await search.fill("PLUGIN_COMMAND");
  const results = page.locator('[role="option"]');
  await results.first().waitFor({ state: "visible" });
  const result = results.filter({ hasText: "PLUGIN_COMMAND" }).first();
  assert.ok(await result.count(), `${viewportName}: external catch-all is not searchable by its pattern`);
  await result.click();
  const inspector = page.locator("[data-inspector]").first();
  await inspector.waitFor({ state: "visible" });
  const inspectorText = await inspector.innerText();
  assert.match(inspectorText, /pattern only/i, `${viewportName}: external inspector omits pattern guidance`);
  assert.equal(await inspector.locator("[data-copy]").count(), 0, `${viewportName}: external catch-all exposes a copy button`);
  await search.fill("");
  await page.waitForTimeout(80);
}

async function assertDesktopOverview(page, viewportName) {
  if (viewportName !== "desktop") return;
  const appBox = await page.locator(".cli-explorer-app").boundingBox();
  const viewport = page.viewportSize();
  assert.ok(appBox && viewport, `${viewportName}: explorer or viewport bounds are unavailable`);
  assert.ok(
    appBox.height >= viewport.height - 120,
    `${viewportName}: explorer does not use the available viewport height (${appBox.height}px of ${viewport.height}px)`
  );
  const scale = await page.locator('svg[data-tree]').evaluate((svg) => window.d3.zoomTransform(svg).k);
  assert.ok(scale >= 0.899, `${viewportName}: initial overview scale fell below the 0.9 readability floor (${scale})`);

  const canvas = await page.locator(".cli-explorer-canvas-wrap").boundingBox();
  const root = page.locator('g.cli-explorer-node.root[role="treeitem"]').first();
  const rootBox = await root.boundingBox();
  assert.ok(rootBox && canvas, `${viewportName}: root or canvas bounds are unavailable`);
  assert.ok(
    rootBox.x < canvas.x + canvas.width
      && rootBox.x + rootBox.width > canvas.x
      && rootBox.y < canvas.y + canvas.height
      && rootBox.y + rootBox.height > canvas.y,
    `${viewportName}: root is outside the initial viewport`
  );
  const rootLabel = await root.locator("text:not(.cli-explorer-node-count)").first().textContent();
  assert.ok(rootLabel.trim().length > 0, `${viewportName}: root label is empty`);

  const bus = page.locator("path.cli-explorer-root-bus").first();
  assert.equal(await bus.count(), 1, `${viewportName}: root bus is missing`);
  const busMetrics = await bus.evaluate((element) => {
    const box = element.getBoundingClientRect();
    return { x: box.x, y: box.y, width: box.width, height: box.height };
  });
  assert.ok(
    (busMetrics.width > 1 || busMetrics.height > 1)
      && busMetrics.x < canvas.x + canvas.width
      && busMetrics.x + busMetrics.width > canvas.x
      && busMetrics.y < canvas.y + canvas.height
      && busMetrics.y + busMetrics.height > canvas.y,
    `${viewportName}: root bus is not visible in the initial viewport`
  );

  const readableLabels = await page.locator('g.cli-explorer-node[role="treeitem"] text:not(.cli-explorer-node-count)').evaluateAll((labels) =>
    labels
      .map((label) => {
        const box = label.getBoundingClientRect();
        const style = getComputedStyle(label);
        return {
          text: label.textContent?.trim() || "",
          fontSize: Number.parseFloat(style.fontSize),
          visible: style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0.05,
          width: box.width,
          height: box.height,
        };
      })
      .filter((label) => label.visible && label.width > 1 && label.height > 1)
  );
  assert.ok(readableLabels.length >= 2, `${viewportName}: fewer than two readable initial node labels`);
  assert.ok(readableLabels.every((label) => label.text && label.fontSize >= 10), `${viewportName}: initial node labels are not readable at the 0.9 floor`);
}

async function assertIntroAlignment(page, viewportName) {
  if (viewportName !== "desktop") return;
  const original = page.viewportSize();
  assert.ok(original, `${viewportName}: viewport size is unavailable for intro alignment testing`);
  for (const width of [original.width, 2200]) {
    await page.setViewportSize({ width, height: original.height });
    await page.waitForTimeout(120);
    const edges = await page.evaluate(() => {
      const intro = document.querySelector(".cli-explorer-intro")?.getBoundingClientRect();
      const app = document.querySelector(".cli-explorer-app")?.getBoundingClientRect();
      return { introLeft: intro?.left ?? null, appLeft: app?.left ?? null };
    });
    assert.ok(edges.introLeft !== null && edges.appLeft !== null, `${viewportName}: intro/workspace bounds are unavailable at ${width}px`);
    assert.ok(
      Math.abs(edges.introLeft - edges.appLeft) <= 2,
      `${viewportName}: intro/workspace left edges drift by ${Math.abs(edges.introLeft - edges.appLeft).toFixed(1)}px at ${width}px`
    );
  }
  await page.setViewportSize(original);
  await page.waitForTimeout(120);
}

async function assertBlankCanvasZoom(page, viewportName) {
  const viewport = page.viewportSize();
  const canvas = await page.locator(".cli-explorer-canvas-wrap").boundingBox();
  assert.ok(canvas && viewport, `${viewportName}: canvas or viewport bounds are unavailable for zoom testing`);
  const svg = page.locator("svg[data-tree]");
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(40);
  const point = {
    x: canvas.x + canvas.width - 20,
    y: Math.min(canvas.y + canvas.height - 20, viewport.height - 20),
  };
  assert.ok(point.y > canvas.y, `${viewportName}: no visible blank canvas point is available for zoom testing`);
  const before = await svg.evaluate((element) => window.d3.zoomTransform(element).k);
  const pageScrollBefore = await page.evaluate(() => window.scrollY);
  await page.mouse.move(point.x, point.y);
  await page.mouse.wheel(0, 240);
  await page.waitForTimeout(100);
  const after = await svg.evaluate((element) => window.d3.zoomTransform(element).k);
  const pageScrollAfter = await page.evaluate(() => window.scrollY);
  assert.ok(after < before, `${viewportName}: wheel zoom does not work over blank canvas wrapper point (${before} → ${after})`);
  assert.equal(pageScrollAfter, pageScrollBefore, `${viewportName}: blank canvas wheel input scrolled the document`);

  const legend = await page.locator(".cli-explorer-legend").boundingBox();
  assert.ok(legend, `${viewportName}: legend bounds are unavailable for wrapper wheel testing`);
  const overlayPoint = { x: legend.x + legend.width / 2, y: legend.y + legend.height / 2 };
  const overlayBefore = after;
  await page.mouse.move(overlayPoint.x, overlayPoint.y);
  await page.mouse.wheel(0, 120);
  await page.waitForTimeout(100);
  const overlayAfter = await svg.evaluate((element) => window.d3.zoomTransform(element).k);
  assert.ok(overlayAfter < overlayBefore, `${viewportName}: wheel over the legend/status overlay did not zoom`);
  assert.equal(await page.evaluate(() => window.scrollY), pageScrollBefore, `${viewportName}: overlay wheel input scrolled the document`);

  await page.mouse.wheel(0, 10000);
  await page.waitForTimeout(100);
  const scaleAtLimit = await svg.evaluate((element) => window.d3.zoomTransform(element).k);
  const scrollAtLimit = await page.evaluate(() => window.scrollY);
  await page.mouse.wheel(0, 600);
  await page.waitForTimeout(100);
  const scalePastLimit = await svg.evaluate((element) => window.d3.zoomTransform(element).k);
  const scrollPastLimit = await page.evaluate(() => window.scrollY);
  assert.ok(Math.abs(scalePastLimit - scaleAtLimit) < 0.001, `${viewportName}: scale escaped its lower zoom limit`);
  assert.equal(scrollPastLimit, scrollAtLimit, `${viewportName}: wheel input leaked to the page at the canvas zoom limit`);

  await page.mouse.wheel(0, -10000);
  await page.waitForTimeout(100);
  const scaleAtUpperLimit = await svg.evaluate((element) => window.d3.zoomTransform(element).k);
  assert.ok(scaleAtUpperLimit > 1.99, `${viewportName}: wheel input did not reach its upper zoom limit (${scaleAtUpperLimit})`);
  const scrollAtUpperLimit = await page.evaluate(() => window.scrollY);
  await page.mouse.wheel(0, -600);
  await page.waitForTimeout(100);
  const scalePastUpperLimit = await svg.evaluate((element) => window.d3.zoomTransform(element).k);
  assert.ok(Math.abs(scalePastUpperLimit - scaleAtUpperLimit) < 0.001, `${viewportName}: scale escaped its upper zoom limit`);
  assert.equal(await page.evaluate(() => window.scrollY), scrollAtUpperLimit, `${viewportName}: wheel input leaked to the page at the upper canvas zoom limit`);

  await page.evaluate(() => window.scrollTo(0, 0));
  const intro = await page.locator(".cli-explorer-intro").boundingBox();
  assert.ok(intro, `${viewportName}: intro bounds are unavailable for outside-canvas wheel testing`);
  await page.mouse.move(intro.x + Math.min(24, intro.width / 2), intro.y + Math.min(24, intro.height / 2));
  const outsideBefore = await page.evaluate(() => window.scrollY);
  await page.mouse.wheel(0, 500);
  await page.waitForTimeout(100);
  const outsideAfter = await page.evaluate(() => window.scrollY);
  assert.ok(outsideAfter > outsideBefore, `${viewportName}: wheel outside the canvas did not scroll the document`);

  await page.evaluate(() => window.scrollTo(0, 0));
  await page.locator('[data-action="fit"]').click();
  await page.waitForTimeout(100);
}

async function assertLinksClearSourceNodes(page, viewportName) {
  if (viewportName !== "desktop") return;
  const metrics = await page.locator("svg[data-tree]").evaluate((svg) => {
    const scale = window.d3.zoomTransform(svg).k;
    const shells = [...svg.querySelectorAll(".cli-explorer-node-shell")].map((shell) => {
      const box = shell.getBoundingClientRect();
      return { left: box.left, right: box.right, centerY: box.top + box.height / 2 };
    });
    const link = svg.querySelector(".cli-explorer-link");
    const linkWeight = link ? Number.parseFloat(getComputedStyle(link).strokeWidth) : 0;
    return [...svg.querySelectorAll(".cli-explorer-branch-bus")].flatMap((path) => {
      const matrix = path.getScreenCTM();
      if (!matrix || path.getTotalLength() === 0) return [];
      const point = path.getPointAtLength(0);
      const start = new DOMPoint(point.x, point.y).matrixTransform(matrix);
      const busBox = path.getBoundingClientRect();
      const source = shells
        .filter((shell) => Math.abs(shell.centerY - start.y) <= 2 && shell.right <= start.x + 1)
        .sort((left, right) => right.right - left.right)[0];
      const children = shells.filter((shell) =>
        shell.left >= busBox.right
          && shell.centerY >= busBox.top - 1
          && shell.centerY <= busBox.bottom + 1
      );
      if (!source || !children.length) return [];
      return [{
        sourceGap: (start.x - source.right) / scale,
        childGap: Math.min(...children.map((shell) => shell.left - busBox.right)) / scale,
        busWeight: Number.parseFloat(getComputedStyle(path).strokeWidth),
        linkWeight,
      }];
    });
  });
  assert.ok(metrics.length > 0, `${viewportName}: no expanded branch-bus spacing was measurable`);
  assert.ok(
    metrics.every(({ sourceGap }) => sourceGap >= 7),
    `${viewportName}: a branch bus is too close to its source node (${metrics.map(({ sourceGap }) => sourceGap).join(", ")})`
  );
  assert.ok(
    metrics.every(({ childGap }) => childGap >= 28),
    `${viewportName}: child nodes are too close to their branch bus (${metrics.map(({ childGap }) => childGap).join(", ")})`
  );
  assert.ok(
    metrics.every(({ busWeight, linkWeight }) => Math.abs(busWeight - 2) <= 0.05 && Math.abs(linkWeight - 2) <= 0.05),
    `${viewportName}: branch bus/link tangents must share the 2px connector width`
  );
}

async function assertMobileToggleText(page, viewportName) {
  if (viewportName !== "mobile") return;
  for (const [action, expected] of [["root-options", "Root flags"], ["hidden", "Hidden"]]) {
    const toggle = page.locator(`[data-action="${action}"]`).first();
    const label = toggle.locator("span").first();
    assert.ok(await toggle.isVisible(), `${viewportName}: ${expected} toggle is not visible`);
    assert.ok(await label.isVisible(), `${viewportName}: ${expected} toggle text is hidden`);
    assert.equal((await label.innerText()).trim(), expected, `${viewportName}: ${expected} toggle text changed`);
    const metrics = await label.evaluate((element) => {
      const box = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      return { display: style.display, visibility: style.visibility, width: box.width, height: box.height };
    });
    assert.ok(metrics.display !== "none" && metrics.visibility !== "hidden" && metrics.width > 1 && metrics.height > 1, `${viewportName}: ${expected} toggle text has no visible box`);
  }
}

async function assertDisclosureChevrons(page, viewportName) {
  const chevrons = await page.locator('g.cli-explorer-node[role="treeitem"][aria-expanded]').evaluateAll((nodes) =>
    nodes
      .map((node) => {
        const style = getComputedStyle(node);
        const box = node.getBoundingClientRect();
        const path = node.querySelector(".cli-explorer-node-disclosure")?.getAttribute("d") || "";
        const points = [...path.matchAll(/[ML](-?[0-9.]+),(-?[0-9.]+)/g)].map((match) => ({ x: Number(match[1]), y: Number(match[2]) }));
        return {
          expanded: node.getAttribute("aria-expanded") === "true",
          path,
          points,
          visible: style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0.05 && box.width > 1 && box.height > 1,
        };
      })
      .filter((item) => item.visible)
  );
  assert.ok(chevrons.length > 0, `${viewportName}: no visible expandable treeitems`);
  for (const chevron of chevrons) {
    assert.equal(chevron.points.length, 3, `${viewportName}: malformed disclosure chevron path (${chevron.path})`);
    const [first, middle, last] = chevron.points;
    if (chevron.expanded) {
      assert.ok(Math.abs(first.y - last.y) < 0.1 && middle.y > first.y, `${viewportName}: expanded disclosure chevron is not oriented downward`);
    } else {
      assert.ok(Math.abs(first.x - last.x) < 0.1 && middle.x > first.x, `${viewportName}: collapsed disclosure chevron is not oriented rightward`);
    }
  }
}

async function assertCountsClearDisclosures(page, viewportName) {
  const gaps = await page.locator('g.cli-explorer-node[role="treeitem"][aria-expanded]').evaluateAll((nodes) =>
    nodes
      .map((node) => {
        const count = node.querySelector(".cli-explorer-node-count");
        const disclosure = node.querySelector(".cli-explorer-node-disclosure");
        if (!count?.textContent?.trim() || !disclosure) return null;
        const countBox = count.getBoundingClientRect();
        const disclosureBox = disclosure.getBoundingClientRect();
        const shellBox = node.querySelector(".cli-explorer-node-shell")?.getBoundingClientRect();
        if (!shellBox) return null;
        return {
          gap: disclosureBox.left - countBox.right,
          verticalOffset: Math.abs(
            (disclosureBox.top + disclosureBox.height / 2)
              - (shellBox.top + shellBox.height / 2)
          ),
        };
      })
      .filter((metrics) => metrics !== null)
  );
  assert.ok(gaps.length > 0, `${viewportName}: no count/disclosure pairs were available`);
  assert.ok(
    gaps.every(({ gap }) => gap >= 6),
    `${viewportName}: a disclosure icon overlaps or crowds its numeric count (${gaps.map(({ gap }) => gap).join(", ")})`
  );
  assert.ok(
    gaps.every(({ verticalOffset }) => verticalOffset <= 1),
    `${viewportName}: a disclosure icon is not vertically centered (${gaps.map(({ verticalOffset }) => verticalOffset).join(", ")})`
  );
}

async function assertTreeKeyboard(page, viewportName) {
  const root = page.locator('g.cli-explorer-node.root[role="treeitem"]').first();
  const rootPath = await root.getAttribute("data-path");
  const firstCommand = page.locator('g.cli-explorer-node.command[role="treeitem"]').first();
  const firstCommandPath = await firstCommand.getAttribute("data-path");
  assert.ok(rootPath && firstCommandPath, `${viewportName}: initial root/command paths are missing`);

  await root.focus();
  await page.keyboard.press("ArrowDown");
  assert.equal(await activeTreePath(page), firstCommandPath, `${viewportName}: ArrowDown did not move to the first command`);
  await page.keyboard.press("Home");
  assert.equal(await activeTreePath(page), rootPath, `${viewportName}: Home did not move to the root`);
  await page.keyboard.press("End");
  const visibleAfterEnd = await visibleTreeItemData(page);
  assert.equal(await activeTreePath(page), visibleAfterEnd.at(-1)?.path, `${viewportName}: End did not move to the last visible node`);

  let collapsedCommand = null;
  for (let index = 0; index < await page.locator('g.cli-explorer-node.command[role="treeitem"][aria-expanded="false"]').count(); index += 1) {
    const candidate = page.locator('g.cli-explorer-node.command[role="treeitem"][aria-expanded="false"]').nth(index);
    if (await candidate.isVisible()) {
      collapsedCommand = candidate;
      break;
    }
  }
  assert.ok(collapsedCommand, `${viewportName}: no collapsed command available for ArrowRight`);
  const collapsedPath = await collapsedCommand.getAttribute("data-path");
  const collapsedLevel = Number(await collapsedCommand.getAttribute("aria-level"));
  await collapsedCommand.focus();
  await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(60);
  const commandByPath = page.locator(`g.cli-explorer-node[role="treeitem"][data-path="${collapsedPath}"]`).last();
  assert.equal(await commandByPath.getAttribute("aria-expanded"), "true", `${viewportName}: ArrowRight did not expand a command`);
  assert.equal(await activeTreePath(page), collapsedPath, `${viewportName}: ArrowRight moved focus before expansion completed`);

  const expandedChildren = (await visibleTreeItemData(page)).filter((item) =>
    item.path.startsWith(`${collapsedPath} `) && item.level === collapsedLevel + 1
  );
  assert.ok(expandedChildren.length > 0, `${viewportName}: expanded command has no visible child`);
  await assertLegendDoesNotOverlapTree(page, viewportName, "expanded/focused command branch");
  await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(60);
  assert.equal(await activeTreePath(page), expandedChildren[0].path, `${viewportName}: second ArrowRight did not focus the first child`);

  await page.keyboard.press("ArrowLeft");
  await page.waitForTimeout(60);
  assert.equal(await activeTreePath(page), collapsedPath, `${viewportName}: ArrowLeft from child did not focus its parent`);
  await page.keyboard.press("ArrowLeft");
  await page.waitForTimeout(60);
  assert.equal(await commandByPath.getAttribute("aria-expanded"), "false", `${viewportName}: ArrowLeft did not collapse the expanded command`);
  assert.equal(await activeTreePath(page), collapsedPath, `${viewportName}: ArrowLeft moved focus while collapsing`);
  await page.keyboard.press("ArrowLeft");
  await page.waitForTimeout(60);
  assert.equal(await activeTreePath(page), rootPath, `${viewportName}: second ArrowLeft did not focus the parent root`);
}

async function assertSearchClearFocus(page, viewportName, search) {
  await search.fill("");
  await page.waitForTimeout(80);
  const visible = await visibleTreeItemData(page);
  const tabbable = visible.filter((item) => item.tabindex === "0");
  assert.equal(tabbable.length, 1, `${viewportName}: clearing search left ${tabbable.length} visible tabbable treeitems`);
  const unique = page.locator('g.cli-explorer-node[role="treeitem"][tabindex="0"]').first();
  await unique.focus();
  const beforeTab = await activeTreePath(page);
  await page.keyboard.press("Tab");
  assert.notEqual(await activeTreePath(page), beforeTab, `${viewportName}: Tab did not leave the unique treeitem`);
  await unique.focus();
  await page.keyboard.press("Home");
  const homePath = await activeTreePath(page);
  await page.keyboard.press("ArrowDown");
  assert.notEqual(await activeTreePath(page), homePath, `${viewportName}: ArrowDown did not move focus after clearing search`);
}

async function runExplorerAssertions(page, baseURL, viewportName) {
  const consoleErrors = [];
  const pageErrors = [];
  const onConsole = (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  };
  const onPageError = (error) => pageErrors.push(error.message);
  page.on("console", onConsole);
  page.on("pageerror", onPageError);

  try {
    await page.goto(`${baseURL}${explorerPath}`, { waitUntil: "networkidle" });
    await requireVisible(
      await firstAvailable(page, ["main.docs-shell", "main.cli-explorer-page"]),
      `${viewportName}: docs shell is missing`
    );
    await requireVisible(
      await firstAvailable(page, ["article.doc", "main.cli-explorer-page[data-pagefind-body]", "[data-cli-explorer]"]),
      `${viewportName}: docs article is missing`
    );
    await requireVisible(
      await firstAvailable(page, [
        '[role="tree"]',
        '[role="group"][aria-label*="command" i]',
        'svg#tree',
        '[data-cli-explorer]',
      ]),
      `${viewportName}: interactive CLI tree is missing`
    );
    mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({
      path: path.join(screenshotDirectory, `${viewportName}.png`),
      fullPage: false,
    });
    await assertDesktopOverview(page, viewportName);
    await assertIntroAlignment(page, viewportName);
    await assertBlankCanvasZoom(page, viewportName);
    await assertMobileToggleText(page, viewportName);
    await assertLegendDoesNotOverlapTree(page, viewportName, "initial overview");
    await assertDisclosureChevrons(page, viewportName);
    await assertCountsClearDisclosures(page, viewportName);
    await assertTreeKeyboard(page, viewportName);

    const search = await requireVisible(
      await firstAvailable(page, [
        'input[type="search"][aria-label*="CLI" i]',
        'input[type="search"]',
      ]),
      `${viewportName}: CLI search input is missing`
    );
    await assertInventoryContract(page, viewportName, search);
    const searchLabel = await search.evaluate((input) => {
      const labelledBy = input.getAttribute("aria-labelledby");
      const label = labelledBy ? document.getElementById(labelledBy) : document.querySelector(`label[for="${input.id}"]`);
      return input.getAttribute("aria-label") || label?.textContent?.trim() || "";
    });
    assert.match(searchLabel, /search.*(cli|command|option)|cli.*search/i, `${viewportName}: search input needs an accessible CLI label`);

    await assertResponsiveOrientationDefault(page, viewportName);
    await assertExplicitOrientationAcrossResize(page, viewportName);
    await assertOrientationBehavior(page, viewportName, search);
    await page.evaluate(() => document.activeElement?.blur());
    await page.keyboard.press("/");
    assert.equal(await page.evaluate(() => document.activeElement?.matches('input[type="search"]')), true, `${viewportName}: / does not focus search`);
    await search.fill("model");
    const results = page.locator('[role="option"]');
    await results.first().waitFor({ state: "visible" });
    const resultDataPath = await results.first().getAttribute("data-path");
    const resultPath = resultDataPath
      || await results.first().locator(".cli-explorer-result-path").innerText().catch(() => "");
    const resultText = (await results.first().innerText()).trim();
    await results.first().click();

    const selected = page.locator('[aria-selected="true"], [data-selected="true"]');
    if (await selected.count()) {
      assert.ok(await selected.first().isVisible(), `${viewportName}: search result did not select a visible node`);
      if (resultDataPath) {
        assert.equal(await selected.first().getAttribute("data-path"), resultDataPath, `${viewportName}: selected node lineage differs from search result`);
      }
    }
    const inspector = await requireVisible(
      await firstAvailable(page, [
        '[aria-label*="Selected CLI item" i]',
        '[data-cli-inspector]',
        "#inspector",
      ]),
      `${viewportName}: selected item inspector is missing`
    );
    const inspectorText = await inspector.innerText();
    assert.ok(inspectorText.length > 0, `${viewportName}: selected item inspector is empty`);
    const resultToken = resultText.match(/--?[A-Za-z0-9][A-Za-z0-9-]*/)?.[0]
      || resultText.split(/\s+/)[0];
    const lineageTokens = resultPath.split(/\s+/).filter(Boolean).slice(0, -1);
    assert.ok(
      inspectorText.toLowerCase().includes(resultToken.toLowerCase())
        && lineageTokens.every((token) => inspectorText.includes(token)),
      `${viewportName}: inspector does not describe the selected search result (${JSON.stringify({ resultText, resultPath, inspectorText })})`
    );
    await assertSearchClearFocus(page, viewportName, search);

    const initialNodes = await visibleNodeCount(page);
    const commandCandidates = page.locator('g.cli-explorer-node.command[role="treeitem"][aria-expanded]');
    let command = null;
    for (let index = 0; index < await commandCandidates.count(); index += 1) {
      const candidate = commandCandidates.nth(index);
      if (await candidate.isVisible()) {
        command = candidate;
        break;
      }
    }
    if (command) {
      await command.focus();
      const before = await command.getAttribute("aria-expanded");
      await page.keyboard.press("Enter");
      await page.waitForTimeout(60);
      const afterEnter = await command.getAttribute("aria-expanded");
      assert.notEqual(afterEnter, before, `${viewportName}: Enter did not expand/collapse a command`);
      if (afterEnter === "true") await assertLinksClearSourceNodes(page, viewportName);
      await page.keyboard.press(" ");
      await page.waitForTimeout(60);
      assert.equal(await command.getAttribute("aria-expanded"), before, `${viewportName}: Space did not toggle a command`);
    }
    assert.ok((await visibleNodeCount(page)) >= initialNodes, `${viewportName}: command expansion removed the tree`);

    const hiddenToggle = await firstAvailable(page, [
      '#hidden-toggle',
      'button[aria-label*="hidden" i]',
      '[role="switch"][aria-label*="hidden" i]',
    ]);
    if (hiddenToggle && await hiddenToggle.isVisible()) {
      const before = await hiddenToggle.getAttribute("aria-pressed");
      await hiddenToggle.click();
      assert.notEqual(await hiddenToggle.getAttribute("aria-pressed"), before, `${viewportName}: hidden toggle did not change state`);
    }
    const rootToggle = await firstAvailable(page, [
      '#root-options-toggle',
      'button[aria-label*="root" i][aria-pressed]',
      'button[aria-label*="flag" i][aria-pressed]',
    ]);
    if (rootToggle && await rootToggle.isVisible()) {
      const before = await rootToggle.getAttribute("aria-pressed");
      await rootToggle.click();
      assert.notEqual(await rootToggle.getAttribute("aria-pressed"), before, `${viewportName}: root options toggle did not change state`);
    }

    const copyButton = await firstAvailable(page, [
      'button[aria-label*="copy command path" i]',
      'button[aria-label*="copy" i][data-copy]',
      '[data-copy] button',
    ]);
    if (copyButton && await copyButton.isVisible()) {
      await copyButton.click();
      const copyStates = [];
      for (const delay of [0, 50, 200, 500, 1000]) {
        if (delay) await page.waitForTimeout(delay);
        copyStates.push(await page.evaluate(() => ({
          label: document.querySelector("[data-copy]")?.getAttribute("aria-label"),
          status: document.querySelector("[data-copy-status]")?.textContent,
        })));
      }
      const copyState = `${await copyButton.getAttribute("aria-label")} ${await inspector.innerText()} ${await page.locator("[data-copy-status]").innerText().catch(() => "")}`;
      const copiedState = copyStates.some(({ label, status }) => /copied|copy failed/i.test(`${label || ""} ${status || ""}`));
      assert.ok(copiedState, `${viewportName}: copy action has no accessible status (${JSON.stringify({ copyState, copyStates })})`);
    }

    await assertNoPageOverflow(page);
    await assertNoNodeCollisions(page);
    assert.deepEqual(consoleErrors, [], `${viewportName}: browser console errors: ${consoleErrors.join("; ")}`);
    assert.deepEqual(pageErrors, [], `${viewportName}: page errors: ${pageErrors.join("; ")}`);
  } finally {
    page.removeListener("console", onConsole);
    page.removeListener("pageerror", onPageError);
  }
}

const { server, url } = await serveDocs();
const browser = await chromium.launch({ headless: true });
try {
  for (const [name, viewport] of [
    ["desktop", { width: 1440, height: 1000 }],
    ["mobile", { width: 390, height: 844 }],
  ]) {
    const context = await browser.newContext({
      baseURL: url,
      permissions: ["clipboard-read", "clipboard-write"],
      reducedMotion: "reduce",
      viewport,
    });
    const page = await context.newPage();
    try {
      await page.emulateMedia({ reducedMotion: "reduce" });
      await runExplorerAssertions(page, url, name);
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}

console.log("CLI explorer browser validation passed at desktop and mobile viewports (reduced motion).");

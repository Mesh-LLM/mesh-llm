import assert from "node:assert/strict";
import { createReadStream, existsSync, mkdirSync, statSync } from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(websiteDir, "../..");
const docsRoot = path.resolve(repoRoot, "docs");
const screenshotDirectory = path.resolve(websiteDir, "test-results/paid-inference");
const ctaHref = "https://github.com/Mesh-LLM/mesh-llm#paid-inference-over-lightning";

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
    throw new Error("docs/ does not exist; build the website before running this test");
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

function collectBrowserErrors(page) {
  const consoleErrors = [];
  const pageErrors = [];

  page.on("console", (message) => {
    if (message.type() !== "error") return;
    const location = message.location();
    consoleErrors.push(location.url ? `${message.text()} (${location.url})` : message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  return { consoleErrors, pageErrors };
}

async function mockGithubApi(page) {
  await page.route(/^https:\/\/api\.github\.com\/repos\/Mesh-LLM\/mesh-llm(?:\/releases\/latest)?$/, async (route) => {
    const body = route.request().url().endsWith("/releases/latest")
      ? { tag_name: "v0.0.0-test" }
      : { stargazers_count: 0 };
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
}

async function assertNoHorizontalOverflow(page, label) {
  const overflow = await page.evaluate(() => {
    const section = document.querySelector("#paid-inference");
    return {
      page: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      section: section ? section.scrollWidth - section.clientWidth : Number.POSITIVE_INFINITY,
    };
  });
  assert.ok(overflow.page <= 1, `${label}: page overflows horizontally by ${overflow.page}px`);
  assert.ok(overflow.section <= 1, `${label}: section overflows horizontally by ${overflow.section}px`);
}

async function assertActiveMode(page, mode, label) {
  const active = await page.evaluate(() => {
    const root = document.querySelector("[data-paid-inference]");
    return {
      selectedTab: root?.querySelector('[role="tab"][aria-selected="true"]')?.dataset.paymentTab,
      visiblePanel: root?.querySelector('[data-payment-panel]:not([hidden])')?.dataset.paymentPanel,
      visibleMap: root?.querySelector('[data-payment-map]:not([hidden])')?.dataset.paymentMap,
    };
  });

  assert.deepEqual(active, {
    selectedTab: mode,
    visiblePanel: mode,
    visibleMap: mode,
  }, `${label}: ${mode} selection, panel, and map are out of sync`);
}

async function assertCompleteState(page, label, motionAction = "hidden") {
  const root = page.locator("[data-paid-inference]");
  const control = page.locator("[data-payment-motion]");
  assert.equal(await root.getAttribute("data-payment-state"), "complete", `${label}: flow is not complete`);
  if (motionAction === "hidden") {
    assert.equal(await control.isHidden(), true, `${label}: motion control should be hidden`);
    return;
  }

  assert.equal(await control.isVisible(), true, `${label}: motion control should remain visible`);
  assert.equal(await control.getAttribute("data-payment-motion-action"), motionAction, `${label}: motion control action is wrong`);
  assert.equal((await control.textContent())?.trim(), "Replay animation", `${label}: motion control should offer replay`);
}

async function assertDocumentContract(page, label) {
  const contract = await page.evaluate((expectedHref) => {
    const catalog = document.querySelector("#catalog");
    const section = document.querySelector("#paid-inference");
    const install = document.querySelector("#install");
    const root = section?.querySelector("[data-paid-inference]");
    const tablist = root?.querySelector('[role="tablist"]');
    const tabs = [...(root?.querySelectorAll('[role="tab"]') ?? [])];
    const panels = [...(root?.querySelectorAll('[role="tabpanel"]') ?? [])];
    const cta = section?.querySelector(".paid-inference-cta");

    return {
      order: Boolean(catalog && section && install
        && catalog.compareDocumentPosition(section) & Node.DOCUMENT_POSITION_FOLLOWING
        && section.compareDocumentPosition(install) & Node.DOCUMENT_POSITION_FOLLOWING),
      heading: section?.querySelector("h2")?.textContent?.replace(/\s+/g, " ").trim(),
      ctaHref: cta?.href,
      ctaContainerOpacity: cta?.parentElement ? getComputedStyle(cta.parentElement).opacity : null,
      tabCount: tabs.length,
      panelCount: panels.length,
      controls: tabs.map((tab) => tab.getAttribute("aria-controls")),
      tabIds: tabs.map((tab) => tab.id),
      tablistLabel: tablist?.getAttribute("aria-label"),
      panelIds: panels.map((panel) => panel.id),
      panelLabels: panels.map((panel) => panel.getAttribute("aria-labelledby")),
      selected: tabs.map((tab) => tab.getAttribute("aria-selected")),
      tabIndexes: tabs.map((tab) => tab.getAttribute("tabindex")),
      activePanel: panels.find((panel) => !panel.hidden)?.dataset.paymentPanel,
      activeMap: root?.querySelector("[data-payment-map]:not([hidden])")?.dataset.paymentMap,
      expectedHref,
    };
  }, ctaHref);

  assert.equal(contract.order, true, `${label}: section is not between catalog and install`);
  assert.equal(contract.heading, "Earn on the mesh.Spend on the mesh.", `${label}: heading changed`);
  assert.equal(contract.ctaHref, contract.expectedHref, `${label}: CTA target changed`);
  assert.equal(contract.ctaContainerOpacity, "1", `${label}: CTA is hidden by the title reveal`);
  assert.equal(contract.tabCount, 2, `${label}: expected two tabs`);
  assert.equal(contract.panelCount, 2, `${label}: expected two tabpanels`);
  assert.equal(contract.tablistLabel, "Paid inference direction", `${label}: tablist has no accessible label`);
  assert.deepEqual(contract.controls, contract.panelIds, `${label}: tabs do not control their matching panels`);
  assert.deepEqual(contract.panelLabels, contract.tabIds, `${label}: tabpanels are not labelled by their matching tabs`);
  assert.deepEqual(contract.selected, ["true", "false"], `${label}: initial tab selection is wrong`);
  assert.deepEqual(contract.tabIndexes, ["0", "-1"], `${label}: initial roving tabindex is wrong`);
  assert.equal(contract.activePanel, "earn", `${label}: initial panel is not Earn`);
  assert.equal(contract.activeMap, "earn", `${label}: initial diagram is not Earn`);
}

async function assertFixedWallet(page, label) {
  const earnTab = page.locator('[data-payment-tab="earn"]');
  const spendTab = page.locator('[data-payment-tab="spend"]');
  const root = page.locator("[data-paid-inference]");

  await earnTab.click();
  await root.waitFor({ state: "visible" });
  const earnBox = await page.locator('[data-payment-map="earn"] .paid-inference-wallet--local').boundingBox();

  await spendTab.click();
  const spendBox = await page.locator('[data-payment-map="spend"] .paid-inference-wallet--local').boundingBox();

  assert.ok(earnBox && spendBox, `${label}: local wallet is missing`);
  assert.ok(Math.abs(earnBox.x - spendBox.x) <= 1, `${label}: local wallet moves horizontally between modes (${JSON.stringify({ earnBox, spendBox })})`);
  assert.ok(Math.abs(earnBox.y - spendBox.y) <= 1, `${label}: local wallet moves vertically between modes (${JSON.stringify({ earnBox, spendBox })})`);
}

async function assertKeyboardTabs(page, label) {
  const earnTab = page.locator('[data-payment-tab="earn"]');
  const spendTab = page.locator('[data-payment-tab="spend"]');

  await earnTab.focus();
  await page.keyboard.press("End");
  assert.equal(await spendTab.getAttribute("aria-selected"), "true", `${label}: End did not select Spend`);
  assert.equal(await page.evaluate(() => document.activeElement?.getAttribute("data-payment-tab")), "spend", `${label}: End did not move focus`);
  await assertActiveMode(page, "spend", label);

  const endScroll = await page.evaluate(() => window.scrollY);
  await page.keyboard.press("End");
  assert.equal(await page.evaluate(() => window.scrollY), endScroll, `${label}: End scrolled the page while already on the last tab`);

  await page.keyboard.press("Home");
  assert.equal(await earnTab.getAttribute("aria-selected"), "true", `${label}: Home did not select Earn`);
  await assertActiveMode(page, "earn", label);

  const homeScroll = await page.evaluate(() => window.scrollY);
  await page.keyboard.press("Home");
  assert.equal(await page.evaluate(() => window.scrollY), homeScroll, `${label}: Home scrolled the page while already on the first tab`);

  await page.keyboard.press("ArrowRight");
  assert.equal(await spendTab.getAttribute("aria-selected"), "true", `${label}: ArrowRight did not select Spend`);
  await assertActiveMode(page, "spend", label);

  await page.keyboard.press("ArrowLeft");
  assert.equal(await earnTab.getAttribute("aria-selected"), "true", `${label}: ArrowLeft did not select Earn`);
  await assertActiveMode(page, "earn", label);
}

async function installTransitionRecorder(page) {
  await page.evaluate(() => {
    const root = document.querySelector("[data-paid-inference]");
    window.__paidInferenceTransitions = [];
    window.__paidInferencePhaseSnapshots = {};
    window.__paidInferenceAnimatedAxes = [];
    if (!root) return;

    const record = () => {
      const value = `${root.dataset.paymentMode}:${root.dataset.paymentState}`;
      const transitions = window.__paidInferenceTransitions;
      if (transitions.at(-1) !== value) transitions.push(value);

      const map = root.querySelector(`[data-payment-map="${root.dataset.paymentMode}"]`);
      const activeRoute = map?.querySelector("[data-payment-route].is-active")?.dataset.paymentRoute ?? null;
      window.__paidInferencePhaseSnapshots[value] = { activeRoute };
    };

    record();
    new MutationObserver(record).observe(root, {
      attributes: true,
      attributeFilter: ["data-payment-mode", "data-payment-state"],
    });
    new MutationObserver((records) => {
      records.forEach(({ target }) => {
        if (!(target instanceof HTMLElement) || !target.matches("[data-payment-packet]")) return;
        const opacity = Number.parseFloat(target.style.opacity || "0");
        if (opacity <= 0) return;
        ["X", "Y"].forEach((axis) => {
          if (target.style.transform.includes(`translate${axis}(`) && !window.__paidInferenceAnimatedAxes.includes(axis)) {
            window.__paidInferenceAnimatedAxes.push(axis);
          }
        });
      });
    }).observe(root, {
      attributes: true,
      attributeFilter: ["style"],
      subtree: true,
    });
  });
}

function assertOrderedTransitions(transitions, expected, label) {
  let cursor = -1;
  expected.forEach((transition) => {
    cursor = transitions.indexOf(transition, cursor + 1);
    assert.notEqual(cursor, -1, `${label}: missing ${transition} in ${transitions.join(" -> ")}`);
  });
}

async function assertPhaseSnapshots(page, modes, phases, label) {
  const snapshots = await page.evaluate(() => window.__paidInferencePhaseSnapshots ?? {});
  modes.forEach((mode) => {
    phases.forEach((phase) => {
      assert.equal(snapshots[`${mode}:${phase}`]?.activeRoute, phase, `${label}: ${mode} ${phase} route was not active`);
    });
  });
}

async function assertPauseControl(page, label) {
  const control = page.locator("[data-payment-motion]");
  assert.equal(await control.isVisible(), true, `${label}: pause control is not visible during autoplay`);
  await control.click();
  assert.equal(await control.getAttribute("aria-pressed"), "true", `${label}: pause control did not expose its pressed state`);
  assert.equal((await control.textContent())?.trim(), "Resume animation", `${label}: pause control label did not change`);

  await page.waitForTimeout(50);
  const packetStyle = () => page.evaluate(() => {
    const map = document.querySelector("[data-payment-map]:not([hidden])");
    const packet = map?.querySelector("[data-payment-route].is-active [data-payment-packet]")
      ?? map?.querySelector("[data-payment-packet]");
    return packet?.getAttribute("style") ?? null;
  });
  const before = await packetStyle();
  assert.notEqual(before, null, `${label}: active flow has no packet to pause`);
  await page.waitForTimeout(250);
  const after = await packetStyle();
  assert.equal(after, before, `${label}: packet continued moving while paused`);

  await control.click();
  assert.equal(await control.getAttribute("aria-pressed"), "false", `${label}: pause control did not clear its pressed state`);
  assert.equal((await control.textContent())?.trim(), "Pause animation", `${label}: resume control label did not change back`);
}

async function runFocusedTabCase(browser, baseUrl) {
  const context = await browser.newContext({
    baseURL: baseUrl,
    reducedMotion: "no-preference",
    viewport: { width: 1440, height: 1000 },
  });
  const page = await context.newPage();
  const errors = collectBrowserErrors(page);

  try {
    await mockGithubApi(page);
    await page.goto("/", { waitUntil: "networkidle" });
    const earnTab = page.locator('[data-payment-tab="earn"]');
    await earnTab.focus();
    await page.waitForTimeout(3500);

    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute("data-payment-tab")), "earn", "focused tab: focus moved unexpectedly");
    await assertActiveMode(page, "earn", "focused tab");
    await assertCompleteState(page, "focused tab", "replay");
    assert.deepEqual(errors.consoleErrors, [], `focused tab: browser console errors: ${errors.consoleErrors.join("; ")}`);
    assert.deepEqual(errors.pageErrors, [], `focused tab: page errors: ${errors.pageErrors.join("; ")}`);
  } finally {
    await context.close();
  }
}

async function runReducedMotionCase(browser, baseUrl, name, viewport) {
  const context = await browser.newContext({
    baseURL: baseUrl,
    reducedMotion: "reduce",
    viewport,
  });
  const page = await context.newPage();
  const errors = collectBrowserErrors(page);

  try {
    await mockGithubApi(page);
    await page.goto("/", { waitUntil: "networkidle" });
    await assertDocumentContract(page, name);

    const section = page.locator("#paid-inference");
    await section.scrollIntoViewIfNeeded();
    await assertCompleteState(page, `${name}: initial reduced-motion state`);

    await assertFixedWallet(page, name);
    await assertCompleteState(page, `${name}: reduced-motion click replay`);
    await assertKeyboardTabs(page, name);
    await assertCompleteState(page, `${name}: reduced-motion keyboard replay`);
    await assertNoHorizontalOverflow(page, name);

    mkdirSync(screenshotDirectory, { recursive: true });
    await section.screenshot({ path: path.join(screenshotDirectory, `${name}.png`) });

    assert.deepEqual(errors.consoleErrors, [], `${name}: browser console errors: ${errors.consoleErrors.join("; ")}`);
    assert.deepEqual(errors.pageErrors, [], `${name}: page errors: ${errors.pageErrors.join("; ")}`);
  } finally {
    await context.close();
  }
}

async function runAnimationCase(browser, baseUrl, name, viewport, options = {}) {
  const context = await browser.newContext({
    baseURL: baseUrl,
    reducedMotion: "no-preference",
    viewport,
  });
  const page = await context.newPage();
  const errors = collectBrowserErrors(page);
  const axis = viewport.width <= 720 ? "Y" : "X";
  const phases = ["request", "deposit", "stream", "settlement"];

  try {
    await mockGithubApi(page);
    await page.goto("/", { waitUntil: "networkidle" });
    await assertDocumentContract(page, name);
    await installTransitionRecorder(page);
    const section = page.locator("#paid-inference");
    await section.scrollIntoViewIfNeeded();

    if (options.pause) {
      await page.waitForFunction(() => {
        const root = document.querySelector("[data-paid-inference]");
        return !document.querySelector("[data-payment-motion]")?.hidden
          && root?.dataset.paymentState !== "idle"
          && root?.dataset.paymentState !== "complete";
      }, null, { timeout: 5000 });
      await assertPauseControl(page, name);
    }

    await page.waitForFunction(() => {
      const rootElement = document.querySelector("[data-paid-inference]");
      return rootElement?.dataset.paymentMode === "spend" && rootElement?.dataset.paymentState === "complete";
    }, null, { timeout: 15000 });
    await assertActiveMode(page, "spend", name);
    await assertCompleteState(page, `${name}: autoplay completion`, "replay");
    if (options.pause) {
      assert.equal(await page.evaluate(() => document.activeElement?.hasAttribute("data-payment-motion")), true, `${name}: motion control lost focus when autoplay completed`);
    }

    const autoplayTransitions = await page.evaluate(() => window.__paidInferenceTransitions ?? []);
    assertOrderedTransitions(autoplayTransitions, [
      ...phases.map((phase) => `earn:${phase}`),
      ...phases.map((phase) => `spend:${phase}`),
      "spend:complete",
    ], `${name}: autoplay`);
    await assertPhaseSnapshots(page, ["earn", "spend"], phases, `${name}: autoplay`);
    assert.equal(await page.evaluate((expectedAxis) => window.__paidInferenceAnimatedAxes?.includes(expectedAxis), axis), true, `${name}: packets did not animate on the ${axis} axis`);

    if (options.controlReplay) {
      await page.evaluate(() => { window.__paidInferenceTransitions = []; });
      await page.locator("[data-payment-motion]").click();
      await page.waitForFunction(() => document.querySelector("[data-paid-inference]")?.dataset.paymentState !== "complete");
      await page.waitForFunction(() => document.querySelector("[data-paid-inference]")?.dataset.paymentState === "complete", null, { timeout: 7000 });
      const controlTransitions = await page.evaluate(() => window.__paidInferenceTransitions ?? []);
      assertOrderedTransitions(controlTransitions, [
        ...phases.map((phase) => `spend:${phase}`),
        "spend:complete",
      ], `${name}: control replay`);
      assert.equal(await page.evaluate(() => document.activeElement?.hasAttribute("data-payment-motion")), true, `${name}: motion control lost focus when replay completed`);
      await assertCompleteState(page, `${name}: control replay completion`, "replay");
    }

    if (options.replay) {
      await page.evaluate(() => { window.__paidInferenceTransitions = []; });
      await page.locator('[data-payment-tab="earn"]').click();
      await page.waitForFunction(() => document.querySelector("[data-paid-inference]")?.dataset.paymentState === "complete", null, { timeout: 7000 });
      await assertActiveMode(page, "earn", `${name}: manual replay`);

      const replayTransitions = await page.evaluate(() => window.__paidInferenceTransitions ?? []);
      assertOrderedTransitions(replayTransitions, [
        ...phases.map((phase) => `earn:${phase}`),
        "earn:complete",
      ], `${name}: manual replay`);
    }

    if (options.breakpoint) {
      const control = page.locator("[data-payment-motion]");
      await control.focus();
      await page.setViewportSize({ width: 667, height: 500 });
      assert.equal(await page.evaluate(() => document.activeElement?.hasAttribute("data-payment-motion")), true, `${name}: motion control lost focus across the responsive breakpoint`);
      await assertCompleteState(page, `${name}: responsive breakpoint`, "replay");
    }

    await assertNoHorizontalOverflow(page, name);
    assert.deepEqual(errors.consoleErrors, [], `${name}: browser console errors: ${errors.consoleErrors.join("; ")}`);
    assert.deepEqual(errors.pageErrors, [], `${name}: page errors: ${errors.pageErrors.join("; ")}`);
  } finally {
    await context.close();
  }
}

const { server, url } = await serveDocs();
const browser = await chromium.launch({ headless: true });

try {
  await runReducedMotionCase(browser, url, "desktop", { width: 1440, height: 1000 });
  await runReducedMotionCase(browser, url, "mobile", { width: 390, height: 844 });
  await runAnimationCase(browser, url, "desktop animation", { width: 1440, height: 1000 }, { pause: true, replay: true, controlReplay: true });
  await runAnimationCase(browser, url, "tablet animation", { width: 800, height: 700 }, { breakpoint: true });
  await runAnimationCase(browser, url, "mobile landscape animation", { width: 667, height: 375 });
  await runFocusedTabCase(browser, url);
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}

console.log("Paid inference browser validation passed for desktop, tablet, mobile, reduced-motion, and short-viewport animation cases.");

/**
 * Events-over-time chart stability regression guards.
 *
 * The /logs page crashed with React's "Maximum update depth exceeded" (minified
 * React error #185) in both the dev server and the production console bundle.
 *
 * Root cause was NOT in the chart, and not ResizeObserver-driven. The chart was
 * the victim: `useLogsLiveRecovery` returned a fresh `[]` literal for
 * `auditEntries` when audit streaming was disabled, while every sibling
 * collection was memoized. That new identity per render invalidated the ledger
 * memo chain (auditEntries -> filteredAuditEntries -> mergedRows ->
 * categoryRows), handing `<BarChart>` a new `data` array on every render.
 * recharts' ChartDataContextProvider then re-dispatched `setChartData`, and
 * react-redux v9's synchronous `defaultNoopBatch` notified subscribers inline,
 * re-rendering the tree that minted the next `[]` — self-sustaining until
 * React's 50-nested-update ceiling tripped the route error boundary.
 *
 * The fix is a shared module-level empty array in that hook. Prop-hygiene
 * fixes inside the chart (memoizing tick/margin/cursor, bypassing
 * ResponsiveContainer) were measured and do NOT close the loop, because they
 * sit downstream of the identity churn — do not re-attempt them.
 *
 * The mechanism is guarded at the unit level by the referential-stability tests
 * in use-logs-live-recovery.test.tsx; these e2e specs guard the integrated
 * perturbation shapes.
 *
 * These tests mount the chart with the same mocked /api/logs/* backend shape
 * used by log-workflows.spec.ts and exercise the perturbation shapes that
 * triggered the loop:
 *
 * - frozen time via page.clock.setFixedTime (Date.now() pinned while real
 *   timers and ResizeObserver keep firing) — the exact timing combo from the
 *   bug report;
 * - real-timer interaction variants: idle, resize storms, SSE stream bursts,
 *   opening the request inspector;
 * - a high-volume dataset (370 rows) under frozen time;
 * - viewport-growth tracking (the chart must re-measure when the container
 *   grows, guarding against a fixed-width "measure once" wrapper).
 *
 * With the fix in place these variants are expected to pass; they guard against
 * regressions in each perturbation shape. Reverting the hook one-liner turns
 * them red again (a variable subset per run, since the loop trips
 * probabilistically once the 50-update ceiling is in reach).
 */

import { expect, test, type Page } from '../fixtures/base'

const PINNED_NOW = new Date('2026-08-15T12:34:56Z')
const NOW_MS = PINNED_NOW.getTime()

/**
 * The dev server (vite, which Playwright's webServer runs) defaults to harness
 * data mode, which renders built-in fixtures and ignores /api/logs/*. The
 * tracked e2e specs opt the page into live mode before registering routes;
 * without this the mocked datasets below never reach the ledger.
 */
const DATA_MODE_STORAGE_KEY = 'mesh-llm-ui-preview:data-mode:v2'

type RequestKind = 'completed' | 'failed' | 'active'

function logsPage(items: unknown[]): { items: unknown[]; nextCursor: null } {
  return { items, nextCursor: null }
}

function minutesAgoIso(minutes: number): string {
  return new Date(NOW_MS - minutes * 60_000).toISOString()
}

function requestRow(index: number, kind: RequestKind, createdMinutesAgo: number) {
  return {
    requestId: `10000000-0000-4000-8000-${String(index + 1).padStart(12, '0')}`,
    outcome: kind,
    createdAt: minutesAgoIso(createdMinutesAgo),
    terminalAt: kind === 'active' ? null : minutesAgoIso(Math.max(0, createdMinutesAgo - 1)),
    route: 'chat_completions',
    model: 'Qwen3-8B-Q4_K_M.gguf',
    provider: 'mesh',
    engine: 'skippy',
    statusCode: kind === 'completed' ? 200 : kind === 'failed' ? 502 : null,
    source: kind === 'active' ? 'active' : 'durable'
  }
}

const REQUEST_ROWS = [
  requestRow(0, 'completed', 10),
  requestRow(1, 'completed', 70),
  requestRow(2, 'failed', 35),
  requestRow(3, 'completed', 180),
  requestRow(4, 'failed', 300),
  requestRow(5, 'completed', 420),
  requestRow(6, 'completed', 540),
  requestRow(7, 'failed', 640),
  requestRow(8, 'completed', 760),
  requestRow(9, 'active', 4)
]

const AUDIT_CODES: Record<string, string> = {
  system: 'runtime_startup_started',
  quic: 'mesh_quic_inbound_accepted',
  gossip: 'gossip_direct_peer_promoted'
}

function auditRow(category: keyof typeof AUDIT_CODES, index: number, createdMinutesAgo: number) {
  return {
    entryId: `audit-${category}-${String(index).padStart(4, '0')}`,
    occurredAt: minutesAgoIso(createdMinutesAgo),
    source: category === 'gossip' || category === 'quic' ? 'mesh' : 'runtime',
    code: AUDIT_CODES[category],
    severity: 'info',
    sequence: index + 1
  }
}

const AUDIT_ROWS = [
  auditRow('system', 1, 20),
  auditRow('quic', 1, 30),
  auditRow('system', 2, 95),
  auditRow('gossip', 1, 50),
  auditRow('quic', 2, 240),
  auditRow('system', 3, 460),
  auditRow('gossip', 2, 490)
]

type StreamMode = 'error' | 'burst'

/**
 * Mocks the /api/logs/* route family with the same response shapes as
 * log-workflows.spec.ts's installLogsBackend. Re-registering the route
 * (e.g. to swap in a different dataset or stream mode) intercepts the earlier
 * registration because Playwright matches routes in reverse registration order.
 */
function mockLogsApi(
  page: Page,
  options: {
    streamMode?: StreamMode
    eventsUrlCount?: number
    requests?: unknown[]
    audit?: unknown[]
  } = {}
) {
  const { streamMode = 'error', eventsUrlCount = 3 } = options
  const requests = options.requests ?? REQUEST_ROWS
  const audit = options.audit ?? AUDIT_ROWS
  let eventStreamCalls = 0
  const burstEntries = [1, 2, 3].map((n) => ({ ...auditRow('gossip', n + 10, n * 7), sequence: 100 + n }))

  return page
    .addInitScript((storageKey) => window.localStorage.setItem(storageKey, 'live'), DATA_MODE_STORAGE_KEY)
    .then(() =>
      page.context().route('**/api/logs/**', async (route) => {
        const url = new URL(route.request().url())
        const path = url.pathname

        if (path === '/api/logs/requests' && route.request().method() === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(logsPage(requests))
          })
        }
        if (path === '/api/logs/audit' && route.request().method() === 'GET') {
          return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(logsPage(audit)) })
        }
        if (path === '/api/logs/events') {
          eventStreamCalls += 1
          if (streamMode === 'burst' && eventStreamCalls <= eventsUrlCount) {
            const body = burstEntries
              .map((entry) => `id: v1:${entry.sequence}.0.0\nevent: audit_entry\ndata: ${JSON.stringify(entry)}\n\n`)
              .join('')
            return route.fulfill({ status: 200, contentType: 'text/event-stream', body })
          }
          return route.fulfill({
            status: 200,
            contentType: 'text/event-stream',
            body: 'id: v1:0.0.0\nevent: stream_error\ndata: {"code":"invalid_event"}\n\n'
          })
        }
        return route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({ error: { code: 'unsupported' } })
        })
      })
    )
}

/** Collects console errors/warnings and pageerrors so tests can fail on render loops. */
function captureErrors(page: Page) {
  const consoleIssues: string[] = []
  let reactCrash = ''
  page.on('console', (msg) => {
    if (msg.type() === 'error' || msg.type() === 'warning') {
      consoleIssues.push(`[${msg.type()}] ${msg.text()}`)
    }
  })
  page.on('pageerror', (error) => {
    reactCrash = error.toString()
  })
  return {
    consoleIssues,
    reactCrash: () => reactCrash,
    depthErrors: () =>
      [...consoleIssues, reactCrash].filter((m) => /Maximum update depth exceeded|Too many re-renders/i.test(m))
  }
}

/** Mounts the /logs page and waits for the chart to render real data. */
async function mountChart(page: Page) {
  const capture = captureErrors(page)
  await page.goto('/logs')
  await expect(page.getByRole('heading', { name: /Logs/i })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('img', { name: /Events over time/i }).first()).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('path.recharts-rectangle').first()).toBeVisible({ timeout: 30_000 })
  return capture
}

/** Fails the test if React hit the update-depth limit or the chart unmounted. */
async function assertChartAndNoLoop(page: Page, capture: ReturnType<typeof captureErrors>) {
  if (capture.depthErrors().length > 0 || capture.reactCrash()) {
    throw new Error(
      `render loop detected:\nreactCrash=${capture.reactCrash()}\nconsoleTail=\n${capture.consoleIssues
        .slice(-12)
        .join('\n')}`
    )
  }
  await expect(
    page.getByRole('img', { name: /Events over time/i }).first(),
    'chart subtree disappeared from the tree'
  ).toBeVisible({ timeout: 8_000 })
}

test.describe('events over time chart stability', () => {
  test.use({ viewport: { width: 1400, height: 950 } })

  test.describe('frozen time (setFixedTime)', () => {
    test.beforeEach(async ({ page }) => {
      await mockLogsApi(page)
      await page.clock.setFixedTime(PINNED_NOW)
    })

    test('renders real data without exceeding React update depth', async ({ page }) => {
      const info = test.info()
      const capture = captureErrors(page)

      await page.goto('/logs')
      await expect(page.getByRole('heading', { name: /Logs/i })).toBeVisible({ timeout: 30_000 })
      await page.waitForTimeout(2500)
      await info.attach('console-issues', {
        contentType: 'text/plain',
        body: capture.consoleIssues.join('\n') || '(none)'
      })

      const chart = page.getByRole('img', { name: /Events over time/i }).first()
      await expect(chart).toBeVisible({ timeout: 30_000 })
      const barRects = page.locator('path.recharts-rectangle')
      await expect(barRects.first()).toBeVisible({ timeout: 30_000 })
      expect(await barRects.count()).toBeGreaterThan(0)

      const maxDepthMessages = capture.depthErrors()
      expect(capture.reactCrash()).toBe('')
      expect(maxDepthMessages, `render loop detected:\n${maxDepthMessages.join('\n')}`).toEqual([])
    })

    test('renders a high-volume dataset without exceeding React update depth', async ({ page }) => {
      // 370 requests spanning [now-370min, now-1min] under frozen time.
      //
      // mergeLogEventWindow caps the MERGED request+audit list at 64 rows
      // (LOG_EVENT_WINDOW_LIMIT), newest first — the cap is not per-category.
      // AUDIT_ROWS contributes 3 entries inside the newest 64 (system@20min,
      // quic@30min, gossip@50min); the 4th (gossip@490min) falls outside the
      // window. So the legend reports Requests61 + System1 + QUIC1 + Gossip1,
      // summing to exactly 64. Asserting the total proves the full mock dataset
      // reached the ledger and was windowed rather than silently dropped.
      // The bar count itself stays small, so the fidelity gate is "bars render
      // at all and no render loop fires".
      const manyRequests = Array.from({ length: 370 }, (_, i) =>
        requestRow(i, i === 0 ? 'active' : i % 3 === 0 ? 'failed' : 'completed', 370 - i)
      )
      await mockLogsApi(page, { requests: manyRequests })

      const capture = captureErrors(page)
      await page.goto('/logs')
      await expect(page.getByRole('heading', { name: /Logs/i })).toBeVisible({ timeout: 30_000 })
      await page.waitForTimeout(2500)

      const legend = page.getByRole('list', { name: 'Visible event categories' })
      await expect(legend).toContainText('Requests61')
      await expect(legend).toContainText('System1')
      await expect(legend).toContainText('QUIC1')
      await expect(legend).toContainText('Gossip1')
      const bars = await page.locator('path.recharts-rectangle').count()
      expect(bars).toBeGreaterThan(0)
      expect(capture.depthErrors(), `render loop detected:\n${capture.depthErrors().join('\n')}`).toEqual([])
    })

    test('chart tracks viewport growth', async ({ page }) => {
      // Guards against a fixed-width "measure once" chart wrapper: the chart
      // must re-measure and grow when the viewport (and container) grows.
      const capture = captureErrors(page)
      await page.goto('/logs')
      await expect(page.getByRole('heading', { name: /Logs/i })).toBeVisible({ timeout: 30_000 })
      await expect(page.getByRole('img', { name: /Events over time/i }).first()).toBeVisible({ timeout: 30_000 })

      await page.setViewportSize({ width: 900, height: 600 })
      await page.waitForTimeout(700)
      const wSmall = (await page.locator('.recharts-surface').first().boundingBox())?.width ?? 0

      await page.setViewportSize({ width: 1920, height: 1080 })
      await page.waitForTimeout(1200)
      const wLarge = (await page.locator('.recharts-surface').first().boundingBox())?.width ?? 0

      expect(wSmall).toBeGreaterThan(0)
      expect(wLarge).toBeGreaterThan(wSmall * 1.15)
      expect(capture.depthErrors(), `render loop detected:\n${capture.depthErrors().join('\n')}`).toEqual([])
    })
  })

  test.describe('real timers (no clock)', () => {
    test.beforeEach(async ({ page }) => {
      await mockLogsApi(page)
    })

    test('idle: static mount with no interaction stays stable', async ({ page }) => {
      const capture = await mountChart(page)
      await page.waitForTimeout(2500)
      await assertChartAndNoLoop(page, capture)
    })

    test('resize-storm: repeated viewport resizes stay stable', async ({ page }) => {
      const capture = await mountChart(page)
      const sizes = [
        { width: 1200, height: 800 },
        { width: 375, height: 667 },
        { width: 900, height: 700 },
        { width: 1400, height: 950 }
      ]
      for (let cycle = 0; cycle < 3; cycle += 1) {
        for (const size of sizes) {
          await page.setViewportSize(size)
          await page.waitForTimeout(160)
        }
      }
      await page.waitForTimeout(2500)
      await assertChartAndNoLoop(page, capture)
    })

    test('stream-burst: SSE audit events mutating chart data stay stable', async ({ page }) => {
      // Re-register the mock with burst mode so the first event stream
      // deliveries mutate the chart data while it is mounted.
      await mockLogsApi(page, { streamMode: 'burst', eventsUrlCount: 3 })
      const capture = await mountChart(page)
      await page.waitForTimeout(4000)
      await assertChartAndNoLoop(page, capture)
    })

    test('row-click: opening the request inspector stays stable', async ({ page }) => {
      // Reproduces the live crash on the unfixed tree: opening the inspector
      // shifts the layout while the chart is mounted.
      //
      // Note the inspector is a modal dialog: while it is open the ledger (and
      // therefore the chart) is correctly removed from the accessibility tree,
      // so the chart must NOT be asserted visible in that state. The render-loop
      // check still applies throughout, and the chart must come back on close.
      const capture = await mountChart(page)
      const row = page.getByRole('row', { name: 'Inspect request 10000000-0000-4000-8000-000000000001' })
      await expect(row).toBeVisible({ timeout: 30_000 })
      await row.click()

      const inspector = page.getByRole('dialog')
      await expect(inspector).toBeVisible({ timeout: 30_000 })
      await page.waitForTimeout(2500)
      if (capture.depthErrors().length > 0 || capture.reactCrash()) {
        throw new Error(
          `render loop detected while the inspector was open:\nreactCrash=${capture.reactCrash()}\nconsoleTail=\n${capture.consoleIssues
            .slice(-12)
            .join('\n')}`
        )
      }

      // Closing the inspector must restore the ledger and remount the chart
      // without tripping the loop.
      await page.keyboard.press('Escape')
      await expect(inspector).toBeHidden({ timeout: 30_000 })
      await page.waitForTimeout(2500)
      await assertChartAndNoLoop(page, capture)
    })
  })
})

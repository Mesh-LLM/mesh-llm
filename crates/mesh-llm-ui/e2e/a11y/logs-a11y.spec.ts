import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '../fixtures/base'

const accents = [undefined, 'blue', 'cyan', 'violet', 'green', 'amber', 'pink'] as const
const themes = ['light', 'dark'] as const

test('request logs keeps text, controls, and live status badges AA-compliant across themes and accents', async ({
  page
}) => {
  // Install (not pause) before navigation: per Playwright's clock docs,
  // pausing before goto lets the page get stuck (React's own mount work
  // depends on real timers while the route loads). Install here, keep
  // ticking through navigation and mount, and only pause once the route has
  // rendered and the SSE connection below is deliberately held open — see
  // the `releaseEventsStream` route below for how FALLBACK_DELAY_MS (1s) is
  // kept from racing the `Reconnecting` assertion.
  await page.clock.install({ time: new Date('2026-08-04T12:00:00Z') })
  await page.addInitScript(
    (storageKey) => window.localStorage.setItem(storageKey, 'live'),
    'mesh-llm-ui-preview:data-mode:v2'
  )
  await page.route('**/api/logs/requests*', (route) =>
    route.fulfill({
      json: {
        items: [
          {
            requestId: '00000000-0000-4000-8000-000000000001',
            outcome: 'active',
            createdAt: '2026-08-04T12:00:00Z',
            terminalAt: null,
            route: 'reserve',
            model: 'Qwen3',
            provider: 'reserve-a',
            engine: 'skippy',
            statusCode: null,
            source: 'active'
          },
          {
            requestId: '00000000-0000-4000-8000-000000000002',
            outcome: 'completed',
            createdAt: '2026-08-04T12:01:00Z',
            terminalAt: '2026-08-04T12:01:01Z',
            route: 'reserve',
            model: 'Qwen3',
            provider: 'reserve-a',
            engine: 'skippy',
            statusCode: 200,
            source: 'durable'
          },
          {
            requestId: '00000000-0000-4000-8000-000000000003',
            outcome: 'cancelled',
            createdAt: '2026-08-04T12:02:00Z',
            terminalAt: '2026-08-04T12:02:01Z',
            route: 'reserve',
            model: 'Qwen3',
            provider: 'reserve-a',
            engine: 'skippy',
            statusCode: null,
            source: 'durable'
          },
          {
            requestId: '00000000-0000-4000-8000-000000000004',
            outcome: 'failed',
            createdAt: '2026-08-04T12:03:00Z',
            terminalAt: '2026-08-04T12:03:01Z',
            route: 'reserve',
            model: 'Qwen3',
            provider: 'reserve-a',
            engine: 'skippy',
            statusCode: 500,
            source: 'durable'
          }
        ],
        nextCursor: null
      }
    })
  )
  await page.route('**/api/logs/audit*', (route) =>
    route.fulfill({
      json: {
        items: [
          {
            entryId: 'audit-0001',
            occurredAt: '2026-08-04T12:00:00Z',
            source: 'runtime',
            code: 'runtime_ready',
            severity: 'info',
            sequence: 1
          }
        ],
        nextCursor: null
      }
    })
  )
  // Hold the SSE connection open instead of erroring it immediately: closing
  // it (below) is what drives the browser's EventSource `onerror`, which is
  // what starts the `reconnecting` -> `polling` clock. Releasing it only
  // once the route has mounted and the clock is paused keeps that transition
  // from racing real (dev-server compile/module-fetch) time.
  let releaseEventsStream: (() => void) | undefined
  await page.route('**/api/logs/events*', async (route) => {
    await new Promise<void>((resolve) => {
      releaseEventsStream = resolve
    })
    await route.fulfill({
      contentType: 'text/event-stream',
      body: 'retry: 600000\nid: v1:0.0.0\nevent: stream_error\ndata: {"code":"invalid_event"}\n\n'
    })
  })
  await page.goto('/logs')
  const infoBanner = page.getByRole('region', { name: 'System logs' })
  await expect(infoBanner.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
  await expect.poll(() => releaseEventsStream).toBeDefined()
  await page.clock.pauseAt(new Date(await page.evaluate(() => Date.now() + 50)))
  releaseEventsStream?.()
  await expect(infoBanner).toContainText('Monitor request activity and operational events from this MeshLLM host.')
  await expect(infoBanner.getByText('Reconnecting', { exact: true })).toBeVisible()
  // The rest of this test doesn't depend on the live-recovery clock — resume
  // it so axe's own internal scheduling (which yields via real timers) can
  // run. A frozen clock through the whole theme/accent loop below hangs
  // AxeBuilder#analyze() until the test-level timeout.
  await page.clock.resume()
  await expect(infoBanner.getByRole('button', { name: 'Clean up logs' })).toBeVisible()
  await expect(
    page.getByRole('region', { name: 'Event log controls' }).getByRole('button', { name: 'Export view' })
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Dead-letter retry' })).toHaveCount(0)
  await expect(page.getByRole('table', { name: 'MeshLLM event logs' })).toHaveCount(1)
  const root = page.locator('html')
  await expect(root).toHaveAttribute('data-theme-preference')

  for (const theme of themes) {
    for (const accent of accents) {
      await root.evaluate(
        (element, preference) => {
          element.dataset.theme = preference.theme
          if (preference.accent === undefined) {
            delete element.dataset.accent
          } else {
            element.dataset.accent = preference.accent
          }
        },
        { theme, accent }
      )
      await expect(root).toHaveAttribute('data-theme', theme)
      await expect(root).toHaveCSS('color-scheme', theme)
      if (accent === undefined) {
        await expect(root).not.toHaveAttribute('data-accent')
      } else {
        await expect(root).toHaveAttribute('data-accent', accent)
      }
      await expect
        .poll(() =>
          root.evaluate(() =>
            document
              .getAnimations({ subtree: true })
              .filter((animation) => animation instanceof CSSTransition)
              .every((transition) => transition.playState !== 'running')
          )
        )
        .toBe(true)

      const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze()
      expect(results.violations).toEqual([])
    }
  }
})

test('fallback log polling toggle remains AA-compliant while paused', async ({ page }) => {
  // Install (not pause) before navigation, and hold the SSE connection open
  // until the clock is paused — see the comment on the previous test.
  await page.clock.install({ time: new Date('2026-08-04T13:00:00Z') })
  await page.addInitScript(
    (storageKey) => window.localStorage.setItem(storageKey, 'live'),
    'mesh-llm-ui-preview:data-mode:v2'
  )
  await page.route('**/api/logs/requests*', (route) => route.fulfill({ json: { items: [], nextCursor: null } }))
  await page.route('**/api/logs/audit*', (route) => route.fulfill({ json: { items: [], nextCursor: null } }))
  // Hold the SSE connection open — see the comment on the previous test.
  let releaseEventsStream: (() => void) | undefined
  await page.route('**/api/logs/events*', async (route) => {
    await new Promise<void>((resolve) => {
      releaseEventsStream = resolve
    })
    await route.fulfill({
      contentType: 'text/event-stream',
      body: 'retry: 600000\nid: v1:0.0.0\nevent: stream_error\ndata: {"code":"invalid_event"}\n\n'
    })
  })

  await page.goto('/logs')
  await expect(page.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
  await expect.poll(() => releaseEventsStream).toBeDefined()
  await page.clock.pauseAt(new Date(await page.evaluate(() => Date.now() + 50)))
  releaseEventsStream?.()
  await expect(page.getByText('Reconnecting', { exact: true })).toBeVisible()
  // Step deliberately past FALLBACK_DELAY_MS (1s) into the `polling` state,
  // where the toggle under test renders.
  await page.clock.runFor(1_000)
  const pollingToggle = page.getByRole('button', { name: 'Fallback log polling' })
  await expect(pollingToggle).toHaveAttribute('aria-pressed', 'true')

  await pollingToggle.click()

  await expect(pollingToggle).toHaveAttribute('aria-pressed', 'false')
  await expect(pollingToggle).toContainText('Polling paused')
  await page.clock.resume()
  await expect(pollingToggle).toContainText('Polling paused')
  const results = await new AxeBuilder({ page })
    .include('[aria-label="Fallback log polling"]')
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze()

  expect(results.violations).toEqual([])
})

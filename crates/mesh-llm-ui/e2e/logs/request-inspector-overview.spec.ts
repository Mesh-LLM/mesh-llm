import type { Locator, Page } from '@playwright/test'
import { openInspector, requestRow, selectInspectorTab } from './request-inspector-helpers'
import {
  installRequestInspectorRoutes,
  REQUEST_INSPECTOR_IDS,
  type RequestInspectorBackendEvidence
} from './request-inspector-routes'
import { expect, test } from './request-inspector-test'

async function renderedTokenLineCount(value: Locator, token: string): Promise<number> {
  return value.evaluate((element, target) => {
    const textNode = element.firstChild
    const tokenStart = element.textContent?.indexOf(target) ?? -1
    if (!(textNode instanceof Text) || tokenStart < 0) return 0
    const range = document.createRange()
    range.setStart(textNode, tokenStart)
    range.setEnd(textNode, tokenStart + target.length)
    return new Set(Array.from(range.getClientRects(), (rect) => Math.round(rect.top))).size
  }, token)
}

async function useDarkTheme(page: Page): Promise<void> {
  await page.addInitScript(
    (storageKey) =>
      window.localStorage.setItem(
        storageKey,
        JSON.stringify({
          theme: 'dark',
          accent: 'blue',
          density: 'normal',
          panelStyle: 'soft',
          panelStyleOverride: false
        })
      ),
    'mesh-llm-ui-preview:preferences:v1'
  )
}

test('keeps completed evidence and footer actions visible while only the inspector body scrolls', async ({ page }) => {
  await installRequestInspectorRoutes(page)
  await page.setViewportSize({ width: 375, height: 520 })
  await page.goto('/logs')
  const row = requestRow(page, REQUEST_INSPECTOR_IDS.completed)
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)
  const overview = inspector.getByRole('region', { name: 'Request overview' })

  await expect(inspector).toHaveAttribute('data-request-inspector-shell', 'fixed')
  await expect(overview.getByText('1 attempt / 0 retries', { exact: true })).toBeVisible()
  const caller = overview.getByRole('region', { name: 'Caller' })
  await expect(caller).toContainText('9f0c…bb04')
  await expect(caller).toContainText('203.0.113.24:48712')
  await expect(caller).toContainText('Remote QUIC HTTP')
  const callerCopy = caller.getByRole('button', { name: 'Copy caller endpoint ID' })
  await expect(callerCopy).toBeVisible()
  await expect
    .poll(() => callerCopy.evaluate((element) => element.getBoundingClientRect().height))
    .toBeGreaterThanOrEqual(44)
  const lifecycle = overview.getByRole('list', { name: 'Lifecycle events' })
  await expect(lifecycle.locator('li[data-event-kind="stream_started"]')).toHaveCount(1)
  await expect(lifecycle).toContainText('Stream started')
  await overview.getByRole('button', { name: 'Later lifecycle events' }).click()
  await expect(lifecycle.locator('li[data-event-kind="stream_completed"]')).toHaveCount(1)
  await expect(lifecycle).toContainText('Stream done')
  await expect(overview.getByRole('list', { name: 'Routing attempts' })).toContainText('mesh-primary')
  const retention = overview.getByRole('region', { name: 'Artifact retention' })
  await expect(retention).toContainText('2 available · 1 unavailable · 1 missing · 1 corrupt')

  const footer = inspector.getByRole('contentinfo', { name: 'Request inspector actions' })
  await expect(footer.getByRole('button', { name: 'Close', exact: true })).toBeVisible()
  await expect(footer.getByRole('button', { name: 'Delete terminal request' })).toBeVisible()
  const scrollBody = inspector.locator('[data-request-inspector-scroll="body"]')
  const copyControl = inspector.getByRole('button', { name: 'Copy Request ID' })
  await expect(scrollBody).toHaveCount(1)
  await expect(scrollBody).toHaveAttribute('data-request-inspector-scroll', 'body')
  await expect.poll(() => scrollBody.evaluate((element) => element.scrollHeight > element.clientHeight)).toBe(true)
  const [scrollBodyBounds, footerBounds, copyControlBounds] = await Promise.all([
    scrollBody.boundingBox(),
    footer.boundingBox(),
    copyControl.boundingBox()
  ])
  if (scrollBodyBounds === null) throw new Error('Request Inspector scroll body bounds missing')
  if (footerBounds === null) throw new Error('Request Inspector footer bounds missing')
  if (copyControlBounds === null) throw new Error('Request Inspector copy control bounds missing')
  expect(scrollBodyBounds.y + scrollBodyBounds.height).toBeLessThanOrEqual(footerBounds.y + 1)
  expect(copyControlBounds.height).toBeGreaterThanOrEqual(44)
  expect(copyControlBounds.width).toBeGreaterThanOrEqual(44)

  await inspector.getByRole('button', { name: 'Close inspector' }).focus()
  await page.keyboard.press('Tab')
  await expect(copyControl).toBeFocused()
  await expect
    .poll(() =>
      copyControl.evaluate((element) => {
        const styles = getComputedStyle(element)
        return (
          element.matches(':focus-visible') &&
          styles.outlineStyle !== 'none' &&
          Number.parseFloat(styles.outlineWidth) >= 2 &&
          styles.outlineColor !== 'transparent' &&
          styles.outlineColor !== 'rgba(0, 0, 0, 0)'
        )
      })
    )
    .toBe(true)

  const footerTop = await footer.evaluate((element) => element.getBoundingClientRect().top)
  await scrollBody.hover()
  await page.mouse.wheel(0, 1_000)
  await expect.poll(() => scrollBody.evaluate((element) => element.scrollTop > 0)).toBe(true)
  await expect(footer).toBeVisible()
  await expect.poll(() => footer.evaluate((element) => element.getBoundingClientRect().top)).toBe(footerTop)

  await footer.getByRole('button', { name: 'Close', exact: true }).click()
  await expect(inspector).toHaveCount(0)
  await expect(row).toBeFocused()
})

test('covers the odd request-metadata row with a coherent cell at tablet width in dark mode', async ({ page }) => {
  await installRequestInspectorRoutes(page)
  await useDarkTheme(page)
  await page.setViewportSize({ width: 768, height: 900 })
  await page.goto('/logs')
  const root = page.locator('html')
  await expect(root).toHaveAttribute('data-theme', 'dark')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)
  const metadata = inspector.getByRole('region', { name: 'Request metadata' })
  const finalCell = metadata.getByText('Record source', { exact: true }).locator('..')
  await finalCell.scrollIntoViewIfNeeded()

  const coverage = await finalCell.evaluate((cell) => {
    const grid = cell.parentElement
    if (!(grid instanceof HTMLElement)) return null
    const cellBounds = cell.getBoundingClientRect()
    const gridBounds = grid.getBoundingClientRect()
    const probe = document.elementFromPoint(gridBounds.right - 1, cellBounds.top + cellBounds.height / 2)
    return {
      cellBackground: getComputedStyle(cell).backgroundColor,
      probeBackground: probe === null ? null : getComputedStyle(probe).backgroundColor,
      probeCovered: probe !== null && (probe === cell || cell.contains(probe)),
      rightGap: gridBounds.right - cellBounds.right
    }
  })

  expect(coverage).not.toBeNull()
  if (coverage === null) throw new Error('Request metadata grid coverage could not be measured')
  expect(coverage.rightGap).toBeLessThanOrEqual(1)
  expect(coverage.probeCovered).toBe(true)
  expect(coverage.probeBackground).toBe(coverage.cellBackground)
})

test('always exposes Close but limits Delete to durable terminal requests', async ({ page }) => {
  await installRequestInspectorRoutes(page)
  await page.goto('/logs')

  for (const requestId of [REQUEST_INSPECTOR_IDS.active, REQUEST_INSPECTOR_IDS.transient] as const) {
    const inspector = await openInspector(page, requestId)
    const footer = inspector.getByRole('contentinfo', { name: 'Request inspector actions' })
    await expect(footer.getByRole('button', { name: 'Close', exact: true })).toBeVisible()
    await expect(footer.getByRole('button', { name: 'Delete terminal request' })).toHaveCount(0)
    await footer.getByRole('button', { name: 'Close', exact: true }).click()
  }
})

test('keeps short metric words intact at tablet width', async ({ page }) => {
  await installRequestInspectorRoutes(page)
  await page.setViewportSize({ width: 768, height: 900 })
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)
  const streamMetric = inspector.getByText('3 stream events / 42 completion tokens', { exact: true })

  expect(await renderedTokenLineCount(streamMetric, 'completion tokens')).toBe(1)
})

test('keeps successful diagnostic words whole at mobile width while retaining overflow-safe wrapping', async ({
  page
}) => {
  await installRequestInspectorRoutes(page)
  await useDarkTheme(page)
  await page.setViewportSize({ width: 375, height: 900 })
  await page.goto('/logs')
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)
  await selectInspectorTab(page, inspector, { id: 'diagnostics', name: 'Diagnostics' })
  const values = [
    {
      element: inspector
        .getByRole('definition')
        .filter({ hasText: '2 available · 1 unavailable · 1 missing · 1 corrupt' }),
      token: 'missing'
    },
    {
      element: inspector.getByRole('definition').filter({ hasText: 'Metadata only; body content not requested' }),
      token: 'requested'
    }
  ] as const

  for (const value of values) {
    await expect(value.element).toBeVisible()
    expect(
      await value.element.evaluate((element) => ({
        overflowWrap: getComputedStyle(element).overflowWrap,
        wordBreak: getComputedStyle(element).wordBreak
      }))
    ).toEqual({ overflowWrap: 'break-word', wordBreak: 'normal' })
    expect(await renderedTokenLineCount(value.element, value.token)).toBe(1)
    await expect.poll(() => value.element.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
  }
})

test('restores delete-trigger focus after cancellation and audited completion', async ({ page }) => {
  const backend = await installRequestInspectorRoutes(page)
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)
  const trigger = inspector.getByRole('button', { name: 'Delete terminal request' })

  await trigger.click()
  let confirmation = page.getByRole('dialog', { name: 'Delete terminal request?' })
  await confirmation.getByRole('button', { name: 'Cancel' }).click()
  await expect(trigger).toBeFocused()

  await trigger.click()
  confirmation = page.getByRole('dialog', { name: 'Delete terminal request?' })
  await confirmation.getByLabel('Required audit reason').fill('remove invalid request')
  await confirmation.getByRole('button', { name: 'Confirm deletion' }).click()
  await expect(confirmation.getByText('Request removed.')).toBeVisible()
  expect(backend.deleteRequestBodies).toHaveLength(1)
  expect(JSON.parse(backend.deleteRequestBodies[0] ?? '')).toMatchObject({ reason: 'remove invalid request' })
  await confirmation.getByRole('button', { name: 'Cancel' }).click()
  await expect(trigger).toBeFocused()
})

const SUCCESS_DIAGNOSTICS = [
  { id: REQUEST_INSPECTOR_IDS.completed, state: '2 available · 1 unavailable · 1 missing · 1 corrupt' },
  { id: REQUEST_INSPECTOR_IDS.empty, state: 'None retained' },
  { id: REQUEST_INSPECTOR_IDS.malformed, state: '1 available' }
] as const

for (const scenario of SUCCESS_DIAGNOSTICS) {
  test(`shows truthful successful diagnostics for ${scenario.id}`, async ({ page }) => {
    const backend = await installRequestInspectorRoutes(page)
    await page.goto('/logs')
    const inspector = await openInspector(page, scenario.id)
    await selectInspectorTab(page, inspector, { id: 'diagnostics', name: 'Diagnostics' })

    await expect(inspector.getByRole('heading', { name: 'No errors' })).toBeVisible()
    const summary = inspector.getByRole('definition').filter({ hasText: scenario.state })
    await expect(summary).toBeVisible()
    await expect(inspector.getByText('Metadata only; body content not requested', { exact: true })).toBeVisible()
    expect(backend.artifactDetailCalls).toEqual([])
  })
}

test('shows failed diagnostics, retries, terminal evidence, and truthful artifact states', async ({ page }) => {
  const backend: RequestInspectorBackendEvidence = await installRequestInspectorRoutes(page)
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.failed)
  await selectInspectorTab(page, inspector, { id: 'diagnostics', name: 'Diagnostics' })

  await expect(inspector.getByRole('heading', { name: 'Request failed' })).toBeVisible()
  const evidence = inspector.getByRole('list', { name: 'Ordered diagnostic evidence' })
  await expect(evidence).toContainText(/retry-primary[\s\S]*retry-secondary/)
  await expect(evidence).toContainText(/attempt_failed[\s\S]*stream_error[\s\S]*audit_error[\s\S]*failed/)
  await expect(evidence).toContainText('http://peer-b.mesh.invalid:9337')
  await expect(evidence).toContainText('https://peer-b.mesh.invalid')
  await expect(inspector.getByText('error_diagnostic', { exact: true })).toBeVisible()
  await expect(inspector.getByText('error_trace', { exact: true })).toBeVisible()
  await expect(inspector.getByText('corrupt', { exact: true })).toBeVisible()
  await expect(inspector.getByText('missing', { exact: true })).toBeVisible()
  expect(backend.artifactDetailCalls).toEqual([])
})

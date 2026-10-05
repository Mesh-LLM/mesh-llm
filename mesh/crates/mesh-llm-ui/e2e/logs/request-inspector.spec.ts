import AxeBuilder from '@axe-core/playwright'
import { openInspector, selectInspectorTab } from './request-inspector-helpers'
import { installRequestInspectorRoutes, REQUEST_INSPECTOR_IDS } from './request-inspector-routes'
import { expect, test } from './request-inspector-test'

const EXPECTED_TAB_NAMES = ['Overview', 'Payloads', 'Timeline', 'Diagnostics'] as const
const GEOMETRY_TABS = [
  { id: 'payloads', name: 'Payloads', settledRole: 'heading', settledName: 'Request' },
  { id: 'timeline', name: 'Timeline', settledRole: 'list', settledName: 'Stream timeline' },
  { id: 'diagnostics', name: 'Diagnostics', settledRole: 'heading', settledName: 'No errors' },
  { id: 'overview', name: 'Overview', settledRole: 'region', settledName: 'Request overview' }
] as const

test('keeps fixed geometry across all four canonical tabs for a completed request', async ({ page }, testInfo) => {
  await installRequestInspectorRoutes(page)
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)

  await expect(inspector).toHaveAttribute('data-request-inspector-shell', 'fixed')
  await expect(inspector.getByRole('tab')).toHaveText(EXPECTED_TAB_NAMES)

  const geometry: Array<{ readonly tab: string; readonly width: number; readonly height: number }> = []
  for (const tab of GEOMETRY_TABS) {
    await selectInspectorTab(page, inspector, tab)
    await expect(inspector.getByRole(tab.settledRole, { name: tab.settledName, exact: true })).toBeVisible()
    const bounds = await inspector.boundingBox()
    expect(bounds).not.toBeNull()
    if (bounds === null) throw new Error(`Request Inspector bounds missing on ${tab.name}`)
    geometry.push({ tab: tab.name, width: bounds.width, height: bounds.height })
  }

  const baseline = geometry[0]
  if (baseline === undefined) throw new Error('Request Inspector geometry was not captured')
  expect(baseline.width).toBeGreaterThan(720)
  for (const bounds of geometry.slice(1)) {
    expect(Math.abs(bounds.width - baseline.width)).toBeLessThanOrEqual(1)
    expect(Math.abs(bounds.height - baseline.height)).toBeLessThanOrEqual(1)
  }
  await testInfo.attach('request-inspector-geometry.json', {
    body: JSON.stringify(geometry, null, 2),
    contentType: 'application/json'
  })

  const axe = await new AxeBuilder({ page })
    .include('[data-request-inspector-shell="fixed"]')
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze()
  expect(axe.violations.filter((violation) => ['serious', 'critical'].includes(violation.impact ?? ''))).toEqual([])
})

test('orders timeline evidence and renders both empty-state messages', async ({ page }) => {
  await installRequestInspectorRoutes(page)
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)
  await selectInspectorTab(page, inspector, { id: 'timeline', name: 'Timeline' })

  const streamTimeline = inspector.getByRole('list', { name: 'Stream timeline' })
  await expect(streamTimeline).toContainText(/stream_started[\s\S]*stream_chunk[\s\S]*stream_completed/)
  const attemptsTimeline = inspector.getByRole('list', { name: 'Routing attempts timeline' })
  await expect(attemptsTimeline).toContainText(/mesh-primary[\s\S]*reserve-a \/ skippy[\s\S]*Success/)

  await inspector.getByRole('button', { name: 'Close inspector' }).click()
  const emptyInspector = await openInspector(page, REQUEST_INSPECTOR_IDS.empty)
  await selectInspectorTab(page, emptyInspector, { id: 'timeline', name: 'Timeline' })
  await expect(emptyInspector.getByText('No lifecycle or stream markers were retained for this request.')).toBeVisible()
  await expect(emptyInspector.getByText('No proxy attempts were retained for this request.')).toBeVisible()
})

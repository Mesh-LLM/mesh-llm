import { expect, test, type Page } from '../fixtures/base'
import type { RequestInspectorBackendEvidence } from './request-inspector-routes'
import { installRequestInspectorRoutes, REQUEST_INSPECTOR_IDS } from './request-inspector-routes'

const DEEP_LINK = `/logs?inspectType=request&inspectId=${REQUEST_INSPECTOR_IDS.completed}&tab=diagnostics`
const pageErrors = new WeakMap<Page, string[]>()

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  pageErrors.set(page, errors)
  page.on('pageerror', (error) => errors.push(error.message))
})

test.afterEach(async ({ page }) => {
  expect(pageErrors.get(page) ?? []).toEqual([])
})

function expectNoDetailCalls(backend: RequestInspectorBackendEvidence): void {
  expect(backend.summaryCalls).toEqual([])
  expect(backend.eventCalls).toEqual([])
  expect(backend.artifactListCalls).toEqual([])
  expect(backend.proxyCalls).toEqual([])
  expect(backend.artifactDetailCalls).toEqual([])
}

test('keeps an unsupported request deep link inert with upgrade guidance and no polling', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-08-09T12:00:00Z') })
  const backend = await installRequestInspectorRoutes(page, { capability: 'unsupported' })
  await page.goto(DEEP_LINK)

  await expect(page.getByText('Request window unavailable')).toBeVisible()
  await expect(page.getByText(/Upgrade the host to inspect request history here/)).toBeVisible()
  await expect(page.getByRole('dialog', { name: 'Request Inspector' })).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'Something went wrong' })).toHaveCount(0)
  await page.clock.runFor(30_000)
  expect(backend.requestListCalls).toBe(1)
  expectNoDetailCalls(backend)
  expect(backend.logStreamCalls).toBe(0)
  await expect.poll(() => backend.auditStreamCalls).toBe(1)
})

test('keeps a loading request deep link inert until capability resolves', async ({ page }) => {
  const backend = await installRequestInspectorRoutes(page, { capability: 'loading' })
  await page.goto(DEEP_LINK)

  await expect.poll(() => backend.requestListCalls).toBe(1)
  await expect(page.getByRole('dialog', { name: 'Request Inspector' })).toHaveCount(0)
  await expect(page.getByText('Request window unavailable')).toHaveCount(0)
  expectNoDetailCalls(backend)
  expect(backend.logStreamCalls).toBe(0)
  await expect.poll(() => backend.auditStreamCalls).toBe(1)

  backend.releaseCapability()
  const inspector = page.getByRole('dialog', { name: 'Request Inspector' })
  await expect(inspector).toBeVisible()
  await expect(inspector.getByRole('tab', { name: 'Diagnostics' })).toHaveAttribute('data-state', 'active')
})

test('opens a supported canonical request deep link through normal detail routes', async ({ page }) => {
  const backend = await installRequestInspectorRoutes(page)
  await page.goto(DEEP_LINK)

  const inspector = page.getByRole('dialog', { name: 'Request Inspector' })
  await expect(inspector).toBeVisible()
  await expect(inspector.getByRole('tab', { name: 'Diagnostics' })).toHaveAttribute('data-state', 'active')
  await expect(inspector.getByRole('heading', { name: 'No errors' })).toBeVisible()
  await expect.poll(() => backend.summaryCalls).toEqual([REQUEST_INSPECTOR_IDS.completed])
  expect(backend.eventCalls).toEqual([REQUEST_INSPECTOR_IDS.completed])
  expect(backend.artifactListCalls).toEqual([REQUEST_INSPECTOR_IDS.completed])
  expect(backend.proxyCalls).toEqual([REQUEST_INSPECTOR_IDS.completed])
  expect(backend.artifactDetailCalls).toEqual([])
  await expect.poll(() => backend.logStreamCalls).toBe(1)
  await expect.poll(() => backend.auditStreamCalls).toBe(1)
})

test('returns an invalid legacy request ID to the ledger without an error boundary', async ({ page }) => {
  const backend = await installRequestInspectorRoutes(page)
  await page.goto('/logs/not-a-valid-request-id?tab=errors')

  await expect(page).toHaveURL(/\/logs$/)
  await expect(page.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
  await expect(page.getByRole('dialog', { name: 'Request Inspector' })).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'Something went wrong' })).toHaveCount(0)
  expectNoDetailCalls(backend)
  await expect.poll(() => backend.logStreamCalls).toBe(1)
  await expect.poll(() => backend.auditStreamCalls).toBe(1)
})

import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'

const enabled = process.env.MESH_LOGS_E2E === '1'
const openAiUrl = process.env.MESH_LOGS_E2E_OPENAI_URL
const persistedRequestId = process.env.MESH_LOGS_E2E_PERSISTED_REQUEST_ID
const DATA_MODE_STORAGE_KEY = 'mesh-llm-ui-preview:data-mode:v2'

type LogPage = {
  readonly items: Array<{ readonly requestId: string; readonly outcome: string; readonly source: string }>
}

function requestRow(page: Page, requestId: string) {
  return page.getByRole('row', { name: `Inspect request ${requestId}` })
}

function requireHarnessValue(value: string | undefined, name: string) {
  if (!value) throw new Error(`real console harness did not provide ${name}`)
  return value
}

function requireObservedRequest(value: string | undefined): string {
  if (!value) throw new Error('real console harness did not observe a durable request id')
  return value
}

test.describe('real embedded logging console', () => {
  test.skip(!enabled, 'runs only under scripts/qa-logging-console-e2e.sh')
  test.describe.configure({ mode: 'serial' })

  test('renders actual logging DTOs, durable lifecycle records, and real operator receipts', async ({
    page
  }, testInfo) => {
    test.setTimeout(60_000)
    const openAiEndpoint = requireHarnessValue(openAiUrl, 'MESH_LOGS_E2E_OPENAI_URL')
    const persistedId = requireHarnessValue(persistedRequestId, 'MESH_LOGS_E2E_PERSISTED_REQUEST_ID')
    const observedRequests: string[] = []
    let ledgerGets = 0
    let auditGets = 0
    let streamReconnectUrl = ''

    page.on('request', (request) => {
      const url = new URL(request.url())
      if (url.pathname === '/api/logs/requests' && request.method() === 'GET') ledgerGets += 1
      if (url.pathname === '/api/logs/audit' && request.method() === 'GET') auditGets += 1
      if (url.pathname === '/api/logs/events' && url.searchParams.has('cursor')) streamReconnectUrl = url.toString()
    })

    await page.addInitScript((storageKey) => window.localStorage.setItem(storageKey, 'live'), DATA_MODE_STORAGE_KEY)
    await page.goto(`/logs?replayCursor=${encodeURIComponent('v1:0.0.0')}`)
    await expect(page.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
    await expect(requestRow(page, persistedId)).toBeVisible()
    await expect.poll(() => streamReconnectUrl).not.toBe('')
    // This harness deliberately supplies an evicted cursor, so the initial GET
    // is followed by one authoritative replay-gap recovery before the stream
    // becomes the sole source of current request projections.
    await expect.poll(() => ledgerGets).toBeGreaterThanOrEqual(2)
    await page.waitForTimeout(500)
    const connectedLedgerGets = ledgerGets
    const connectedAuditGets = auditGets
    await page.waitForTimeout(5_500)
    expect(ledgerGets).toBe(connectedLedgerGets)
    expect(auditGets).toBe(connectedAuditGets)

    await requestRow(page, persistedId).click()
    const persistedInspector = page.getByRole('dialog', { name: 'Request Inspector' })
    await expect(persistedInspector).toBeVisible()
    await expect(persistedInspector.getByRole('region', { name: 'Request overview' })).toBeVisible()

    await page.getByRole('button', { name: 'Close inspector' }).click()
    const firstResponse = await page.request.post(openAiEndpoint, {
      data: { model: 'qa-no-model', messages: [{ role: 'user', content: 'real logging console lifecycle' }] }
    })
    expect(firstResponse.status()).toBeGreaterThanOrEqual(400)

    await expect
      .poll(async () => {
        const response = await page.request.get('/api/logs/requests?limit=10')
        if (!response.ok()) return undefined
        const body = (await response.json()) as LogPage
        const match = body.items.find((item) => item.requestId !== persistedId && item.source === 'durable')
        if (match) observedRequests.push(match.requestId)
        return match?.requestId
      })
      .toBeTruthy()
    const lifecycleRequestId = requireObservedRequest(observedRequests.at(-1))
    await expect(requestRow(page, lifecycleRequestId)).toBeVisible()
    expect(ledgerGets).toBe(connectedLedgerGets)
    await requestRow(page, lifecycleRequestId).click()
    await expect(page.getByRole('dialog', { name: 'Request Inspector' })).toBeVisible()
    await page.getByRole('tab', { name: 'Diagnostics' }).click()
    await expect(page.getByText('rejected', { exact: true }).first()).toBeVisible()

    const artifactsResponse = await page.request.get(`/api/logs/requests/${lifecycleRequestId}/artifacts`)
    expect(artifactsResponse.ok()).toBeTruthy()
    const artifacts = await artifactsResponse.json()
    const artifactStates = Array.isArray(artifacts.items)
      ? artifacts.items.map((artifact: { contentState?: unknown; redacted?: unknown }) => ({
          contentState: artifact.contentState,
          redacted: artifact.redacted
        }))
      : []
    await testInfo.attach('real-log-artifact-states.json', {
      body: JSON.stringify({ artifactStates }, null, 2),
      contentType: 'application/json'
    })

    await page.getByRole('button', { name: 'Close inspector' }).click()
    await page.getByRole('button', { name: 'Export view' }).click()
    const exportDialog = page.getByRole('dialog', { name: 'Export current log view' })
    await exportDialog.getByLabel('Required audit reason').fill('real embedded console certification')
    const download = page.waitForEvent('download')
    await exportDialog.getByRole('button', { name: 'Download export' }).click()
    await (await download).saveAs(testInfo.outputPath('mesh-llm-log-export.json'))
    await expect(exportDialog.getByRole('status')).toContainText(
      /(Bounded log export downloaded|bounded partial export was downloaded)/i
    )
    await exportDialog.getByRole('button', { name: 'Cancel' }).click()

    await page.getByRole('button', { name: 'Clean up logs' }).click()
    const cleanupDialog = page.getByRole('dialog', { name: 'Review log cleanup' })
    await cleanupDialog.getByLabel('Reason for removal').fill('real scoped cleanup certification')
    const previewResponse = page.waitForResponse(
      (response) => response.url().endsWith('/api/logs/cleanup/preview') && response.request().method() === 'POST'
    )
    await cleanupDialog.getByRole('button', { name: 'Review deletion' }).click()
    const previewReceipt = await (await previewResponse).json()
    await testInfo.attach('real-log-cleanup-preview.json', {
      body: JSON.stringify(previewReceipt, null, 2),
      contentType: 'application/json'
    })
    const confirmDialog = page.getByRole('dialog', { name: 'Review log cleanup' })
    await expect(confirmDialog.getByRole('heading', { name: 'Review log cleanup' })).toBeVisible()
    await confirmDialog.getByText('Audit details').click()
    await expect(confirmDialog.getByText('Operation ID', { exact: true })).toBeVisible()
    await expect(confirmDialog.getByText('Audit ID', { exact: true })).toBeVisible()
    const cleanupResponse = page.waitForResponse(
      (response) => response.url().endsWith('/api/logs/cleanup/run') && response.request().method() === 'POST'
    )
    await confirmDialog.getByRole('button', { name: 'Delete this batch' }).click()
    const cleanupReceipt = await (await cleanupResponse).json()
    await testInfo.attach('real-log-cleanup-run.json', {
      body: JSON.stringify(cleanupReceipt, null, 2),
      contentType: 'application/json'
    })
    await expect(confirmDialog.getByRole('status').last()).toContainText(
      /Log cleanup completed(\.| with diagnostics\.)/
    )
    await confirmDialog.getByRole('button', { name: 'Close' }).click()
    await expect(requestRow(page, lifecycleRequestId)).toHaveCount(0)
    await expect
      .poll(async () => {
        const response = await page.request.get('/api/logs/requests?limit=100')
        const body = (await response.json()) as LogPage
        return body.items.some((item) => item.requestId === lifecycleRequestId)
      })
      .toBe(false)

    const secondResponse = await page.request.post(openAiEndpoint, {
      data: { model: 'qa-no-model', messages: [{ role: 'user', content: 'real delete receipt lifecycle' }] }
    })
    expect(secondResponse.status()).toBeGreaterThanOrEqual(400)
    await expect
      .poll(async () => {
        const response = await page.request.get('/api/logs/requests?limit=10')
        const body = (await response.json()) as LogPage
        return body.items.find((item) => item.requestId !== persistedId && item.requestId !== lifecycleRequestId)
          ?.requestId
      })
      .toBeTruthy()

    const current = await page.request.get('/api/logs/requests?limit=10')
    const currentPage = (await current.json()) as LogPage
    const deleteId = requireObservedRequest(
      currentPage.items.find((item) => item.requestId !== persistedId && item.requestId !== lifecycleRequestId)
        ?.requestId
    )
    expect(deleteId).toBeTruthy()
    await page.reload()
    await requestRow(page, deleteId).click()
    await page.getByRole('button', { name: 'Delete terminal request' }).click()
    const deleteDialog = page.getByRole('dialog', { name: 'Delete terminal request?' })
    await deleteDialog.getByLabel('Required audit reason').fill('real delete receipt certification')
    await deleteDialog.getByRole('button', { name: 'Confirm deletion' }).click()
    await expect(deleteDialog.getByText('Request removed.')).toBeVisible()
    await expect(deleteDialog.getByText('Audit ID', { exact: true })).toBeVisible()
    await deleteDialog.getByRole('button', { name: 'Cancel' }).click()

    await page.getByRole('button', { name: 'Close inspector' }).click()
    for (const colorScheme of ['light', 'dark'] as const) {
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' })
      for (const width of [375, 768, 1280]) {
        await page.setViewportSize({ width, height: 900 })
        await page.goto('/logs')
        await expect(page.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
        await expect
          .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
          .toBe(true)
      }
    }
    const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze()
    await testInfo.attach('real-console-axe.json', {
      body: JSON.stringify(axe, null, 2),
      contentType: 'application/json'
    })
    expect(axe.violations.filter((violation) => ['serious', 'critical'].includes(violation.impact ?? ''))).toEqual([])
    await page.screenshot({ path: testInfo.outputPath('real-console-logs.png'), fullPage: true })
    await testInfo.attach('real-console-summary.json', {
      body: JSON.stringify(
        { persistedId, lifecycleRequestId, deleteId, ledgerGets, auditGets, streamReconnectUrl },
        null,
        2
      ),
      contentType: 'application/json'
    })
  })
})

import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '../fixtures/base'

const REQUEST_ID = '00000000-0000-4000-8000-000000000001'
const EVENT_ID = '00000000-0000-4000-8000-000000000002'
const ARTIFACT_ID = '00000000-0000-4000-8000-000000000003'
const OPERATION_ID = '00000000-0000-4000-8000-000000000004'
const AUDIT_ID = '00000000-0000-4000-8000-000000000005'
const OCCURRED_AT = '2026-08-04T12:00:00Z'
const LATER_OCCURRED_AT = '2026-08-04T12:30:00Z'
const TERMINAL_AT = '2026-08-04T12:00:01Z'
const FILTER_TO = '2026-08-04T13:00:00.000Z'
const DATA_MODE_STORAGE_KEY = 'mesh-llm-ui-preview:data-mode:v2'
const LONG_AUDIT_ID = `audit-${'identifier-'.repeat(12)}tail`
const LONG_AUDIT_CODE = `runtime_${'configuration_diagnostics_'.repeat(6)}warning`

type Lifecycle = 'active' | 'completed' | 'failed' | 'rejected' | 'cancelled' | 'dropped'
type StreamMode = 'event' | 'gap' | 'error' | 'unavailable'
type MaintenanceResult = 'completed' | 'partial' | 'failure'
type AuditIdentity = { readonly entryId: string; readonly code: string }

type LogsBackendOptions = {
  lifecycle?: Lifecycle
  streamMode?: StreamMode
  cleanupRunResults?: readonly MaintenanceResult[]
  deleteResults?: readonly MaintenanceResult[]
  auditIdentity?: AuditIdentity
  auditOccurredAt?: string
  delaySecondRequestsResponse?: boolean
}

type LogsBackend = {
  auditListCalls: number
  auditStreamUrls: string[]
  cleanupRunBodies: string[]
  cleanupRunResults: MaintenanceResult[]
  deleteBodies: string[]
  deleteResults: MaintenanceResult[]
  lifecycle: Lifecycle
  listCalls: number
  operationBodies: string[]
  streamUrls: string[]
  releaseAuditStream: (() => void) | undefined
  releaseSecondRequestsResponse: () => void
  releaseStream: (() => void) | undefined
  streamMode: StreamMode
}

function request(outcome: Lifecycle, source: 'active' | 'durable' = outcome === 'active' ? 'active' : 'durable') {
  return {
    requestId: REQUEST_ID,
    outcome,
    createdAt: OCCURRED_AT,
    terminalAt: outcome === 'active' ? null : TERMINAL_AT,
    route: 'reserve',
    model: 'Qwen3',
    provider: 'reserve-a',
    engine: 'skippy',
    statusCode: outcome === 'completed' ? 200 : outcome === 'active' ? null : 502,
    source
  }
}

function logsPage(items: readonly object[]) {
  return { items, nextCursor: null }
}

function auditEntry(
  identity: AuditIdentity = { entryId: 'audit-0001', code: 'runtime_config_diagnostics_warning' },
  occurredAt = OCCURRED_AT
) {
  return {
    entryId: identity.entryId,
    occurredAt,
    source: 'logs_api',
    code: identity.code,
    severity: 'warning',
    sequence: 1
  }
}

function artifact(contentState: 'available' | 'missing') {
  return {
    artifactId: ARTIFACT_ID,
    requestId: REQUEST_ID,
    occurredAt: TERMINAL_AT,
    kind: contentState === 'available' ? 'request' : 'response',
    mediaKind: 'application/json',
    checksum: null,
    bytes: 0,
    version: 1,
    redacted: contentState === 'available',
    truncated: false,
    contentState,
    contentBase64: null
  }
}

function artifactDeletion(state: 'previewed' | 'completed' | 'partial') {
  const failed = state === 'partial' ? 1 : 0
  const removed = state === 'previewed' ? 0 : 2 - failed
  return { removed, failed, ...(failed > 0 ? { failureClass: 'unsafe_path' } : {}) }
}

function cleanupReceipt(state: 'previewed' | 'completed' | 'partial') {
  return {
    operationId: OPERATION_ID,
    auditId: AUDIT_ID,
    cutoffBefore: TERMINAL_AT,
    requestLimit: 1,
    scope: {
      source: 'durable',
      cutoffBefore: TERMINAL_AT,
      requestLimit: 1,
      route: 'reserve',
      model: 'Qwen3',
      provider: 'reserve-a',
      engine: 'skippy',
      outcome: 'completed'
    },
    state,
    hasMore: state === 'partial',
    selectionFingerprint: 'bounded-selection',
    planned: { requests: 1, events: 1, artifacts: 2, proxyRecords: 0, databaseRows: 4 },
    executed: {
      requests: state === 'previewed' ? 0 : 1,
      events: 0,
      artifacts: 0,
      proxyRecords: 0,
      databaseRows: state === 'previewed' ? 0 : 1
    },
    artifactDeletion: artifactDeletion(state)
  }
}

function deleteReceipt(state: 'completed' | 'partial') {
  return {
    operationId: OPERATION_ID,
    auditId: AUDIT_ID,
    requestId: REQUEST_ID,
    state,
    selectionFingerprint: 'bounded-selection',
    planned: { requests: 1, events: 1, artifacts: 2, proxyRecords: 0, databaseRows: 4 },
    executed: { requests: 1, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 1 },
    artifactDeletion: artifactDeletion(state)
  }
}

async function tabTo(page: Page, locator: ReturnType<Page['getByLabel']>, maxTabs = 32) {
  for (let attempt = 0; attempt < maxTabs; attempt += 1) {
    if (await locator.evaluate((element) => element === document.activeElement)) return
    await page.keyboard.press('Tab')
  }
  await expect(locator).toBeFocused()
}

async function previewScopedCleanup(page: Page, reason = 'retention review') {
  await page.getByRole('button', { name: 'Clean up logs' }).click()
  const cleanupDialog = page.getByRole('dialog', { name: 'Review log cleanup' })
  await cleanupDialog.getByLabel('Reason for removal').fill(reason)
  await cleanupDialog.getByRole('button', { name: 'Review deletion' }).click()
  const reviewDialog = page.getByRole('dialog', { name: 'Review log cleanup' })
  await expect(reviewDialog.getByRole('heading', { name: 'Review log cleanup' })).toBeVisible()
  return reviewDialog
}

function requestRow(page: Page, requestId: string) {
  return page.getByRole('row', { name: `Inspect request ${requestId}` })
}

async function installLogsBackend(page: Page, options: LogsBackendOptions = {}) {
  await page.addInitScript((storageKey) => window.localStorage.setItem(storageKey, 'live'), DATA_MODE_STORAGE_KEY)
  let releaseSecondRequestsResponse: () => void = () => undefined
  const secondRequestsResponse = new Promise<void>((resolve) => {
    releaseSecondRequestsResponse = resolve
  })
  const state: LogsBackend = {
    auditListCalls: 0,
    auditStreamUrls: [],
    cleanupRunBodies: [],
    cleanupRunResults: [...(options.cleanupRunResults ?? ['completed'])],
    deleteBodies: [],
    deleteResults: [...(options.deleteResults ?? ['completed'])],
    lifecycle: options.lifecycle ?? 'active',
    listCalls: 0,
    operationBodies: [],
    streamUrls: [],
    releaseAuditStream: undefined,
    releaseSecondRequestsResponse,
    releaseStream: undefined,
    streamMode: options.streamMode ?? 'event'
  }

  await page.context().route('**/api/logs/**', async (route) => {
    const url = new URL(route.request().url())
    const method = route.request().method()

    if (url.pathname === '/api/logs/requests' && method === 'GET') {
      state.listCalls += 1
      if (options.delaySecondRequestsResponse && state.listCalls === 2) await secondRequestsResponse
      await route.fulfill({ json: logsPage([request(state.lifecycle)]) })
      return
    }
    if (url.pathname === '/api/logs/audit' && method === 'GET') {
      state.auditListCalls += 1
      await route.fulfill({ json: logsPage([auditEntry(options.auditIdentity, options.auditOccurredAt)]) })
      return
    }
    if (url.pathname === '/api/logs/events') {
      if (url.searchParams.get('audit') === '1') {
        state.auditStreamUrls.push(url.search)
        await new Promise<void>((resolve) => {
          state.releaseAuditStream = resolve
        })
        await route.fulfill({
          contentType: 'text/event-stream',
          body:
            'id: a1:2\n' +
            'event: audit_entry\n' +
            `data: ${JSON.stringify({ ...auditEntry(options.auditIdentity, options.auditOccurredAt), sequence: 2 })}\n\n`
        })
        return
      }
      state.streamUrls.push(url.search)
      if (state.streamMode === 'unavailable') {
        await route.abort('failed')
        return
      }
      if (state.streamMode === 'error') {
        await route.fulfill({
          contentType: 'text/event-stream',
          body: 'id: v1:0.0.0\nevent: stream_error\ndata: {"code":"invalid_event"}\n\n'
        })
        return
      }
      await new Promise<void>((resolve) => {
        state.releaseStream = resolve
      })
      if (state.streamMode === 'gap') {
        await route.fulfill({
          contentType: 'text/event-stream',
          body:
            'id: v1:2.0.0\n' +
            'event: replay_gap\n' +
            'data: {"channel":"requests","fromSequence":1,"toSequence":2,"recovery":{"endpoint":"/api/logs/requests","cursor":null}}\n\n'
        })
        return
      }
      state.lifecycle = 'completed'
      await route.fulfill({
        contentType: 'text/event-stream',
        body:
          'id: v1:1.0.0\n' +
          'event: log_event\n' +
          `data: {"eventId":"${EVENT_ID}","requestId":"${REQUEST_ID}","occurredAt":"${TERMINAL_AT}","channel":"requests","sequence":1,"kind":"completed"}\n\n`
      })
      return
    }
    if (url.pathname === `/api/logs/requests/${REQUEST_ID}` && method === 'GET') {
      await route.fulfill({ json: request(state.lifecycle) })
      return
    }
    if (url.pathname === `/api/logs/requests/${REQUEST_ID}/delete` && method === 'POST') {
      const body = route.request().postData() ?? ''
      state.operationBodies.push(body)
      state.deleteBodies.push(body)
      const result = state.deleteResults.shift() ?? 'completed'
      if (result === 'failure') {
        await route.fulfill({ status: 500, json: { error: { code: 'internal' } } })
        return
      }
      await route.fulfill({ json: deleteReceipt(result) })
      return
    }
    if (url.pathname === `/api/logs/requests/${REQUEST_ID}/events`) {
      await route.fulfill({
        json: logsPage([
          {
            eventId: EVENT_ID,
            requestId: REQUEST_ID,
            occurredAt: TERMINAL_AT,
            kind: state.lifecycle === 'failed' ? 'failed' : 'completed',
            model: 'Qwen3',
            provider: 'reserve-a',
            engine: 'skippy',
            attemptId: null,
            statusCode: state.lifecycle === 'completed' ? 200 : 502,
            durationMs: 1,
            tokens: 0
          }
        ])
      })
      return
    }
    if (url.pathname === `/api/logs/requests/${REQUEST_ID}/artifacts`) {
      await route.fulfill({ json: logsPage([artifact('available'), artifact('missing')]) })
      return
    }
    if (url.pathname === `/api/logs/artifacts/${ARTIFACT_ID}` && method === 'GET') {
      await route.fulfill({ json: { ...artifact('available'), contentBase64: 'eA==' } })
      return
    }
    if (url.pathname === '/api/logs/proxy') {
      await route.fulfill({ json: logsPage([]) })
      return
    }
    if (url.pathname === '/api/logs/requests/export' && method === 'POST') {
      state.operationBodies.push(route.request().postData() ?? '')
      await route.fulfill({
        json: { items: [], nextCursor: null, truncated: false, retryRequired: false, artifactContentIncluded: false }
      })
      return
    }
    if (url.pathname === '/api/logs/cleanup/preview' && method === 'POST') {
      state.operationBodies.push(route.request().postData() ?? '')
      await route.fulfill({ json: cleanupReceipt('previewed') })
      return
    }
    if (url.pathname === '/api/logs/cleanup/run' && method === 'POST') {
      const body = route.request().postData() ?? ''
      state.operationBodies.push(body)
      state.cleanupRunBodies.push(body)
      const result = state.cleanupRunResults.shift() ?? 'completed'
      if (result === 'failure') {
        await route.fulfill({ status: 500, json: { error: { code: 'internal' } } })
        return
      }
      await route.fulfill({ json: cleanupReceipt(result) })
      return
    }
    await route.fulfill({ status: 404, json: { error: { code: 'unsupported' } } })
  })

  return state
}

test('logs ledger follows a lifecycle event into immediate details and safe artifact states', async ({
  page: browserPage
}) => {
  await browserPage.clock.setFixedTime(new Date(FILTER_TO))
  const backend = await installLogsBackend(browserPage)

  await browserPage.goto(
    '/logs?timeRange=1h&model=Qwen3&provider=reserve-a&engine=skippy&route=reserve&source=durable&outcome=completed'
  )
  await expect(browserPage.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
  await expect(browserPage.getByRole('table', { name: 'MeshLLM event logs' })).toContainText(
    'runtime_config_diagnostics_warning'
  )
  await expect(browserPage.getByText('active', { exact: true }).first()).toBeVisible()
  await expect.poll(() => backend.releaseStream).toBeDefined()
  await expect.poll(() => backend.releaseAuditStream).toBeDefined()
  expect(backend.streamUrls[0]).toBe(
    '?channel=requests&channel=operations&filter=model%3AQwen3&filter=provider%3Areserve-a&filter=engine%3Askippy&filter=route%3Areserve&filter=outcome%3Acompleted'
  )
  expect(backend.streamUrls[0]).toContain('filter=route%3Areserve')
  expect(backend.streamUrls[0]).not.toContain('filter=source%3A')
  expect(backend.streamUrls[0]).not.toContain('filter=from%3A')
  expect(backend.streamUrls[0]).not.toContain('filter=to%3A')
  // The app resumes the audit stream from the last-seen sequence in the
  // initial audit list fixture (sequence: 1 -> cursor a1:1), not a cold start.
  expect(backend.auditStreamUrls[0]).toBe('?audit=1&cursor=a1%3A1')
  const auditListCallsBeforeEvent = backend.auditListCalls
  backend.releaseAuditStream?.()
  await expect.poll(() => backend.auditListCalls).toBeGreaterThan(auditListCallsBeforeEvent)
  backend.releaseStream?.()
  await expect(browserPage.getByText('completed', { exact: true })).toBeVisible()
  expect(backend.listCalls).toBeGreaterThanOrEqual(2)

  await requestRow(browserPage, REQUEST_ID).click()
  await expect(browserPage).toHaveURL(/inspectType=request&inspectId=00000000-0000-4000-8000-000000000001/)
  const requestInspector = browserPage.getByRole('dialog', { name: 'Request Inspector' })
  await expect(requestInspector.getByRole('region', { name: 'Request overview' })).toBeVisible()

  await requestInspector.getByRole('tab', { name: 'Payloads' }).click()
  await expect(requestInspector.getByText('Redacted', { exact: true }).first()).toBeVisible()
  await expect(requestInspector.getByText('missing', { exact: true }).first()).toBeVisible()
  await requestInspector.getByRole('button', { name: 'Download redacted artifact' }).click()
  await expect(requestInspector.getByText('Artifact download started.')).toBeVisible()
})

test('events chart applies the active populated bucket with real keyboard input', async ({ page: browserPage }) => {
  // Given
  await browserPage.clock.setFixedTime(new Date(FILTER_TO))
  await installLogsBackend(browserPage, {
    auditOccurredAt: LATER_OCCURRED_AT,
    lifecycle: 'completed',
    streamMode: 'unavailable'
  })
  await browserPage.goto('/logs?timeRange=1h')
  const listbox = browserPage.getByRole('listbox', { name: /Events over time stacked bar chart/ })
  const options = listbox.getByRole('option')
  await expect(options).toHaveCount(2)
  await listbox.focus()

  // When
  await browserPage.keyboard.press('ArrowRight')
  await expect(options.nth(1)).toHaveAttribute('aria-selected', 'true')
  await browserPage.keyboard.press('Enter')

  // Then
  await expect(browserPage.getByLabel('Chart time range')).toHaveValue('selected')
  await expect(browserPage.getByRole('button', { name: 'Clear window' })).toBeVisible()
  await expect(browserPage).toHaveURL((url) => {
    return (
      url.pathname === '/logs' &&
      url.searchParams.get('from') === '2026-08-04T12:30:00.000Z' &&
      url.searchParams.get('to') === '2026-08-04T12:30:59.999Z'
    )
  })
})

test('logs recovery uses the dedicated stream gap and bounded polling fallback', async ({ page: browserPage }) => {
  await browserPage.clock.install({ time: new Date(FILTER_TO) })
  const backend = await installLogsBackend(browserPage, { lifecycle: 'failed', streamMode: 'gap' })

  await browserPage.goto('/logs')
  await expect(browserPage.getByText('failed', { exact: true })).toBeVisible()
  await expect.poll(() => backend.releaseStream).toBeDefined()
  backend.releaseStream?.()
  await expect.poll(() => backend.listCalls).toBeGreaterThanOrEqual(2)

  backend.streamMode = 'error'
  await browserPage.reload()
  await expect(browserPage.getByText('Reconnecting', { exact: true })).toBeVisible()
  await expect(browserPage.getByRole('button', { name: 'Fallback log polling' })).toHaveCount(0)
  await browserPage.clock.pauseAt(new Date(await browserPage.evaluate(() => Date.now() + 2_000)))
  const pollingToggle = browserPage.getByRole('button', { name: 'Fallback log polling' })
  await expect(pollingToggle).toHaveAttribute('aria-pressed', 'true')
  await expect(pollingToggle).toContainText('Reconnecting')
  backend.streamMode = 'event'

  const streamAttemptsBeforePause = backend.streamUrls.length
  const listCallsBeforePause = backend.listCalls
  await pollingToggle.click()

  await expect(pollingToggle).toHaveAttribute('aria-pressed', 'false')
  await expect(pollingToggle).toContainText('Polling paused')
  expect(backend.streamUrls).toHaveLength(streamAttemptsBeforePause)
  await browserPage.clock.runFor(15_000)
  expect(backend.listCalls).toBe(listCallsBeforePause)

  const streamAttemptsBeforeResume = backend.streamUrls.length
  await pollingToggle.click()

  await expect(pollingToggle).toHaveAttribute('aria-pressed', 'true')
  await expect(pollingToggle).toContainText('Reconnecting')
  expect(backend.listCalls).toBe(listCallsBeforePause)
  expect(backend.streamUrls).toHaveLength(streamAttemptsBeforeResume)
  await browserPage.clock.runFor(5_000)
  await expect.poll(() => backend.listCalls).toBe(listCallsBeforePause + 1)

  await pollingToggle.focus()
  await browserPage.keyboard.press('Tab')
  await expect(browserPage.getByRole('button', { name: 'Clean up logs' })).toBeFocused()
  await browserPage.keyboard.press('Shift+Tab')
  await expect(pollingToggle).toBeFocused()
  await expect.poll(() => pollingToggle.evaluate((element) => getComputedStyle(element).outlineStyle)).toBe('solid')
  await browserPage.keyboard.press('Space')
  await expect(pollingToggle).toHaveAttribute('aria-pressed', 'false')
  await browserPage.keyboard.press('Enter')
  await expect(pollingToggle).toHaveAttribute('aria-pressed', 'true')
})

test('metadata-only export and previewed cleanup stay separated without dead-letter retry UI', async ({
  page: browserPage
}) => {
  const backend = await installLogsBackend(browserPage, { lifecycle: 'completed', streamMode: 'unavailable' })

  await browserPage.setViewportSize({ width: 1280, height: 900 })
  await browserPage.goto('/logs')
  const infoBanner = browserPage.getByRole('region', { name: 'System logs' })
  const ledgerControls = browserPage.getByRole('region', { name: 'Event log controls' })
  await expect(infoBanner.getByRole('button', { name: 'Clean up logs' })).toBeVisible()
  await expect(infoBanner.getByRole('button', { name: 'Export view' })).toHaveCount(0)
  await expect(ledgerControls.getByRole('button', { name: 'Export view' })).toBeVisible()
  await expect(ledgerControls.getByRole('button', { name: 'Clean up logs' })).toHaveCount(0)
  await expect(browserPage.getByRole('button', { name: 'Dead-letter retry' })).toHaveCount(0)

  await ledgerControls.getByRole('button', { name: 'Export view' }).click()
  const exportDialog = browserPage.getByRole('dialog', { name: 'Export current log view' })
  await exportDialog.getByLabel('Required audit reason').fill('retention review')
  await exportDialog.getByRole('button', { name: 'Download export' }).click()
  await expect(browserPage.getByText('Bounded log export downloaded.')).toBeVisible()
  expect(backend.operationBodies[0]).toContain('"includeArtifacts":false')
  await exportDialog.getByRole('button', { name: 'Cancel' }).click()

  await infoBanner.getByRole('button', { name: 'Clean up logs' }).click()
  const cleanupDialog = browserPage.getByRole('dialog', { name: 'Review log cleanup' })
  await expect(cleanupDialog.getByRole('slider', { name: 'Window start' })).toBeVisible()
  await expect(cleanupDialog.getByRole('slider', { name: 'Window end' })).toBeVisible()
  await expect(
    cleanupDialog.getByRole('button', { name: /Requests chart layer.*selected for cleanup preview/ })
  ).toHaveAttribute('data-state', 'on')
  await expect(
    cleanupDialog.getByRole('button', { name: /System chart layer.*retained during cleanup/ })
  ).toHaveAttribute('data-state', 'on')
  const layerControls = cleanupDialog.getByRole('button', { name: /chart layer/ })
  const desktopLayerLayout = await layerControls.evaluateAll(([requests, system, quic]) => {
    if (!(requests instanceof HTMLElement) || !(system instanceof HTMLElement) || !(quic instanceof HTMLElement)) {
      return { controlsFit: false, twoColumns: false }
    }
    const requestsBounds = requests.getBoundingClientRect()
    const systemBounds = system.getBoundingClientRect()
    const quicBounds = quic.getBoundingClientRect()
    return {
      controlsFit: [system, quic].every((control) => control.scrollWidth <= control.clientWidth),
      twoColumns: Math.abs(requestsBounds.top - systemBounds.top) <= 1 && quicBounds.top >= requestsBounds.bottom - 1
    }
  })
  expect.soft(desktopLayerLayout).toEqual({ controlsFit: true, twoColumns: true })

  await browserPage.setViewportSize({ width: 375, height: 900 })
  const mobileLayerLayout = await layerControls.evaluateAll(([requests, system, quic]) => {
    if (!(requests instanceof HTMLElement) || !(system instanceof HTMLElement) || !(quic instanceof HTMLElement)) {
      return { controlsFit: false, oneColumn: false }
    }
    const requestsBounds = requests.getBoundingClientRect()
    const systemBounds = system.getBoundingClientRect()
    const quicBounds = quic.getBoundingClientRect()
    return {
      controlsFit: [system, quic].every((control) => control.scrollWidth <= control.clientWidth),
      oneColumn:
        systemBounds.top >= requestsBounds.bottom - 1 &&
        quicBounds.top >= systemBounds.bottom - 1 &&
        Math.abs(requestsBounds.left - systemBounds.left) <= 1 &&
        Math.abs(systemBounds.left - quicBounds.left) <= 1
    }
  })
  expect.soft(mobileLayerLayout).toEqual({ controlsFit: true, oneColumn: true })

  const selectorPanelBottom = await cleanupDialog
    .getByRole('slider', { name: 'Window start' })
    .evaluate(
      (element) => element.parentElement?.parentElement?.getBoundingClientRect().bottom ?? Number.POSITIVE_INFINITY
    )
  const helperTop = await cleanupDialog
    .getByText('Drag either edge to narrow the loaded history. The server preview confirms what can be removed.', {
      exact: true
    })
    .evaluate((element) => element.getBoundingClientRect().top)
  expect.soft(helperTop).toBeGreaterThanOrEqual(selectorPanelBottom)

  const requestSummaryGeometry = await cleanupDialog
    .locator('p')
    .filter({
      hasText: /loaded request events? in this window\. Server review identifies removable terminal request groups\./
    })
    .evaluate((explanation) => {
      const summary = explanation.parentElement
      const counter = summary?.querySelector(':scope > span')
      if (!(summary instanceof HTMLElement) || !(counter instanceof HTMLElement)) return null
      const counterBounds = counter.getBoundingClientRect()
      const explanationBounds = explanation.getBoundingClientRect()
      const counterStyle = getComputedStyle(counter)
      return {
        counter: counter.textContent?.trim() ?? '',
        fontFamily: counterStyle.fontFamily,
        fontVariantNumeric: counterStyle.fontVariantNumeric,
        gap: explanationBounds.left - counterBounds.right
      }
    })
  expect.soft(requestSummaryGeometry?.counter ?? '').toMatch(/^\d+$/)
  expect.soft(requestSummaryGeometry?.fontFamily ?? '').toMatch(/JetBrains Mono|ui-monospace|Menlo|monospace/i)
  expect.soft(requestSummaryGeometry?.fontVariantNumeric ?? '').toContain('tabular-nums')
  expect.soft(requestSummaryGeometry?.gap ?? 0).toBeGreaterThan(0)

  await browserPage.setViewportSize({ width: 1280, height: 900 })
  await cleanupDialog.getByRole('button', { name: /System chart layer.*retained during cleanup/ }).click()
  await expect(
    cleanupDialog.getByRole('button', { name: /System chart layer.*retained during cleanup/ })
  ).toHaveAttribute('data-state', 'off')
  await cleanupDialog.getByLabel('Reason for removal').fill('retention review')
  await cleanupDialog.getByRole('button', { name: 'Review deletion' }).click()
  expect(JSON.parse(backend.operationBodies[1] ?? '')).toMatchObject({ requestLimit: 100 })
  const confirmDialog = browserPage.getByRole('dialog', { name: 'Review log cleanup' })
  await expect(confirmDialog.getByRole('heading', { name: 'Review log cleanup' })).toBeVisible()
  await expect(confirmDialog.getByText('Operational events stay retained.')).toBeVisible()
  await confirmDialog.getByRole('button', { name: 'Delete this batch' }).click()
  await expect(browserPage.getByText('Log cleanup completed.')).toBeVisible()
  await expect(confirmDialog.getByRole('heading', { name: 'Cleanup complete' })).toBeVisible()
  await confirmDialog.getByRole('button', { name: 'Close' }).click()
  expect(backend.operationBodies).toHaveLength(3)
})

test('partial cleanup retries retained artifact work and refetches the active ledger after successful receipts', async ({
  page: browserPage
}) => {
  const backend = await installLogsBackend(browserPage, {
    lifecycle: 'completed',
    cleanupRunResults: ['partial', 'completed']
  })

  await browserPage.goto('/logs')
  await expect(browserPage.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
  await expect.poll(() => backend.releaseStream).toBeDefined()
  // The SSE route is held open until released; without releasing it here the
  // connection never closes, `onerror` never fires, and `listCalls` can never
  // reach 2 (the assertion below hung indefinitely before this line existed).
  backend.releaseStream?.()
  await expect.poll(() => backend.listCalls).toBeGreaterThanOrEqual(2)
  const listCallsBeforeCleanup = backend.listCalls

  const confirmDialog = await previewScopedCleanup(browserPage)
  await expect(confirmDialog.getByRole('heading', { name: 'Review log cleanup' })).toBeVisible()
  expect(backend.listCalls).toBe(listCallsBeforeCleanup)

  await confirmDialog.getByRole('button', { name: 'Delete this batch' }).click()
  await expect(
    confirmDialog.getByText('Cleanup removed 1 request group; 1 linked file still needs attention.')
  ).toBeVisible()
  await expect(confirmDialog.getByRole('button', { name: 'Retry file removal' })).toBeVisible()
  await expect.poll(() => backend.listCalls).toBeGreaterThan(listCallsBeforeCleanup)
  const listCallsAfterPartialCleanup = backend.listCalls

  await confirmDialog.getByRole('button', { name: 'Retry file removal' }).click()
  await expect(confirmDialog.getByText('Log cleanup completed.')).toBeVisible()
  await expect(confirmDialog.getByRole('button', { name: 'Retry file removal' })).toHaveCount(0)
  await expect.poll(() => backend.listCalls).toBeGreaterThan(listCallsAfterPartialCleanup)

  expect(backend.cleanupRunBodies).toHaveLength(2)
  expect(JSON.parse(backend.cleanupRunBodies[1] ?? '')).toEqual(JSON.parse(backend.cleanupRunBodies[0] ?? ''))
})

test('failed cleanup mutation does not refetch the active ledger', async ({ page: browserPage }) => {
  const backend = await installLogsBackend(browserPage, {
    lifecycle: 'completed',
    cleanupRunResults: ['failure']
  })

  await browserPage.goto('/logs')
  await expect(browserPage.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
  await expect.poll(() => backend.releaseStream).toBeDefined()
  // See the comment in the preceding test — the stream must be released for
  // `listCalls` to ever reach 2.
  backend.releaseStream?.()
  await expect.poll(() => backend.listCalls).toBeGreaterThanOrEqual(2)
  const listCallsBeforeFailure = backend.listCalls

  const confirmDialog = await previewScopedCleanup(browserPage, 'failed cleanup should not refresh')
  await confirmDialog.getByRole('button', { name: 'Delete this batch' }).click()
  await expect(confirmDialog.getByText('Logs API request failed with HTTP 500')).toBeVisible()
  expect(backend.cleanupRunBodies).toHaveLength(1)
  expect(backend.listCalls).toBe(listCallsBeforeFailure)
})

test('terminal request deletion uses the details control and sends its audited operation', async ({
  page: browserPage
}) => {
  const backend = await installLogsBackend(browserPage, { lifecycle: 'completed' })

  await browserPage.goto('/logs')
  await expect(requestRow(browserPage, REQUEST_ID)).toBeVisible()
  await requestRow(browserPage, REQUEST_ID).click()
  await expect(browserPage.getByRole('dialog', { name: 'Request Inspector' })).toBeVisible()

  await browserPage.getByRole('button', { name: 'Delete terminal request' }).click()
  const deleteDialog = browserPage.getByRole('dialog', { name: 'Delete terminal request?' })
  await deleteDialog.getByLabel('Required audit reason').fill('remove invalid request')
  await deleteDialog.getByRole('button', { name: 'Confirm deletion' }).click()
  await expect(deleteDialog.getByText('Request removed.')).toBeVisible()
  expect(backend.deleteBodies).toHaveLength(1)
  expect(JSON.parse(backend.deleteBodies[0] ?? '')).toMatchObject({ reason: 'remove invalid request' })
})

test('older hosts present an accessible unsupported state', async ({ page: browserPage }) => {
  let streamAttempts = 0
  await browserPage.addInitScript(
    (storageKey) => window.localStorage.setItem(storageKey, 'live'),
    DATA_MODE_STORAGE_KEY
  )
  await browserPage.context().route('**/api/logs/events', async (route) => {
    streamAttempts += 1
    await route.abort('failed')
  })
  await browserPage
    .context()
    .route('**/api/logs/requests*', (route) => route.fulfill({ status: 404, json: { error: { code: 'unsupported' } } }))
  await browserPage
    .context()
    .route('**/api/logs/audit*', (route) => route.fulfill({ status: 404, json: { error: { code: 'unsupported' } } }))

  await browserPage.goto('/logs')
  await expect(browserPage.getByText('Request window unavailable')).toBeVisible()
  await expect(browserPage.getByText(/Upgrade the host to inspect request history here/)).toBeVisible()
  await expect(browserPage.getByText('Operational window unavailable')).toBeVisible()
  expect(streamAttempts).toBe(0)
})

test('opening a request row keeps ledger content mounted while the inspector appears', async ({
  page: browserPage
}) => {
  const backend = await installLogsBackend(browserPage, {
    delaySecondRequestsResponse: true,
    lifecycle: 'completed',
    streamMode: 'event'
  })
  await browserPage.setViewportSize({ width: 1200, height: 1200 })
  await browserPage.goto('/logs')
  await expect.poll(() => backend.releaseStream).toBeDefined()
  // See the comment in the "partial cleanup" test above — releasing the
  // stream delivers a `log_event` frame with no projected `request`, which
  // triggers the second (delayed) list refetch this test is set up to hold.
  backend.releaseStream?.()
  await expect.poll(() => backend.listCalls).toBe(2)

  const eventsOverTime = browserPage.getByRole('heading', { level: 2, name: 'Events Over Time' })
  // Renamed from "Request summary" (the loading-ghost's label,
  // LogsLedgerLoadingGhost.tsx) to "Request records" (LogsLedger.tsx) in
  // #1339; this locator was never updated because this suite never ran in CI
  // (#1372).
  const requestRecords = browserPage.getByRole('region', { name: 'Request records' })
  const eventControls = browserPage.getByRole('region', { name: 'Event log controls' })
  const row = requestRow(browserPage, REQUEST_ID)
  await expect(eventsOverTime).toBeVisible()
  await expect(requestRecords).toBeVisible()
  await expect(eventControls).toBeVisible()
  await expect(row).toBeVisible()
  const [eventsOverTimeElement, requestRecordsElement, eventControlsElement] = await Promise.all([
    eventsOverTime.evaluateHandle((element) => element),
    requestRecords.evaluateHandle((element) => element),
    eventControls.evaluateHandle((element) => element)
  ])
  const mountedElements = [eventsOverTimeElement, requestRecordsElement, eventControlsElement] as const
  const eventControlsTop = await eventControls.evaluate((element) => element.getBoundingClientRect().top)

  try {
    await row.click()

    const inspector = browserPage.getByRole('dialog', { name: 'Request Inspector' })
    await expect(inspector).toBeVisible()
    expect(backend.listCalls).toBe(2)
    for (const element of mountedElements) {
      expect(await element.evaluate((node) => node.isConnected)).toBe(true)
    }
    expect(await eventControlsElement.evaluate((element) => element.getBoundingClientRect().top)).toBe(eventControlsTop)
    await expect
      .poll(() =>
        inspector.evaluate((element) => getComputedStyle(element).getPropertyValue('--tw-enter-scale').trim())
      )
      .toBe('1')
  } finally {
    backend.releaseSecondRequestsResponse()
    backend.releaseStream?.()
    await Promise.all(mountedElements.map((element) => element.dispose()))
  }
})

test('unified event rows preserve filter state and restore focus after inspecting privacy-safe audit metadata', async ({
  page: browserPage
}) => {
  await installLogsBackend(browserPage, { lifecycle: 'completed', streamMode: 'unavailable' })
  await browserPage.goto('/logs')

  await expect(browserPage.getByRole('table', { name: 'MeshLLM event logs' })).toHaveCount(1)
  const auditRow = browserPage.getByRole('row', {
    name: 'Inspect operational event runtime_config_diagnostics_warning'
  })
  await auditRow.focus()
  await browserPage.keyboard.press('Enter')
  await expect(browserPage).toHaveURL(/inspectType=audit&inspectId=audit-0001/)

  const auditInspector = browserPage.getByRole('dialog', {
    name: 'Operational event runtime_config_diagnostics_warning'
  })
  for (const value of ['audit-0001', OCCURRED_AT, 'logs_api', 'runtime_config_diagnostics_warning', 'warning', '1']) {
    await expect(auditInspector.getByText(value, { exact: true })).toBeVisible()
  }

  await browserPage.keyboard.press('Escape')
  await expect(auditInspector).toHaveCount(0)
  await expect(auditRow).toBeFocused()

  const chartLegend = browserPage.getByRole('list', { name: 'Visible event categories' })
  await expect(chartLegend).toContainText('Requests')
  await expect(chartLegend).toContainText('System')

  await browserPage.getByRole('button', { name: /Filter event logs/ }).click()
  const filterDialog = browserPage.getByRole('dialog', { name: 'Event log filters' })
  await filterDialog.getByRole('checkbox', { name: /Requests/i }).uncheck()
  await expect(browserPage).toHaveURL(/categories=/)
  await expect(requestRow(browserPage, REQUEST_ID)).toHaveCount(0)
  await expect(auditRow).toBeVisible()
  await expect(chartLegend).not.toContainText('Requests')
  await expect(chartLegend).toContainText('System')
})

test('legacy request deep links open the canonical request inspector tab', async ({ page: browserPage }) => {
  await installLogsBackend(browserPage, { lifecycle: 'failed', streamMode: 'unavailable' })

  await browserPage.goto(`/logs/${REQUEST_ID}?tab=errors`)

  await expect
    .poll(() => {
      const currentUrl = new URL(browserPage.url())
      return {
        pathname: currentUrl.pathname,
        query: Object.fromEntries(currentUrl.searchParams.entries())
      }
    })
    .toEqual({
      pathname: '/logs',
      query: { inspectType: 'request', inspectId: REQUEST_ID, tab: 'diagnostics' }
    })
  const inspector = browserPage.getByRole('dialog', { name: 'Request Inspector' })
  await expect(inspector.getByRole('tab', { name: 'Diagnostics' })).toHaveAttribute('data-state', 'active')
  await expect(inspector.getByText('failed', { exact: true }).first()).toBeVisible()
})

test('inspector keeps long audit identity clear and controls touch-safe without bloating desktop', async ({
  page: browserPage
}) => {
  await installLogsBackend(browserPage, {
    auditIdentity: { entryId: LONG_AUDIT_ID, code: LONG_AUDIT_CODE },
    lifecycle: 'completed',
    streamMode: 'unavailable'
  })
  await browserPage.emulateMedia({ colorScheme: 'light', reducedMotion: 'reduce' })

  for (const width of [375, 768, 1280]) {
    await browserPage.setViewportSize({ width, height: 900 })
    await browserPage.goto('/logs')

    const auditRow = browserPage.getByRole('row', { name: `Inspect operational event ${LONG_AUDIT_CODE}` })
    await auditRow.focus()
    await browserPage.keyboard.press('Enter')

    const dialog = browserPage.getByRole('dialog', { name: `Operational event ${LONG_AUDIT_CODE}` })
    const title = dialog.getByRole('heading', { name: `Operational event ${LONG_AUDIT_CODE}` })
    const entryId = dialog.getByText(LONG_AUDIT_ID, { exact: true })
    const close = dialog.getByRole('button', { name: 'Close inspector' })

    await expect(title).toBeVisible()
    await expect(entryId).toBeVisible()
    await expect(close).toBeFocused()
    await expect.poll(() => title.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
    await expect.poll(() => entryId.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
    await expect
      .poll(() => browserPage.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
      .toBe(true)

    const titleRight = await title.evaluate((element) => element.getBoundingClientRect().right)
    const closeBounds = await close.evaluate((element) => {
      const bounds = element.getBoundingClientRect()
      return { height: bounds.height, left: bounds.left, width: bounds.width }
    })
    expect(titleRight).toBeLessThanOrEqual(closeBounds.left)
    if (width < 1024) {
      expect(closeBounds.height).toBeGreaterThanOrEqual(44)
      expect(closeBounds.width).toBeGreaterThanOrEqual(44)
    } else {
      expect(closeBounds.height).toBeLessThanOrEqual(36)
      expect(closeBounds.width).toBeLessThanOrEqual(36)
    }
    await browserPage.keyboard.press('Escape')
    await expect(dialog).toHaveCount(0)
    await expect(auditRow).toBeFocused()
  }
})

test('request inspector exposes a named embedded region and responsive tab targets', async ({ page: browserPage }) => {
  await installLogsBackend(browserPage, { lifecycle: 'completed', streamMode: 'unavailable' })
  await browserPage.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' })

  for (const width of [375, 768, 1280]) {
    await browserPage.setViewportSize({ width, height: width === 375 ? 520 : 900 })
    await browserPage.goto('/logs')
    await requestRow(browserPage, REQUEST_ID).click()

    const inspector = browserPage.getByRole('dialog', { name: 'Request Inspector' })
    const details = inspector.getByRole('region', { name: `Request details for ${REQUEST_ID}` })
    await expect(details).toBeVisible()

    const tabBounds = await inspector.getByRole('tab').evaluateAll((tabs) =>
      tabs.map((tab) => {
        const bounds = tab.getBoundingClientRect()
        return { height: bounds.height, width: bounds.width }
      })
    )
    for (const bounds of tabBounds) {
      if (width < 1024) {
        expect(bounds.height).toBeGreaterThanOrEqual(44)
        expect(bounds.width).toBeGreaterThanOrEqual(44)
      } else {
        expect(bounds.height).toBeLessThanOrEqual(36)
      }
    }
    if (width === 375) {
      const scrollBody = inspector.locator('[data-request-inspector-scroll="body"]')
      await expect(scrollBody).toHaveCount(1)
      await expect(scrollBody).toHaveAttribute('data-request-inspector-scroll', 'body')
      await expect.poll(() => scrollBody.evaluate((element) => element.scrollHeight > element.clientHeight)).toBe(true)
      await scrollBody.hover()
      await browserPage.mouse.wheel(0, 1_000)
      await expect.poll(() => scrollBody.evaluate((element) => element.scrollTop > 0)).toBe(true)
      await expect(inspector.getByRole('button', { name: 'Close inspector' })).toBeVisible()
    }

    await browserPage.keyboard.press('Escape')
    await expect(inspector).toHaveCount(0)
  }
})

test('logs pages stay accessible and unclipped across supported visual modes', async ({ page: browserPage }) => {
  await installLogsBackend(browserPage, { lifecycle: 'completed', streamMode: 'unavailable' })

  for (const colorScheme of ['light', 'dark'] as const) {
    await browserPage.emulateMedia({ colorScheme, reducedMotion: 'reduce' })
    for (const width of [375, 768, 1280]) {
      await browserPage.setViewportSize({ width, height: 900 })
      await browserPage.goto('/logs')
      await expect(browserPage.getByRole('heading', { level: 1, name: 'System logs' })).toBeVisible()
      await expect
        .poll(() => browserPage.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
        .toBe(true)
      const results = await new AxeBuilder({ page: browserPage })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
        .analyze()
      expect(
        results.violations.filter((violation) => ['serious', 'critical'].includes(violation.impact ?? ''))
      ).toEqual([])

      // 'Filter logs by time range' was removed in #1339; the chart's own
      // time-range selector is now the sole page-wide time-range control
      // (see LogsLedger.test.tsx "uses the chart selector as the only
      // page-wide time-range control").
      await tabTo(browserPage, browserPage.getByLabel('Chart time range'))
    }
  }
})

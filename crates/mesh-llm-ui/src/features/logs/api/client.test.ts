import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  HARNESS_LOG_AUDIT_FIXTURES,
  HARNESS_LOG_FIXTURES,
  HARNESS_LOG_SCENARIO_IDS
} from '@/features/logs/lib/log-fixtures'
import { LOG_PAYLOAD_RENDER_LIMIT_BYTES } from '@/features/logs/lib/log-payload-content'
import { LogsApiClient, LogsApiError } from './client'
import { LogArtifactId, LogOperationId, LogPageCursor, LogReplayCursor, LogRequestId } from './ids'
import { LogsDtoError } from './schemas'

const REQUEST_ID = '00000000-0000-4000-8000-000000000001'
const ARTIFACT_ID = '00000000-0000-4000-8000-000000000003'
const AUDIT_ID = '00000000-0000-4000-8000-000000000004'
const TIMESTAMP = '2026-08-04T12:00:00Z'

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

function artifactDto(contentState: string, contentBase64: string | null) {
  return {
    artifactId: ARTIFACT_ID,
    requestId: REQUEST_ID,
    occurredAt: TIMESTAMP,
    kind: 'request',
    mediaKind: 'text/plain',
    checksum: 'sha256:abc',
    bytes: 5,
    version: 1,
    redacted: contentState === 'available',
    truncated: false,
    contentState,
    contentBase64
  }
}

function paddedZeroBase64(byteLength: number): string {
  const encodedLength = 4 * Math.ceil(byteLength / 3)
  const paddingLength = (3 - (byteLength % 3)) % 3
  return `${'A'.repeat(encodedLength - paddingLength)}${'='.repeat(paddingLength)}`
}

function cleanupReceiptDto(
  operationId: LogOperationId,
  state: 'completed' | 'partial',
  failedArtifacts: number,
  hasMore: boolean
) {
  return {
    operationId: operationId.toString(),
    auditId: AUDIT_ID,
    cutoffBefore: TIMESTAMP,
    requestLimit: 1,
    scope: {
      source: 'durable',
      cutoffBefore: TIMESTAMP,
      requestLimit: 1
    },
    state,
    hasMore,
    selectionFingerprint: 'safe',
    planned: { requests: 1, events: 0, artifacts: 1, proxyRecords: 0, databaseRows: 2 },
    executed: { requests: 1, events: 0, artifacts: 1, proxyRecords: 0, databaseRows: 2 },
    artifactDeletion: {
      removed: 1,
      failed: failedArtifacts,
      failureClass: failedArtifacts > 0 ? 'unsafe_path' : undefined
    }
  }
}

function deleteReceiptDto(
  operationId: LogOperationId,
  state: 'completed' | 'pending' | 'partial',
  failedArtifacts: number
) {
  return {
    operationId: operationId.toString(),
    ...(state === 'pending' ? {} : { auditId: AUDIT_ID }),
    requestId: REQUEST_ID,
    state,
    selectionFingerprint: 'safe',
    planned: { requests: 1, events: 0, artifacts: 1, proxyRecords: 0, databaseRows: 2 },
    executed: { requests: 1, events: 0, artifacts: 1, proxyRecords: 0, databaseRows: 2 },
    artifactDeletion: {
      removed: 1,
      failed: failedArtifacts,
      failureClass: failedArtifacts > 0 ? 'unsafe_path' : undefined
    }
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('LogsApiClient', () => {
  it('preserves schema compatibility details from an unavailable response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(
        {
          error: {
            code: 'logging_schema_incompatible',
            message: 'the local log database schema is incompatible with this MeshLLM version',
            details: { schema_version: 2, supported_schema_version: 1 }
          }
        },
        503
      )
    )

    const error = await new LogsApiClient(fetchMock).listRequests().catch((reason: unknown) => reason)

    expect(error).toBeInstanceOf(LogsApiError)
    expect(error).toMatchObject({
      status: 503,
      code: 'logging_schema_incompatible',
      message: 'the local log database schema is incompatible with this MeshLLM version',
      details: { schemaVersion: 2, supportedSchemaVersion: 1 }
    })
  })

  it('binds the default browser fetch before issuing a request', async () => {
    const browserFetch = vi.fn(function (this: typeof globalThis, input: RequestInfo | URL) {
      expect(this).toBe(globalThis)
      expect(input).toBe('/api/logs/requests')
      return Promise.resolve(jsonResponse({ items: [], nextCursor: null }))
    })
    vi.stubGlobal('fetch', browserFetch)

    try {
      const result = await new LogsApiClient().listRequests()

      expect(result).toMatchObject({ state: 'supported', value: { items: [] } })
      expect(browserFetch).toHaveBeenCalledTimes(1)
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('uses an injected fetch without rebinding it', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ items: [], nextCursor: null }))

    await new LogsApiClient(fetchMock).listRequests()

    expect(fetchMock).toHaveBeenCalledWith('/api/logs/requests')
  })

  it('serializes repeated SSE channels and filters with an explicit replay cursor', () => {
    const client = new LogsApiClient(vi.fn())
    const url = client.logsEventSourceUrl({
      channels: ['requests', 'operations'],
      filters: [
        { key: 'from', value: '2026-08-03T00:00:00Z' },
        { key: 'route', value: 'chat' },
        { key: 'model', value: 'Qwen/Qwen3' },
        { key: 'model', value: 'Qwen/Qwen2.5' },
        { key: 'provider', value: 'reserve-a' },
        { key: 'engine', value: 'skippy' },
        { key: 'outcome', value: 'completed' }
      ],
      requestIds: [LogRequestId.parse(REQUEST_ID), LogRequestId.parse('00000000-0000-4000-8000-000000000002')],
      cursor: LogReplayCursor.parse('v1:2.3.4')
    })

    expect(url).toBe(
      '/api/logs/events?channel=requests&channel=operations&filter=from%3A2026-08-03T00%3A00%3A00Z&filter=route%3Achat&filter=model%3AQwen%2FQwen3&filter=model%3AQwen%2FQwen2.5&filter=provider%3Areserve-a&filter=engine%3Askippy&filter=outcome%3Acompleted&filter=request_id%3A00000000-0000-4000-8000-000000000001&filter=request_id%3A00000000-0000-4000-8000-000000000002&cursor=v1%3A2.3.4'
    )
  })

  it('returns a typed download only for available redacted artifact content', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(artifactDto('available', 'SGVsbG8=')))
    const client = new LogsApiClient(fetchMock)
    const result = await client.downloadArtifact(LogArtifactId.parse(ARTIFACT_ID))

    expect(result.state).toBe('download')
    if (result.state === 'download') {
      expect(new TextDecoder().decode(result.download.bytes)).toBe('Hello')
      expect(result.download.mediaType).toBe('text/plain')
      expect(result.download.fileName).toBe(`mesh-llm-log-${ARTIFACT_ID}.bin`)
    }
  })

  it('rejects declared artifact content over the render ceiling before atob', async () => {
    // Given
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        ...artifactDto('available', 'AA=='),
        bytes: LOG_PAYLOAD_RENDER_LIMIT_BYTES + 1
      })
    )
    const atobSpy = vi.spyOn(globalThis, 'atob')
    const client = new LogsApiClient(fetchMock)

    // When / Then
    await expect(client.downloadArtifact(LogArtifactId.parse(ARTIFACT_ID))).rejects.toBeInstanceOf(LogsDtoError)
    expect(atobSpy).not.toHaveBeenCalled()
  })

  it('rejects encoded artifact content over the render ceiling before atob', async () => {
    // Given
    const contentBase64 = paddedZeroBase64(LOG_PAYLOAD_RENDER_LIMIT_BYTES + 1)
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        ...artifactDto('available', contentBase64),
        bytes: LOG_PAYLOAD_RENDER_LIMIT_BYTES
      })
    )
    const atobSpy = vi.spyOn(globalThis, 'atob')
    const client = new LogsApiClient(fetchMock)

    // When / Then
    await expect(client.downloadArtifact(LogArtifactId.parse(ARTIFACT_ID))).rejects.toBeInstanceOf(LogsDtoError)
    expect(atobSpy).not.toHaveBeenCalled()
  })

  it('accepts exactly the padded artifact content boundary with one atob call', async () => {
    // Given
    const contentBase64 = paddedZeroBase64(LOG_PAYLOAD_RENDER_LIMIT_BYTES)
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        ...artifactDto('available', contentBase64),
        bytes: LOG_PAYLOAD_RENDER_LIMIT_BYTES
      })
    )
    const atobSpy = vi.spyOn(globalThis, 'atob').mockReturnValue('')
    const client = new LogsApiClient(fetchMock)

    // When
    const result = await client.downloadArtifact(LogArtifactId.parse(ARTIFACT_ID))

    // Then
    expect(result).toMatchObject({ state: 'download', download: { bytes: new Uint8Array() } })
    expect(atobSpy).toHaveBeenCalledOnce()
  })

  it('rejects non-canonical Base64 without calling atob', async () => {
    // Given
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(artifactDto('available', 'AB==')))
    const atobSpy = vi.spyOn(globalThis, 'atob')
    const client = new LogsApiClient(fetchMock)

    // When / Then
    await expect(client.downloadArtifact(LogArtifactId.parse(ARTIFACT_ID))).rejects.toBeInstanceOf(LogsDtoError)
    expect(atobSpy).not.toHaveBeenCalled()
  })

  it('keeps missing and corrupt artifacts out of the download path', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(artifactDto('missing', null)))
    const client = new LogsApiClient(fetchMock)
    const result = await client.downloadArtifact(LogArtifactId.parse(ARTIFACT_ID))

    expect(result).toMatchObject({ state: 'unavailable', artifact: { contentState: 'missing' } })
  })

  it('maps an older host 404 to unsupported after exactly one request', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ error: { code: 'not_found' } }, 404))
    const client = new LogsApiClient(fetchMock)
    const result = await client.listRequests({
      cursor: undefined,
      model: 'model-a',
      source: 'durable'
    })

    expect(result).toEqual({ state: 'unsupported' })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith('/api/logs/requests?model=model-a&source=durable')
  })

  it('serializes an opaque REST cursor without imposing backend limits', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        nextCursor: null
      })
    )
    const client = new LogsApiClient(fetchMock)
    await client.listRequests({ cursor: LogPageCursor.parse('opaque cursor+/=') })

    expect(fetchMock).toHaveBeenCalledWith('/api/logs/requests?cursor=opaque+cursor%2B%2F%3D')
  })

  it('serializes exact and prefix route exclusions as singular request keys', async () => {
    // Given
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ items: [], nextCursor: null }))

    // When
    await new LogsApiClient(fetchMock).listRequests({
      excludeRoute: 'models',
      excludeRoutePrefix: 'management_'
    })

    // Then
    expect(fetchMock).toHaveBeenCalledWith('/api/logs/requests?exclude_route=models&exclude_route_prefix=management_')
  })

  it('uses strict POST bodies for bounded export, cleanup, and request deletion', async () => {
    const operationId = LogOperationId.parse('00000000-0000-4000-8000-000000000002')
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          items: [],
          nextCursor: null,
          truncated: false,
          retryRequired: false,
          artifactContentIncluded: false
        })
      )
      .mockResolvedValueOnce(
        jsonResponse({
          operationId: operationId.toString(),
          auditId: AUDIT_ID,
          cutoffBefore: TIMESTAMP,
          requestLimit: 1,
          scope: {
            source: 'durable',
            cutoffBefore: TIMESTAMP,
            requestLimit: 1,
            from: '2026-08-01T00:00:00Z',
            to: TIMESTAMP,
            route: 'reserve',
            excludeRoute: 'models',
            model: 'Qwen/Qwen3',
            provider: 'reserve-a',
            engine: 'skippy',
            outcome: 'completed'
          },
          state: 'previewed',
          hasMore: false,
          selectionFingerprint: 'safe',
          planned: { requests: 1, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 1 },
          executed: { requests: 0, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 0 },
          artifactDeletion: { removed: 0, failed: 0 }
        })
      )
      .mockResolvedValueOnce(
        jsonResponse({
          operationId: operationId.toString(),
          auditId: AUDIT_ID,
          cutoffBefore: TIMESTAMP,
          requestLimit: 1,
          scope: {
            source: 'durable',
            cutoffBefore: TIMESTAMP,
            requestLimit: 1,
            from: '2026-08-01T00:00:00Z',
            to: TIMESTAMP,
            route: 'reserve',
            model: 'Qwen/Qwen3',
            provider: 'reserve-a',
            engine: 'skippy',
            outcome: 'completed'
          },
          state: 'completed',
          hasMore: false,
          selectionFingerprint: 'safe',
          planned: { requests: 1, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 1 },
          executed: { requests: 1, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 1 },
          artifactDeletion: { removed: 0, failed: 0 }
        })
      )
      .mockResolvedValueOnce(
        jsonResponse({
          operationId: operationId.toString(),
          auditId: AUDIT_ID,
          requestId: REQUEST_ID,
          state: 'completed',
          selectionFingerprint: 'safe',
          planned: { requests: 1, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 1 },
          executed: { requests: 1, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 1 },
          artifactDeletion: { removed: 0, failed: 0 }
        })
      )
    const client = new LogsApiClient(fetchMock)

    await client.exportRequests(
      { cursor: LogPageCursor.parse('page-2'), model: 'Qwen3' },
      { reason: 'audit copy', includeArtifacts: false }
    )
    const preview = await client.previewCleanup({
      operationId,
      cutoffBefore: TIMESTAMP,
      requestLimit: 1,
      source: 'durable',
      from: '2026-08-01T00:00:00Z',
      to: TIMESTAMP,
      route: 'reserve',
      excludeRoute: 'models',
      model: 'Qwen/Qwen3',
      provider: 'reserve-a',
      engine: 'skippy',
      outcome: 'completed',
      reason: 'retention'
    })
    const completed = await client.runCleanup({ operationId, reason: 'retention' })
    const deleted = await client.deleteRequest(LogRequestId.parse(REQUEST_ID), {
      operationId,
      reason: 'incident cleanup'
    })
    expect(preview.auditId.toString()).toBe(AUDIT_ID)
    expect(preview.scope.excludeRoute).toBe('models')
    expect(preview.scope).toMatchObject({
      source: 'durable',
      model: 'Qwen/Qwen3',
      outcome: 'completed'
    })
    expect(completed.auditId.toString()).toBe(AUDIT_ID)
    expect(completed.scope).toMatchObject({ source: 'durable', model: 'Qwen/Qwen3', outcome: 'completed' })
    expect(deleted.state).toBe('completed')
    if (deleted.state === 'completed') expect(deleted.auditId.toString()).toBe(AUDIT_ID)

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/logs/requests/export?cursor=page-2&model=Qwen3', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: 'audit copy', includeArtifacts: false })
    })
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/logs/cleanup/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        operationId: operationId.toString(),
        cutoffBefore: TIMESTAMP,
        requestLimit: 1,
        source: 'durable',
        from: '2026-08-01T00:00:00Z',
        to: TIMESTAMP,
        route: 'reserve',
        model: 'Qwen/Qwen3',
        provider: 'reserve-a',
        engine: 'skippy',
        outcome: 'completed',
        reason: 'retention'
      })
    })
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/logs/cleanup/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operationId: operationId.toString(), reason: 'retention' })
    })
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      `/api/logs/requests/${REQUEST_ID}/delete`,
      expect.objectContaining({ method: 'POST' })
    )
  })

  it('reuses a partial receipt operation and audit reason for cleanup and deletion retries', async () => {
    const operationId = LogOperationId.parse('00000000-0000-4000-8000-000000000002')
    const cleanupReason = 'retention cleanup'
    const deletionReason = 'incident cleanup'
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(cleanupReceiptDto(operationId, 'partial', 1, true)))
      .mockResolvedValueOnce(jsonResponse(cleanupReceiptDto(operationId, 'completed', 0, true)))
      .mockResolvedValueOnce(jsonResponse(deleteReceiptDto(operationId, 'partial', 1)))
      .mockResolvedValueOnce(jsonResponse(deleteReceiptDto(operationId, 'completed', 0)))
    const client = new LogsApiClient(fetchMock)

    const cleanupPartial = await client.runCleanup({ operationId, reason: cleanupReason })
    const cleanupCompleted = await client.runCleanup({ operationId: cleanupPartial.operationId, reason: cleanupReason })
    const deletionPartial = await client.deleteRequest(LogRequestId.parse(REQUEST_ID), {
      operationId,
      reason: deletionReason
    })
    const deletionCompleted = await client.deleteRequest(LogRequestId.parse(REQUEST_ID), {
      operationId: deletionPartial.operationId,
      reason: deletionReason
    })

    expect(cleanupPartial).toMatchObject({ state: 'partial', hasMore: true, artifactDeletion: { failed: 1 } })
    expect(cleanupCompleted).toMatchObject({ state: 'completed', hasMore: true, artifactDeletion: { failed: 0 } })
    expect(deletionPartial).toMatchObject({ state: 'partial', artifactDeletion: { failed: 1 } })
    expect(deletionCompleted).toMatchObject({ state: 'completed', artifactDeletion: { failed: 0 } })
    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/logs/cleanup/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operationId: operationId.toString(), reason: cleanupReason })
    })
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/logs/cleanup/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operationId: operationId.toString(), reason: cleanupReason })
    })
    expect(fetchMock).toHaveBeenNthCalledWith(3, `/api/logs/requests/${REQUEST_ID}/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operationId: operationId.toString(), reason: deletionReason })
    })
    expect(fetchMock).toHaveBeenNthCalledWith(4, `/api/logs/requests/${REQUEST_ID}/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operationId: operationId.toString(), reason: deletionReason })
    })
  })

  it('accepts HTTP 202 pending deletion without fabricating an audit identity', async () => {
    const operationId = LogOperationId.parse('00000000-0000-4000-8000-000000000002')
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(deleteReceiptDto(operationId, 'pending', 0), 202))

    const receipt = await new LogsApiClient(fetchMock).deleteRequest(LogRequestId.parse(REQUEST_ID), {
      operationId,
      reason: 'resume durable deletion'
    })

    expect(receipt).toMatchObject({ state: 'pending', operationId, auditId: undefined })
    expect(fetchMock).toHaveBeenCalledOnce()
  })

  it('serves every harness request read path without using the network', async () => {
    const fetchMock = vi.fn()
    const client = new LogsApiClient(fetchMock)

    const defaultPage = await client.listRequests({}, 'harness')
    const firstPage = await client.listRequests({ limit: 5 }, 'harness')
    const dropped = await client.listRequests({ outcome: 'dropped', source: 'durable' }, 'harness')
    const request = await client.getRequest(HARNESS_LOG_SCENARIO_IDS.failedRetry, 'harness')
    const events = await client.listRequestEvents(HARNESS_LOG_SCENARIO_IDS.failedRetry, {}, 'harness')
    const artifacts = await client.listRequestArtifacts(HARNESS_LOG_SCENARIO_IDS.failedRetry, {}, 'harness')
    const attempts = await client.listProxy({ requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry }, 'harness')
    const allAttempts = await client.listProxy({}, 'harness')
    const partialOutcome = await client.listRequests({ outcome: 'drop' }, 'harness')
    const availableArtifact = artifacts.items.find((artifact) => artifact.contentState === 'available')
    if (!availableArtifact) throw new TypeError('Expected an available harness artifact')
    const artifact = await client.getArtifact(availableArtifact.artifactId, 'harness')

    expect(defaultPage).toMatchObject({ state: 'supported' })
    if (defaultPage.state === 'supported') {
      expect(defaultPage.value.items).toHaveLength(HARNESS_LOG_FIXTURES.length)
      expect(defaultPage.value.nextCursor).toBeUndefined()
    }
    expect(firstPage).toMatchObject({
      state: 'supported',
      value: { items: expect.arrayContaining([expect.objectContaining({ outcome: 'active' })]) }
    })
    if (firstPage.state === 'supported') {
      expect(firstPage.value.items).toHaveLength(5)
      expect(firstPage.value.nextCursor?.toString()).toBe('5')
    }
    expect(dropped).toMatchObject({
      state: 'supported',
      value: { items: expect.arrayContaining([expect.objectContaining({ outcome: 'dropped', source: 'durable' })]) }
    })
    expect(request.requestId.toString()).toBe(HARNESS_LOG_SCENARIO_IDS.failedRetry.toString())
    expect(events.items.map((event) => event.kind)).toEqual(expect.arrayContaining(['stream_error', 'failed']))
    expect(artifacts.items.map((artifact) => artifact.contentState)).toEqual(
      expect.arrayContaining(['available', 'unavailable', 'missing', 'corrupt'])
    )
    expect(attempts.items).toHaveLength(2)
    expect(allAttempts.items.length).toBeGreaterThan(attempts.items.length)
    expect(partialOutcome).toMatchObject({ state: 'supported', value: { items: [] } })
    expect(artifact.artifactId).toEqual(availableArtifact.artifactId)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('filters harness request bounds by instant across offsets', async () => {
    // Given
    const target = HARNESS_LOG_FIXTURES.find(
      (request) => request.requestId.toString() === HARNESS_LOG_SCENARIO_IDS.activeStream.toString()
    )
    if (target === undefined) throw new TypeError('Expected the active stream harness fixture')
    const targetInstant = Date.parse(target.createdAt)
    const hourMilliseconds = 60 * 60 * 1000
    const fromSameInstant = new Date(targetInstant + hourMilliseconds).toISOString().replace('Z', '+01:00')
    const toSameInstant = new Date(targetInstant - hourMilliseconds).toISOString().replace('Z', '-01:00')
    const fetchMock = vi.fn()
    const client = new LogsApiClient(fetchMock)

    // When
    const result = await client.listRequests({ from: fromSameInstant, to: toSameInstant }, 'harness')

    // Then
    expect(result).toMatchObject({ state: 'supported' })
    if (result.state === 'supported') {
      expect(result.value.items).toContainEqual(target)
      expect(result.value.items.every((item) => Date.parse(item.createdAt) === targetInstant)).toBe(true)
    }
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('filters harness route exclusions before applying the page limit', async () => {
    // Given
    const fetchMock = vi.fn()
    const client = new LogsApiClient(fetchMock)

    // When
    const result = await client.listRequests(
      { limit: 2, excludeRoute: 'models', excludeRoutePrefix: 'management_' },
      'harness'
    )

    // Then
    expect(result).toMatchObject({ state: 'supported', value: { items: expect.any(Array) } })
    if (result.state === 'supported') {
      expect(result.value.items).toHaveLength(2)
      expect(
        result.value.items.every((item) => item.route !== 'models' && !item.route?.startsWith('management_'))
      ).toBe(true)
    }
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('returns a typed harness not-found error for an unknown request', async () => {
    const fetchMock = vi.fn()
    const client = new LogsApiClient(fetchMock)
    const unknownRequestId = LogRequestId.parse('00000000-0000-4000-8000-ffffffffffff')

    await expect(client.getRequest(unknownRequestId, 'harness')).rejects.toMatchObject({
      status: 404,
      code: 'not_found'
    })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('filters and paginates non-empty harness audits without using the network', async () => {
    const fetchMock = vi.fn()
    const client = new LogsApiClient(fetchMock)

    const firstPage = await client.listAudits({ limit: 3 }, 'harness')
    const secondPage = await client.listAudits({ cursor: LogPageCursor.parse('3'), limit: 3 }, 'harness')
    const meshWarnings = await client.listAudits({ source: 'mesh', severity: 'warning' }, 'harness')

    expect(firstPage).toMatchObject({ state: 'supported', value: { items: expect.any(Array) } })
    if (firstPage.state === 'supported') {
      expect(firstPage.value.items).toHaveLength(3)
      expect(firstPage.value.nextCursor?.toString()).toBe('3')
    }
    if (secondPage.state === 'supported') {
      expect(secondPage.value.items).toHaveLength(3)
      expect(secondPage.value.items[0]?.entryId).toBe(HARNESS_LOG_AUDIT_FIXTURES[3]?.entryId)
    }
    if (meshWarnings.state === 'supported') {
      expect(meshWarnings.value.items.length).toBeGreaterThan(0)
      expect(meshWarnings.value.items.every((entry) => entry.source === 'mesh' && entry.severity === 'warning')).toBe(
        true
      )
    }
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('keeps live audit query serialization and parsing unchanged', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [
          {
            entryId: 'audit-live',
            occurredAt: TIMESTAMP,
            source: 'logs_api',
            code: 'logging_cleanup_completed',
            severity: 'info',
            sequence: 7
          }
        ],
        nextCursor: null
      })
    )
    const client = new LogsApiClient(fetchMock)

    const result = await client.listAudits({
      cursor: LogPageCursor.parse('opaque audit cursor'),
      limit: 2,
      from: '2026-08-04T00:00:00Z',
      to: '2026-08-05T00:00:00Z',
      source: 'logs_api',
      severity: 'info'
    })

    expect(result).toMatchObject({ state: 'supported', value: { items: [{ entryId: 'audit-live' }] } })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/logs/audit?cursor=opaque+audit+cursor&limit=2&from=2026-08-04T00%3A00%3A00Z&to=2026-08-05T00%3A00%3A00Z&source=logs_api&severity=info'
    )
  })
})

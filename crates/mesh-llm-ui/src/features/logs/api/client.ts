import { env } from '@/lib/env'

import type { DataMode } from '@/lib/data-mode'
import { compareLogInstants } from '@/features/logs/lib/log-instant'
import { decodeBase64, isLogArtifactContentTooLarge } from '@/features/logs/lib/log-payload-content'
import {
  HARNESS_LOG_AUDIT_FIXTURES,
  HARNESS_LOG_FIXTURES,
  generateLifecycleEvents,
  generateArtifacts,
  generateProxyAttempts
} from '../lib/log-fixtures'

import { LogArtifactId, LogOperationId, LogPageCursor, LogRequestId } from './ids'
import {
  LogsDtoError,
  parseLogCleanupReceipt,
  parseLogDeleteReceipt,
  parseLogExport,
  parseLogArtifact,
  parseLogArtifactPage,
  parseLogAuditPage,
  parseLogLifecycleEventPage,
  parseLogProxyPage,
  parseLogRequest,
  parseLogRequestPage,
  type LogArtifact,
  type LogAuditEntry,
  type LogAuditPage,
  type LogCleanupReceipt,
  type LogCleanupOutcome,
  type LogDeleteReceipt,
  type LogExport,
  type LogLifecycleEvent,
  type LogProxyAttempt,
  type LogRequest,
  type LogsPage
} from './schemas'
import { serializeLogsSseSubscription, type LogsSseSubscription } from './sse'

export type LogsCapability<T> = { readonly state: 'supported'; readonly value: T } | { readonly state: 'unsupported' }

export class LogsApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string | undefined,
    readonly details?: LogsApiErrorDetails,
    message?: string
  ) {
    super(message ?? `Logs API request failed with HTTP ${status}`)
    this.name = 'LogsApiError'
  }
}

export type LogsApiErrorDetails = {
  readonly schemaVersion?: number
  readonly supportedSchemaVersion?: number
}

export type LogsRequestQuery = {
  readonly cursor?: LogPageCursor
  readonly limit?: number
  readonly from?: string
  readonly to?: string
  readonly route?: string
  readonly excludeRoute?: string
  readonly excludeRoutePrefix?: string
  readonly model?: string
  readonly provider?: string
  readonly engine?: string
  readonly status?: number
  readonly outcome?: string
  readonly source?: string
  readonly sort?: 'asc' | 'desc'
}

export type LogsPageQuery = {
  readonly cursor?: LogPageCursor
  readonly limit?: number
  readonly sort?: 'asc' | 'desc'
}

export type LogsProxyQuery = LogsPageQuery & {
  readonly requestId?: LogRequestId
  readonly provider?: string
  readonly engine?: string
  readonly status?: number
}

export type LogAuditQuery = {
  readonly cursor?: LogPageCursor
  readonly limit?: number
  readonly from?: string
  readonly to?: string
  readonly source?: string
  readonly severity?: string
}

export type LogArtifactDownload = {
  readonly artifact: Extract<LogArtifact, { readonly contentState: 'available' }>
  readonly bytes: Uint8Array
  readonly fileName: string
  readonly mediaType: string
}

export type LogArtifactDownloadResult =
  | { readonly state: 'download'; readonly download: LogArtifactDownload }
  | { readonly state: 'unavailable'; readonly artifact: LogArtifact }

export type LogExportRequest = {
  readonly reason: string
  readonly includeArtifacts: boolean
}

export type LogCleanupPreviewRequest = {
  readonly operationId: LogOperationId
  readonly cutoffBefore: string
  readonly requestLimit: number
  readonly source?: 'durable'
  readonly from?: string
  readonly to?: string
  readonly route?: string
  readonly excludeRoute?: string
  readonly model?: string
  readonly provider?: string
  readonly engine?: string
  readonly outcome?: LogCleanupOutcome
  readonly reason: string
}

export type LogCleanupRunRequest = {
  readonly operationId: LogOperationId
  readonly reason: string
}

export type LogDeleteRequest = {
  readonly operationId: LogOperationId
  readonly reason: string
}

type FetchFunction = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

function endpoint(path: string) {
  return `${env.managementApiUrl}${path}`
}

function setQueryValue(query: URLSearchParams, key: string, value: string | number | undefined) {
  if (value !== undefined) query.set(key, String(value))
}

function serializeRequestQuery(input: LogsRequestQuery) {
  const query = new URLSearchParams()
  setQueryValue(query, 'cursor', input.cursor?.toString())
  setQueryValue(query, 'limit', input.limit)
  setQueryValue(query, 'from', input.from)
  setQueryValue(query, 'to', input.to)
  setQueryValue(query, 'route', input.route)
  setQueryValue(query, 'exclude_route', input.excludeRoute)
  setQueryValue(query, 'exclude_route_prefix', input.excludeRoutePrefix)
  setQueryValue(query, 'model', input.model)
  setQueryValue(query, 'provider', input.provider)
  setQueryValue(query, 'engine', input.engine)
  setQueryValue(query, 'status', input.status)
  setQueryValue(query, 'outcome', input.outcome)
  setQueryValue(query, 'source', input.source)
  setQueryValue(query, 'sort', input.sort)
  return query.toString()
}

function serializePageQuery(input: LogsPageQuery) {
  const query = new URLSearchParams()
  setQueryValue(query, 'cursor', input.cursor?.toString())
  setQueryValue(query, 'limit', input.limit)
  setQueryValue(query, 'sort', input.sort)
  return query.toString()
}

function appendQuery(path: string, query: string) {
  return query.length === 0 ? path : `${path}?${query}`
}

async function responseJson(response: Response): Promise<unknown> {
  let text: string
  try {
    text = await response.text()
  } catch {
    throw new LogsDtoError()
  }
  try {
    return JSON.parse(text)
  } catch {
    throw new LogsDtoError()
  }
}

function parsedErrorBody(body: unknown) {
  if (!isRecord(body)) return {}
  const error = body['error']
  if (!isRecord(error)) return {}
  const code = error['code']
  const message = error['message']
  const rawDetails = error['details']
  const schemaVersion = isRecord(rawDetails) ? rawDetails['schema_version'] : undefined
  const supportedSchemaVersion = isRecord(rawDetails) ? rawDetails['supported_schema_version'] : undefined
  return {
    code: typeof code === 'string' ? code : undefined,
    message: typeof message === 'string' ? message : undefined,
    details:
      typeof schemaVersion === 'number' || typeof supportedSchemaVersion === 'number'
        ? {
            schemaVersion: typeof schemaVersion === 'number' ? schemaVersion : undefined,
            supportedSchemaVersion: typeof supportedSchemaVersion === 'number' ? supportedSchemaVersion : undefined
          }
        : undefined
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object'
}

async function responseError(response: Response) {
  let body: unknown
  try {
    body = await responseJson(response)
  } catch {
    body = undefined
  }
  const error = parsedErrorBody(body)
  return new LogsApiError(response.status, error.code, error.details, error.message)
}

function isUnsupportedResponse(response: Response, error: LogsApiError) {
  return response.status === 404 || error.code === 'unsupported' || error.code === 'not_implemented'
}

function safeMediaType(mediaKind: string | undefined) {
  return mediaKind && /^[a-z0-9!#$&^_.+-]+\/[a-z0-9!#$&^_.+-]+$/i.test(mediaKind)
    ? mediaKind
    : 'application/octet-stream'
}

const HARNESS_PAGE_SIZE = HARNESS_LOG_FIXTURES.length
const HARNESS_AUDIT_PAGE_SIZE = 10

function filterHarnessRequests(items: readonly LogRequest[], query: LogsRequestQuery): readonly LogRequest[] {
  return items.filter((item) => {
    if (query.from && compareLogInstants(item.createdAt, query.from) < 0) return false
    if (query.to && compareLogInstants(item.createdAt, query.to) > 0) return false
    if (query.route && item.route !== query.route) return false
    if (query.excludeRoute && item.route === query.excludeRoute) return false
    if (query.excludeRoutePrefix && item.route?.startsWith(query.excludeRoutePrefix)) return false
    if (query.model && item.model !== query.model) return false
    if (query.provider && item.provider !== query.provider) return false
    if (query.engine && item.engine !== query.engine) return false
    if (query.status != null && item.statusCode !== query.status) return false
    if (query.outcome && item.outcome !== query.outcome) return false
    if (query.source && item.source !== query.source) return false
    return true
  })
}

function paginateHarnessItems<T>(items: readonly T[], cursor?: number, limit = HARNESS_PAGE_SIZE): LogsPage<T> {
  const startIndex = Math.max(0, Number(cursor ?? 0))
  const pageSize = Math.max(1, limit)
  const sliced = items.slice(startIndex, startIndex + pageSize)
  const nextCursor =
    startIndex + sliced.length < items.length ? LogPageCursor.parse((startIndex + sliced.length).toString()) : undefined
  return { items: sliced, nextCursor }
}

function filterHarnessAudits(items: readonly LogAuditEntry[], query: LogAuditQuery): readonly LogAuditEntry[] {
  return items.filter((item) => {
    if (query.from && compareLogInstants(item.occurredAt, query.from) < 0) return false
    if (query.to && compareLogInstants(item.occurredAt, query.to) > 0) return false
    if (query.source && item.source !== query.source) return false
    if (query.severity && item.severity !== query.severity) return false
    return true
  })
}

export class LogsApiClient {
  readonly #fetch: FetchFunction

  constructor(fetchFunction?: FetchFunction) {
    this.#fetch = fetchFunction ?? fetch.bind(globalThis)
  }

  async listRequests(
    query: LogsRequestQuery = {},
    mode: DataMode = 'live'
  ): Promise<LogsCapability<LogsPage<LogRequest>>> {
    if (mode === 'harness') {
      const filtered = filterHarnessRequests(HARNESS_LOG_FIXTURES, query)
      const harnessCursor = query.cursor ? Number(query.cursor.toString()) : undefined
      return { state: 'supported', value: paginateHarnessItems(filtered, harnessCursor, query.limit) }
    }
    const response = await this.#fetch(endpoint(appendQuery('/api/logs/requests', serializeRequestQuery(query))))
    if (!response.ok) {
      const error = await responseError(response)
      if (isUnsupportedResponse(response, error)) return { state: 'unsupported' }
      throw error
    }
    return { state: 'supported', value: parseLogRequestPage(await responseJson(response)) }
  }
  async getRequest(requestId: LogRequestId, mode: DataMode = 'live'): Promise<LogRequest> {
    if (mode === 'harness') {
      const fixture = HARNESS_LOG_FIXTURES.find((f) => f.requestId.toString() === requestId.toString())
      if (!fixture) throw new LogsApiError(404, 'not_found')
      return fixture as LogRequest
    }
    return this.getJson(`/api/logs/requests/${encodeURIComponent(requestId.toString())}`, parseLogRequest)
  }

  async listRequestEvents(
    requestId: LogRequestId,
    query: LogsPageQuery = {},
    mode: DataMode = 'live'
  ): Promise<LogsPage<LogLifecycleEvent>> {
    if (mode === 'harness') {
      const events = generateLifecycleEvents(requestId.toString())
      return { items: events, nextCursor: undefined }
    }
    return this.getJson(
      appendQuery(`/api/logs/requests/${encodeURIComponent(requestId.toString())}/events`, serializePageQuery(query)),
      parseLogLifecycleEventPage
    )
  }

  async listRequestArtifacts(
    requestId: LogRequestId,
    query: LogsPageQuery = {},
    mode: DataMode = 'live'
  ): Promise<LogsPage<LogArtifact>> {
    if (mode === 'harness') {
      const artifacts = generateArtifacts(requestId.toString())
      return { items: artifacts, nextCursor: undefined }
    }
    return this.getJson(
      appendQuery(
        `/api/logs/requests/${encodeURIComponent(requestId.toString())}/artifacts`,
        serializePageQuery(query)
      ),
      parseLogArtifactPage
    )
  }
  async getArtifact(artifactId: LogArtifactId, mode: DataMode = 'live') {
    if (mode === 'harness') {
      const artifact = HARNESS_LOG_FIXTURES.flatMap((request) => generateArtifacts(request.requestId.toString())).find(
        (candidate) => candidate.artifactId.toString() === artifactId.toString()
      )
      if (!artifact) throw new LogsApiError(404, 'not_found')
      return artifact
    }
    return this.getJson(`/api/logs/artifacts/${encodeURIComponent(artifactId.toString())}`, parseLogArtifact)
  }
  async listProxy(query: LogsProxyQuery = {}, mode: DataMode = 'live'): Promise<LogsPage<LogProxyAttempt>> {
    if (mode === 'harness') {
      const attempts = (
        query.requestId
          ? generateProxyAttempts(query.requestId.toString())
          : HARNESS_LOG_FIXTURES.flatMap((request) => generateProxyAttempts(request.requestId.toString()))
      ).filter((attempt) => {
        if (query.provider && attempt.provider !== query.provider) return false
        if (query.engine && attempt.engine !== query.engine) return false
        if (query.status != null && attempt.statusCode !== query.status) return false
        return true
      })
      const harnessCursor = query.cursor ? Number(query.cursor.toString()) : undefined
      return paginateHarnessItems(attempts, harnessCursor, query.limit)
    }
    const params = new URLSearchParams(serializePageQuery(query))
    setQueryValue(params, 'request_id', query.requestId?.toString())
    setQueryValue(params, 'provider', query.provider)
    setQueryValue(params, 'engine', query.engine)
    setQueryValue(params, 'status', query.status)
    return this.getJson(appendQuery('/api/logs/proxy', params.toString()), parseLogProxyPage)
  }
  async listAudits(query: LogAuditQuery = {}, mode: DataMode = 'live'): Promise<LogsCapability<LogAuditPage>> {
    if (mode === 'harness') {
      const filtered = filterHarnessAudits(HARNESS_LOG_AUDIT_FIXTURES, query)
      const harnessCursor = query.cursor ? Number(query.cursor.toString()) : undefined
      return {
        state: 'supported',
        value: paginateHarnessItems(filtered, harnessCursor, query.limit ?? HARNESS_AUDIT_PAGE_SIZE)
      }
    }
    const params = new URLSearchParams()
    setQueryValue(params, 'cursor', query.cursor?.toString())
    setQueryValue(params, 'limit', query.limit)
    setQueryValue(params, 'from', query.from)
    setQueryValue(params, 'to', query.to)
    setQueryValue(params, 'source', query.source)
    setQueryValue(params, 'severity', query.severity)
    const response = await this.#fetch(endpoint(appendQuery('/api/logs/audit', params.toString())))
    if (!response.ok) {
      const error = await responseError(response)
      if (isUnsupportedResponse(response, error)) return { state: 'unsupported' }
      throw error
    }
    return { state: 'supported', value: parseLogAuditPage(await responseJson(response)) }
  }
  logsEventSourceUrl(subscription: LogsSseSubscription) {
    return endpoint(appendQuery('/api/logs/events', serializeLogsSseSubscription(subscription)))
  }
  async downloadArtifact(artifactId: LogArtifactId): Promise<LogArtifactDownloadResult> {
    const artifact = await this.getArtifact(artifactId)
    if (artifact.contentState !== 'available' || artifact.contentBase64 === undefined) {
      return { state: 'unavailable', artifact }
    }
    if (isLogArtifactContentTooLarge(artifact)) throw new LogsDtoError()
    const decoded = decodeBase64(artifact.contentBase64)
    if (decoded === undefined) throw new LogsDtoError()
    return {
      state: 'download',
      download: {
        artifact,
        bytes: Uint8Array.from(decoded, (character) => character.charCodeAt(0)),
        fileName: `mesh-llm-log-${artifact.artifactId.toString()}.bin`,
        mediaType: safeMediaType(artifact.mediaKind)
      }
    }
  }

  async exportRequests(query: LogsRequestQuery, request: LogExportRequest): Promise<LogExport> {
    const result = await this.postJson(
      appendQuery('/api/logs/requests/export', serializeRequestQuery(query)),
      { reason: request.reason, includeArtifacts: request.includeArtifacts },
      parseLogExport
    )
    if (!request.includeArtifacts && result.artifactContentIncluded) throw new LogsDtoError()
    return result
  }

  async previewCleanup(request: LogCleanupPreviewRequest): Promise<LogCleanupReceipt> {
    return this.postJson(
      '/api/logs/cleanup/preview',
      {
        operationId: request.operationId.toString(),
        cutoffBefore: request.cutoffBefore,
        requestLimit: request.requestLimit,
        source: request.source,
        from: request.from,
        to: request.to,
        route: request.route,
        model: request.model,
        provider: request.provider,
        engine: request.engine,
        outcome: request.outcome,
        reason: request.reason
      },
      parseLogCleanupReceipt
    )
  }

  async runCleanup(request: LogCleanupRunRequest): Promise<LogCleanupReceipt> {
    return this.postJson(
      '/api/logs/cleanup/run',
      { operationId: request.operationId.toString(), reason: request.reason },
      parseLogCleanupReceipt
    )
  }

  async deleteRequest(requestId: LogRequestId, request: LogDeleteRequest): Promise<LogDeleteReceipt> {
    return this.postJson(
      `/api/logs/requests/${encodeURIComponent(requestId.toString())}/delete`,
      { operationId: request.operationId.toString(), reason: request.reason },
      parseLogDeleteReceipt
    )
  }

  private async getJson<T>(path: string, parser: (input: unknown) => T): Promise<T> {
    const response = await this.#fetch(endpoint(path))
    if (!response.ok) throw await responseError(response)
    return parser(await responseJson(response))
  }

  private async postJson<T>(path: string, body: unknown, parser: (input: unknown) => T): Promise<T> {
    const response = await this.#fetch(endpoint(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    if (!response.ok) throw await responseError(response)
    return parser(await responseJson(response))
  }
}

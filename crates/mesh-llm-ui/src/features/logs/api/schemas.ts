import * as v from 'valibot'
import {
  LogArtifactId,
  LogAuditId,
  LogEventId,
  LogOperationId,
  LogPageCursor,
  LogRequestId,
  type LogReplayChannel
} from './ids'

import { LogsDtoError } from './schemas/types'
import type {
  LogArtifact,
  LogAuditPage,
  LogCleanupReceipt,
  LogDeleteReceipt,
  LogEventKind,
  LogExport,
  LogLifecycleEvent,
  LogProxyAttempt,
  LogRequest,
  LogsPage
} from './schemas/types'
type LogArtifactBase = {
  readonly artifactId: LogArtifactId
  readonly requestId: LogRequestId
  readonly occurredAt: string
  readonly kind: string
  readonly mediaKind: string | undefined
  readonly checksum: string | undefined
  readonly bytes: number
  readonly version: number
  readonly redacted: boolean
  readonly truncated: boolean
}
export { LogsDtoError } from './schemas/types'
export type {
  LogArtifact,
  LogArtifactDeletion,
  LogArtifactUnavailableReason,
  LogAuditEntry,
  LogAuditPage,
  LogAuditSeverity,
  LogAuditSource,
  LogCleanupOutcome,
  LogCleanupReceipt,
  LogCleanupScope,
  LogDeleteReceipt,
  LogEventKind,
  LogExport,
  LogExportItem,
  LogLifecycleEvent,
  LogMaintenanceCounts,
  LogOutcome,
  LogProxyAttempt,
  LogRequest,
  LogSource,
  LogsPage
} from './schemas/types'

const outcomeSchema = v.picklist(['active', 'completed', 'failed', 'rejected', 'cancelled', 'dropped'])
const cleanupOutcomeSchema = v.picklist(['completed', 'failed', 'rejected', 'cancelled', 'dropped'])
const sourceSchema = v.picklist(['active', 'durable'])
const eventKindSchema = v.picklist([
  'admitted',
  'route_selected',
  'attempt_started',
  'attempt_completed',
  'attempt_failed',
  'backend_stream_first_item',
  'stream_started',
  'stream_chunk',
  'stream_completed',
  'usage_recorded',
  'stream_error',
  'audit_error',
  'completed',
  'failed',
  'rejected',
  'cancelled',
  'dropped'
])
const channelSchema = v.picklist(['requests', 'operations', 'system'])
const auditSourceSchema = v.picklist(['logging_service', 'logs_api', 'runtime', 'mesh', 'cli'])
const auditSeveritySchema = v.picklist(['info', 'warning', 'error'])
const artifactUnavailableReasonSchema = v.picklist([
  'streaming_response_not_assembled',
  'response_body_not_bounded',
  'capture_content_limit_exceeded',
  'capture_memory_budget_exceeded',
  'artifact_capture_disabled',
  'artifact_capture_failed'
])
const safeIntegerSchema = v.pipe(
  v.number(),
  v.integer(),
  v.check((value: number) => Number.isSafeInteger(value))
)
const nonNegativeIntegerSchema = v.pipe(safeIntegerSchema, v.minValue(0))
const statusCodeSchema = v.pipe(safeIntegerSchema, v.minValue(100), v.maxValue(599))
const timestampSchema = v.pipe(
  v.string(),
  v.check((value) => !Number.isNaN(Date.parse(value)))
)
const requestIdSchema = v.pipe(
  v.string(),
  v.transform((value) => LogRequestId.parse(value))
)
const eventIdSchema = v.pipe(
  v.string(),
  v.transform((value) => LogEventId.parse(value))
)
const artifactIdSchema = v.pipe(
  v.string(),
  v.transform((value) => LogArtifactId.parse(value))
)
const operationIdSchema = v.pipe(
  v.string(),
  v.transform((value) => LogOperationId.parse(value))
)
const auditIdSchema = v.pipe(
  v.string(),
  v.transform((value) => LogAuditId.parse(value))
)
const cleanupScopeFilterSchema = v.pipe(
  v.string(),
  v.minLength(1),
  v.maxLength(128),
  v.check((value) => {
    const pathOrSecretShaped =
      value.startsWith('/') ||
      value.startsWith('~/') ||
      value[1] === ':' ||
      /[\\?#=&]/.test(value) ||
      value.includes('://')
    const hasControlCharacter = Array.from(value).some(
      (character) => character <= String.fromCharCode(31) || character === String.fromCharCode(127)
    )
    return value === value.trim() && !hasControlCharacter && !pathOrSecretShaped
  })
)

const requestSchema = v.object({
  requestId: requestIdSchema,
  outcome: outcomeSchema,
  createdAt: timestampSchema,
  terminalAt: v.nullable(timestampSchema),
  route: v.nullable(v.string()),
  model: v.nullable(v.string()),
  provider: v.nullable(v.string()),
  engine: v.nullable(v.string()),
  statusCode: v.nullable(statusCodeSchema),
  source: sourceSchema
})

const lifecycleEventSchema = v.object({
  eventId: eventIdSchema,
  requestId: requestIdSchema,
  occurredAt: timestampSchema,
  kind: eventKindSchema,
  model: v.nullable(v.string()),
  provider: v.nullable(v.string()),
  engine: v.nullable(v.string()),
  attemptId: v.nullable(v.string()),
  statusCode: v.nullable(statusCodeSchema),
  durationMs: v.nullable(nonNegativeIntegerSchema),
  tokens: v.nullable(nonNegativeIntegerSchema),
  promptTokens: v.optional(v.nullable(nonNegativeIntegerSchema)),
  cachedPromptTokens: v.optional(v.nullable(nonNegativeIntegerSchema)),
  completionTokens: v.optional(v.nullable(nonNegativeIntegerSchema)),
  totalTokens: v.optional(v.nullable(nonNegativeIntegerSchema))
})

const artifactSchema = v.object({
  artifactId: artifactIdSchema,
  requestId: requestIdSchema,
  occurredAt: timestampSchema,
  kind: v.string(),
  mediaKind: v.nullable(v.string()),
  checksum: v.nullable(v.string()),
  bytes: v.pipe(safeIntegerSchema, v.minValue(0)),
  version: v.pipe(safeIntegerSchema, v.minValue(1)),
  redacted: v.boolean(),
  truncated: v.boolean(),
  contentState: v.picklist(['available', 'unavailable', 'missing', 'corrupt']),
  unavailableReason: v.optional(v.nullable(artifactUnavailableReasonSchema)),
  contentBase64: v.nullable(v.string())
})

const proxySchema = v.object({
  attemptId: v.string(),
  requestId: requestIdSchema,
  occurredAt: timestampSchema,
  target: v.string(),
  provider: v.nullable(v.string()),
  engine: v.nullable(v.string()),
  startedAt: v.nullable(timestampSchema),
  completedAt: v.nullable(timestampSchema),
  statusCode: v.nullable(statusCodeSchema)
})

const maintenanceCountsSchema = v.object({
  requests: nonNegativeIntegerSchema,
  events: nonNegativeIntegerSchema,
  artifacts: nonNegativeIntegerSchema,
  proxyRecords: nonNegativeIntegerSchema,
  databaseRows: nonNegativeIntegerSchema
})
const artifactDeletionSchema = v.object({
  removed: nonNegativeIntegerSchema,
  failed: nonNegativeIntegerSchema,
  failureClass: v.optional(v.picklist(['io', 'unsafe_path']))
})
const cleanupScopeSchema = v.strictObject({
  source: v.literal('durable'),
  cutoffBefore: timestampSchema,
  requestLimit: v.pipe(nonNegativeIntegerSchema, v.minValue(1)),
  from: v.optional(timestampSchema),
  to: v.optional(timestampSchema),
  route: v.optional(cleanupScopeFilterSchema),
  model: v.optional(cleanupScopeFilterSchema),
  provider: v.optional(cleanupScopeFilterSchema),
  engine: v.optional(cleanupScopeFilterSchema),
  outcome: v.optional(cleanupOutcomeSchema)
})
const cleanupReceiptSchema = v.object({
  operationId: operationIdSchema,
  auditId: auditIdSchema,
  cutoffBefore: timestampSchema,
  requestLimit: v.pipe(nonNegativeIntegerSchema, v.minValue(1)),
  scope: cleanupScopeSchema,
  state: v.picklist(['previewed', 'completed', 'partial']),
  hasMore: v.boolean(),
  selectionFingerprint: v.pipe(v.string(), v.minLength(1)),
  planned: maintenanceCountsSchema,
  executed: maintenanceCountsSchema,
  artifactDeletion: artifactDeletionSchema
})
const deleteReceiptSchema = v.object({
  operationId: operationIdSchema,
  auditId: v.optional(v.nullable(auditIdSchema)),
  requestId: requestIdSchema,
  state: v.picklist(['completed', 'pending', 'partial']),
  selectionFingerprint: v.pipe(v.string(), v.minLength(1)),
  planned: maintenanceCountsSchema,
  executed: maintenanceCountsSchema,
  artifactDeletion: artifactDeletionSchema
})
const exportItemSchema = v.object({
  summary: requestSchema,
  events: v.array(lifecycleEventSchema),
  artifacts: v.array(artifactSchema),
  childIncomplete: v.boolean()
})
const exportSchema = v.object({
  items: v.array(exportItemSchema),
  nextCursor: v.nullable(v.string()),
  truncated: v.boolean(),
  retryRequired: v.boolean(),
  artifactContentIncluded: v.boolean()
})

const replayEventSchema = v.object({
  eventId: eventIdSchema,
  requestId: requestIdSchema,
  occurredAt: timestampSchema,
  channel: channelSchema,
  sequence: v.pipe(nonNegativeIntegerSchema, v.minValue(1)),
  kind: eventKindSchema,
  request: v.optional(requestSchema)
})

const replayGapSchema = v.object({
  channel: channelSchema,
  fromSequence: v.pipe(nonNegativeIntegerSchema, v.minValue(1)),
  toSequence: v.pipe(nonNegativeIntegerSchema, v.minValue(1)),
  recovery: v.object({
    endpoint: v.literal('/api/logs/requests'),
    cursor: v.optional(v.nullable(v.string()))
  })
})

const auditEntrySchema = v.object({
  entryId: v.pipe(v.string(), v.minLength(1)),
  occurredAt: timestampSchema,
  source: auditSourceSchema,
  code: v.string(),
  severity: v.optional(auditSeveritySchema),
  sequence: v.pipe(nonNegativeIntegerSchema, v.minValue(1)),
  contextVersion: v.optional(v.literal(1)),
  subjectKind: v.optional(
    v.union([v.literal('runtime'), v.literal('model'), v.literal('runtime_instance'), v.literal('cli_command')])
  ),
  subjectId: v.optional(v.pipe(v.string(), v.minLength(1), v.maxLength(256))),
  operationId: v.optional(v.pipe(v.string(), v.minLength(1), v.maxLength(256))),
  requestId: v.optional(v.pipe(v.string(), v.minLength(1), v.maxLength(256))),
  reasonCode: v.optional(v.pipe(v.string(), v.minLength(1), v.maxLength(64))),
  outcome: v.optional(v.pipe(v.string(), v.minLength(1), v.maxLength(64))),
  durationMs: v.optional(nonNegativeIntegerSchema),
  numericSummaries: v.optional(v.record(v.string(), nonNegativeIntegerSchema))
})

const auditGapSchema = v.object({
  channel: v.literal('audit'),
  fromSequence: v.pipe(nonNegativeIntegerSchema, v.minValue(1)),
  toSequence: v.pipe(nonNegativeIntegerSchema, v.minValue(1)),
  recovery: v.object({
    endpoint: v.literal('/api/logs/audit'),
    cursor: v.optional(v.nullable(v.string()))
  })
})

function parseRequestWire(input: unknown) {
  try {
    return v.parse(requestSchema, input)
  } catch {
    throw new LogsDtoError()
  }
}

function parseLifecycleEventWire(input: unknown) {
  try {
    return v.parse(lifecycleEventSchema, input)
  } catch {
    throw new LogsDtoError()
  }
}

function parseArtifactWire(input: unknown) {
  try {
    return v.parse(artifactSchema, input)
  } catch {
    throw new LogsDtoError()
  }
}

function parseProxyWire(input: unknown) {
  try {
    return v.parse(proxySchema, input)
  } catch {
    throw new LogsDtoError()
  }
}

function parseReplayEventWire(input: unknown) {
  try {
    return v.parse(replayEventSchema, input)
  } catch {
    throw new LogsDtoError()
  }
}

function parseReplayGapWire(input: unknown) {
  try {
    return v.parse(replayGapSchema, input)
  } catch {
    throw new LogsDtoError()
  }
}

export function parseAuditEntryWire(input: unknown) {
  try {
    return v.parse(auditEntrySchema, input)
  } catch {
    throw new LogsDtoError()
  }
}

export function parseAuditGapWire(input: unknown) {
  try {
    return v.parse(auditGapSchema, input)
  } catch {
    throw new LogsDtoError()
  }
}

function optional<T>(value: T | null | undefined): T | undefined {
  return value ?? undefined
}

function parsePageCursor(value: string | null) {
  let nextCursor: LogPageCursor | undefined
  try {
    nextCursor = value === null ? undefined : LogPageCursor.parse(value)
  } catch {
    throw new LogsDtoError()
  }
  return nextCursor
}

function toLogRequest(value: ReturnType<typeof parseRequestWire>): LogRequest {
  return {
    ...value,
    terminalAt: optional(value.terminalAt),
    route: optional(value.route),
    model: optional(value.model),
    provider: optional(value.provider),
    engine: optional(value.engine),
    statusCode: value.statusCode ?? undefined
  }
}

export function parseLogRequest(input: unknown): LogRequest {
  return toLogRequest(parseRequestWire(input))
}

export function parseLogRequestPage(input: unknown): LogsPage<LogRequest> {
  try {
    const page = v.parse(v.object({ items: v.array(requestSchema), nextCursor: v.nullable(v.string()) }), input)
    return { items: page.items.map(toLogRequest), nextCursor: parsePageCursor(page.nextCursor) }
  } catch (error) {
    if (error instanceof LogsDtoError) throw error
    throw new LogsDtoError()
  }
}

function toLogLifecycleEvent(value: ReturnType<typeof parseLifecycleEventWire>): LogLifecycleEvent {
  return {
    ...value,
    model: optional(value.model),
    provider: optional(value.provider),
    engine: optional(value.engine),
    attemptId: optional(value.attemptId),
    statusCode: value.statusCode ?? undefined,
    durationMs: value.durationMs ?? undefined,
    tokens: value.tokens ?? undefined,
    promptTokens: value.promptTokens ?? undefined,
    cachedPromptTokens: value.cachedPromptTokens ?? undefined,
    completionTokens: value.completionTokens ?? undefined,
    totalTokens: value.totalTokens ?? undefined
  }
}

export function parseLogLifecycleEvent(input: unknown): LogLifecycleEvent {
  return toLogLifecycleEvent(parseLifecycleEventWire(input))
}

export function parseLogLifecycleEventPage(input: unknown): LogsPage<LogLifecycleEvent> {
  try {
    const page = v.parse(v.object({ items: v.array(lifecycleEventSchema), nextCursor: v.nullable(v.string()) }), input)
    return { items: page.items.map(toLogLifecycleEvent), nextCursor: parsePageCursor(page.nextCursor) }
  } catch (error) {
    if (error instanceof LogsDtoError) throw error
    throw new LogsDtoError()
  }
}

function toLogArtifact(value: ReturnType<typeof parseArtifactWire>): LogArtifact {
  const base: LogArtifactBase = {
    artifactId: value.artifactId,
    requestId: value.requestId,
    occurredAt: value.occurredAt,
    kind: value.kind,
    mediaKind: optional(value.mediaKind),
    checksum: optional(value.checksum),
    bytes: value.bytes,
    version: value.version,
    redacted: value.redacted,
    truncated: value.truncated
  }
  switch (value.contentState) {
    case 'available':
      if (!value.redacted || optional(value.unavailableReason) !== undefined) throw new LogsDtoError()
      return { ...base, contentState: 'available', contentBase64: optional(value.contentBase64) }
    case 'unavailable':
      if (value.contentBase64 !== null) throw new LogsDtoError()
      return {
        ...base,
        contentState: 'unavailable',
        unavailableReason: optional(value.unavailableReason),
        contentBase64: undefined
      }
    case 'missing':
    case 'corrupt':
      if (value.contentBase64 !== null || optional(value.unavailableReason) !== undefined) throw new LogsDtoError()
      return { ...base, contentState: value.contentState, contentBase64: undefined }
  }
}

export function parseLogArtifact(input: unknown): LogArtifact {
  return toLogArtifact(parseArtifactWire(input))
}

export function parseLogArtifactPage(input: unknown): LogsPage<LogArtifact> {
  try {
    const page = v.parse(v.object({ items: v.array(artifactSchema), nextCursor: v.nullable(v.string()) }), input)
    return { items: page.items.map(toLogArtifact), nextCursor: parsePageCursor(page.nextCursor) }
  } catch (error) {
    if (error instanceof LogsDtoError) throw error
    throw new LogsDtoError()
  }
}

function isSafeProxyTarget(value: string) {
  if (value === 'opaque') return true
  try {
    const url = new URL(value)
    return (
      (url.protocol === 'http:' || url.protocol === 'https:') &&
      url.hostname.length > 0 &&
      (url.port === '' || (Number.isInteger(Number(url.port)) && Number(url.port) >= 1 && Number(url.port) <= 65535)) &&
      url.username === '' &&
      url.password === '' &&
      url.pathname === '/' &&
      url.search === '' &&
      url.hash === ''
    )
  } catch {
    return false
  }
}

function toLogProxyAttempt(value: ReturnType<typeof parseProxyWire>): LogProxyAttempt {
  if (!isSafeProxyTarget(value.target)) throw new LogsDtoError()
  return {
    ...value,
    provider: optional(value.provider),
    engine: optional(value.engine),
    startedAt: optional(value.startedAt),
    completedAt: optional(value.completedAt),
    statusCode: value.statusCode ?? undefined
  }
}

export function parseLogProxyAttempt(input: unknown): LogProxyAttempt {
  return toLogProxyAttempt(parseProxyWire(input))
}

export function parseLogProxyPage(input: unknown): LogsPage<LogProxyAttempt> {
  try {
    const page = v.parse(v.object({ items: v.array(proxySchema), nextCursor: v.nullable(v.string()) }), input)
    return { items: page.items.map(toLogProxyAttempt), nextCursor: parsePageCursor(page.nextCursor) }
  } catch (error) {
    if (error instanceof LogsDtoError) throw error
    throw new LogsDtoError()
  }
}

export function parseLogAuditPage(input: unknown): LogAuditPage {
  try {
    const page = v.parse(v.object({ items: v.array(auditEntrySchema), nextCursor: v.nullable(v.string()) }), input)
    return { items: page.items, nextCursor: parsePageCursor(page.nextCursor) }
  } catch (error) {
    if (error instanceof LogsDtoError) throw error
    throw new LogsDtoError()
  }
}

function parseOperation<T>(schema: v.BaseSchema<unknown, T, v.BaseIssue<unknown>>, input: unknown): T {
  try {
    return v.parse(schema, input)
  } catch {
    throw new LogsDtoError()
  }
}

export function parseLogCleanupReceipt(input: unknown): LogCleanupReceipt {
  const receipt = parseOperation(cleanupReceiptSchema, input)
  if (receipt.scope.cutoffBefore !== receipt.cutoffBefore || receipt.scope.requestLimit !== receipt.requestLimit) {
    throw new LogsDtoError()
  }
  return {
    ...receipt,
    artifactDeletion: { ...receipt.artifactDeletion, failureClass: receipt.artifactDeletion.failureClass }
  }
}

export function parseLogDeleteReceipt(input: unknown): LogDeleteReceipt {
  const receipt = parseOperation(deleteReceiptSchema, input)
  const auditId = optional(receipt.auditId)
  if (receipt.state === 'pending') {
    if (auditId !== undefined) throw new LogsDtoError()
    return {
      ...receipt,
      state: 'pending',
      auditId: undefined,
      artifactDeletion: { ...receipt.artifactDeletion, failureClass: receipt.artifactDeletion.failureClass }
    }
  }
  if (receipt.state === 'completed') {
    if (auditId === undefined) throw new LogsDtoError()
    return {
      ...receipt,
      state: 'completed',
      auditId,
      artifactDeletion: { ...receipt.artifactDeletion, failureClass: receipt.artifactDeletion.failureClass }
    }
  }
  return {
    ...receipt,
    state: 'partial',
    auditId,
    artifactDeletion: { ...receipt.artifactDeletion, failureClass: receipt.artifactDeletion.failureClass }
  }
}

export function parseLogExport(input: unknown): LogExport {
  const parsed = parseOperation(exportSchema, input)
  try {
    return {
      ...parsed,
      items: parsed.items.map((item) => ({
        summary: toLogRequest(item.summary),
        events: item.events.map(toLogLifecycleEvent),
        artifacts: item.artifacts.map(toLogArtifact),
        childIncomplete: item.childIncomplete
      })),
      nextCursor: parsePageCursor(parsed.nextCursor)
    }
  } catch {
    throw new LogsDtoError()
  }
}

export type ParsedReplayEvent = {
  readonly eventId: LogEventId
  readonly requestId: LogRequestId
  readonly occurredAt: string
  readonly channel: LogReplayChannel
  readonly sequence: number
  readonly kind: LogEventKind
  readonly request?: LogRequest
}

export type ParsedReplayGap = {
  readonly channel: LogReplayChannel
  readonly fromSequence: number
  readonly toSequence: number
  readonly recovery: { readonly endpoint: '/api/logs/requests'; readonly cursor: LogPageCursor | undefined }
}

export function parseReplayEvent(input: unknown): ParsedReplayEvent {
  const event = parseReplayEventWire(input)
  return { ...event, request: event.request ? toLogRequest(event.request) : undefined }
}

export function parseReplayGap(input: unknown): ParsedReplayGap {
  const gap = parseReplayGapWire(input)
  if (gap.toSequence < gap.fromSequence) throw new LogsDtoError()
  let cursor: LogPageCursor | undefined
  try {
    cursor = gap.recovery.cursor == null ? undefined : LogPageCursor.parse(gap.recovery.cursor)
  } catch {
    throw new LogsDtoError()
  }
  return { ...gap, recovery: { endpoint: gap.recovery.endpoint, cursor } }
}

export function parseAuditEntry(input: unknown) {
  return parseAuditEntryWire(input)
}

export function parseAuditGap(input: unknown) {
  const gap = parseAuditGapWire(input)
  if (gap.toSequence < gap.fromSequence) throw new LogsDtoError()
  return gap
}

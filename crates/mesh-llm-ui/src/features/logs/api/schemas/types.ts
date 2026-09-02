import type { LogArtifactId, LogAuditId, LogEventId, LogOperationId, LogPageCursor, LogRequestId } from '../ids'

export class LogsDtoError extends Error {
  constructor() {
    super('The logs service returned an invalid response.')
    this.name = 'LogsDtoError'
  }
}

export type LogOutcome = 'active' | 'completed' | 'failed' | 'rejected' | 'cancelled' | 'dropped'
export type LogSource = 'active' | 'durable'
export type LogAuditSource = 'logging_service' | 'logs_api' | 'runtime' | 'mesh' | 'cli'
export type LogAuditSeverity = 'info' | 'warning' | 'error'
export type LogPeerPathType = 'direct' | 'relay'
export type LogCallerPathType = 'local_http' | 'remote_quic_http' | 'relay'
export type LogArtifactUnavailableReason =
  | 'streaming_response_not_assembled'
  | 'response_body_not_bounded'
  | 'capture_content_limit_exceeded'
  | 'capture_memory_budget_exceeded'
  | 'artifact_capture_disabled'
  | 'artifact_capture_failed'
export type LogEventKind =
  | 'admitted'
  | 'route_selected'
  | 'attempt_started'
  | 'attempt_completed'
  | 'attempt_failed'
  | 'backend_stream_first_item'
  | 'stream_started'
  | 'stream_chunk'
  | 'stream_completed'
  | 'usage_recorded'
  | 'stream_error'
  | 'audit_error'
  | 'completed'
  | 'failed'
  | 'rejected'
  | 'cancelled'
  | 'dropped'

export type LogRequest = {
  readonly requestId: LogRequestId
  readonly outcome: LogOutcome
  readonly createdAt: string
  readonly terminalAt: string | undefined
  readonly route: string | undefined
  readonly model: string | undefined
  readonly provider: string | undefined
  readonly engine: string | undefined
  readonly statusCode: number | undefined
  readonly source: LogSource
  readonly callerEndpointId?: string
  readonly callerAddr?: string
  readonly callerPathType?: LogCallerPathType
}

export type LogAuditEntry = {
  readonly entryId: string
  readonly occurredAt: string
  readonly source: LogAuditSource
  readonly code: string
  readonly severity?: LogAuditSeverity
  readonly sequence: number
  readonly contextVersion?: 1
  readonly subjectKind?: 'runtime' | 'model' | 'runtime_instance' | 'cli_command' | 'mesh_peer'
  readonly subjectId?: string
  readonly remoteAddr?: string
  readonly pathType?: LogPeerPathType
  readonly operationId?: string
  readonly requestId?: string
  readonly reasonCode?: string
  readonly outcome?: string
  readonly durationMs?: number
  readonly numericSummaries?: Readonly<Record<string, number>>
  readonly commandSummary?: string
}

export type LogLifecycleEvent = {
  readonly eventId: LogEventId
  readonly requestId: LogRequestId
  readonly occurredAt: string
  readonly kind: LogEventKind
  readonly model: string | undefined
  readonly provider: string | undefined
  readonly engine: string | undefined
  readonly attemptId: string | undefined
  readonly statusCode: number | undefined
  readonly durationMs: number | undefined
  /** Legacy completion-token count. */
  readonly tokens: number | undefined
  readonly promptTokens?: number
  readonly cachedPromptTokens?: number
  readonly completionTokens?: number
  readonly totalTokens?: number
}

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

export type LogArtifact =
  | (LogArtifactBase & { readonly contentState: 'available'; readonly contentBase64: string | undefined })
  | (LogArtifactBase & {
      readonly contentState: 'unavailable'
      readonly unavailableReason?: LogArtifactUnavailableReason
      readonly contentBase64: undefined
    })
  | (LogArtifactBase & { readonly contentState: 'missing'; readonly contentBase64: undefined })
  | (LogArtifactBase & { readonly contentState: 'corrupt'; readonly contentBase64: undefined })

export type LogProxyAttempt = {
  readonly attemptId: string
  readonly requestId: LogRequestId
  readonly occurredAt: string
  readonly target: string
  readonly provider: string | undefined
  readonly engine: string | undefined
  readonly startedAt: string | undefined
  readonly completedAt: string | undefined
  readonly statusCode: number | undefined
}

export type LogsPage<T> = {
  readonly items: readonly T[]
  readonly nextCursor: LogPageCursor | undefined
  /** True only when the UI safety cap stopped server-side pagination. */
  readonly incomplete?: boolean
}

export type LogAuditPage = {
  readonly items: readonly LogAuditEntry[]
  readonly nextCursor: LogPageCursor | undefined
  /** True only when the UI safety cap stopped server-side pagination. */
  readonly incomplete?: boolean
}

export type LogMaintenanceCounts = {
  readonly requests: number
  readonly events: number
  readonly artifacts: number
  readonly proxyRecords: number
  readonly databaseRows: number
}

export type LogArtifactDeletion = {
  readonly removed: number
  readonly failed: number
  readonly failureClass: 'io' | 'unsafe_path' | undefined
}

export type LogCleanupOutcome = Exclude<LogOutcome, 'active'>

export type LogCleanupScope = {
  readonly source: 'durable'
  readonly cutoffBefore: string
  readonly requestLimit: number
  readonly from?: string
  readonly to?: string
  readonly route?: string
  readonly excludeRoute?: string
  readonly model?: string
  readonly provider?: string
  readonly engine?: string
  readonly outcome?: LogCleanupOutcome
}

export type LogCleanupReceipt = {
  readonly operationId: LogOperationId
  readonly auditId: LogAuditId
  readonly cutoffBefore: string
  readonly requestLimit: number
  readonly scope: LogCleanupScope
  readonly state: 'previewed' | 'completed' | 'partial'
  readonly hasMore: boolean
  readonly selectionFingerprint: string
  readonly planned: LogMaintenanceCounts
  readonly executed: LogMaintenanceCounts
  readonly artifactDeletion: LogArtifactDeletion
}

type LogDeleteReceiptBase = {
  readonly operationId: LogOperationId
  readonly requestId: LogRequestId
  readonly selectionFingerprint: string
  readonly planned: LogMaintenanceCounts
  readonly executed: LogMaintenanceCounts
  readonly artifactDeletion: LogArtifactDeletion
}

export type LogDeleteReceipt =
  | (LogDeleteReceiptBase & { readonly state: 'completed'; readonly auditId: LogAuditId })
  | (LogDeleteReceiptBase & { readonly state: 'partial'; readonly auditId: LogAuditId | undefined })
  | (LogDeleteReceiptBase & { readonly state: 'pending'; readonly auditId: undefined })

export type LogExportItem = {
  readonly summary: LogRequest
  readonly events: readonly LogLifecycleEvent[]
  readonly artifacts: readonly LogArtifact[]
  readonly childIncomplete: boolean
}

export type LogExport = {
  readonly items: readonly LogExportItem[]
  readonly nextCursor: LogPageCursor | undefined
  readonly truncated: boolean
  readonly retryRequired: boolean
  readonly artifactContentIncluded: boolean
}

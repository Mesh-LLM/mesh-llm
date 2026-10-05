import type { ThreadMessage } from '@/features/app-tabs/types'
import type { ChatTimings, ChatUsage } from '@/lib/api/types'

export type ChatResponseMetadata = {
  messageId: string
  model?: string
  usage?: ChatUsage
  timings?: ChatTimings
  servedBy?: string
  /** `x-capsule-client-nonce`, echoed by the serving frontend: the key a
   *  consumer of this turn's exchange events can join on. */
  clientNonce?: string
}

export type ThreadMessageMetadata = Pick<
  ThreadMessage,
  'model' | 'route' | 'routeNode' | 'tokens' | 'tokPerSec' | 'ttft' | 'clientNonce'
>

const metadataKeys = ['model', 'route', 'routeNode', 'tokens', 'tokPerSec', 'ttft', 'clientNonce'] satisfies Array<
  keyof ThreadMessageMetadata
>

/** The client nonce the serving frontend echoes on
 *  its responses, streaming or not (`openai-frontend` router.rs
 *  `frontend_lifecycle_middleware`; the host's `/api/responses` proxy passes
 *  headers through). A response the host writes itself carries none, and
 *  absent stays absent. */
export function clientNonceFromHeaders(headers: Headers): Pick<ChatResponseMetadata, 'clientNonce'> {
  const nonce = headers.get('x-capsule-client-nonce')?.trim()
  return nonce ? { clientNonce: nonce } : {}
}

function formatTokenCount(value: number | undefined): string | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined

  return `${Math.max(0, Math.round(value))} tok`
}

function formatMilliseconds(value: number | undefined): string | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined

  return `${Math.max(0, Math.round(value))}ms`
}

// Below ~50ms there is effectively no streaming gap (e.g. MoA dumps the full
// answer at once after the worker wins). Dividing by that produces absurdly
// high tok/s numbers; the wall clock from request start is the honest figure.
const MIN_DECODE_INTERVAL_MS = 50

function formatTokPerSec(
  tokens: number | undefined,
  decodeTimeMs: number | undefined,
  totalTimeMs: number | undefined
): string | undefined {
  if (typeof tokens !== 'number' || !Number.isFinite(tokens)) return undefined

  const decodeOk =
    typeof decodeTimeMs === 'number' && Number.isFinite(decodeTimeMs) && decodeTimeMs >= MIN_DECODE_INTERVAL_MS
  const totalOk = typeof totalTimeMs === 'number' && Number.isFinite(totalTimeMs) && totalTimeMs > 0

  const denomMs = decodeOk ? decodeTimeMs : totalOk ? totalTimeMs : undefined
  if (denomMs === undefined || denomMs <= 0) return undefined

  return `${(tokens / (denomMs / 1000)).toFixed(1)} tok/s`
}

export function responseMetadataToThreadMessage(metadata: ChatResponseMetadata): ThreadMessageMetadata {
  const outputTokens = metadata.usage?.output_tokens
  const servedBy = metadata.servedBy

  return {
    model: metadata.model,
    route: servedBy,
    routeNode: servedBy,
    tokens: formatTokenCount(outputTokens),
    tokPerSec: formatTokPerSec(outputTokens, metadata.timings?.decode_time_ms, metadata.timings?.total_time_ms),
    ttft: formatMilliseconds(metadata.timings?.ttft_ms),
    clientNonce: metadata.clientNonce
  }
}

export function mergeThreadMessageMetadata(
  message: ThreadMessage,
  metadata: ThreadMessageMetadata | undefined,
  fallbackModel?: string
): ThreadMessage {
  const model = metadata?.model ?? message.model ?? fallbackModel
  const merged: ThreadMessage = { ...message, ...(metadata ?? {}) }

  return model ? { ...merged, model } : merged
}

export function extractThreadMessageMetadata(message: ThreadMessage): ThreadMessageMetadata | undefined {
  const metadata: ThreadMessageMetadata = {}

  for (const key of metadataKeys) {
    const value = message[key]
    if (value) metadata[key] = value
  }

  return metadataKeys.some((key) => metadata[key] != null) ? metadata : undefined
}

export function threadMessageMetadataEquals(left: ThreadMessage, right: ThreadMessage): boolean {
  return metadataKeys.every((key) => left[key] === right[key])
}

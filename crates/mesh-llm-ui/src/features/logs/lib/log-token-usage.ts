import type { LogLifecycleEvent } from '@/features/logs/api/schemas'

type TokenUsageEvent = Pick<
  LogLifecycleEvent,
  'tokens' | 'promptTokens' | 'cachedPromptTokens' | 'completionTokens' | 'totalTokens'
>

export type TokenUsageEntry = {
  readonly label: 'Prompt tokens' | 'Cached prompt tokens' | 'Completion tokens' | 'Total tokens'
  readonly value: number
}

/** Uses legacy `tokens` only as the completion-token count. */
export function tokenUsageEntries(event: TokenUsageEvent): readonly TokenUsageEntry[] {
  const completionTokens = event.completionTokens ?? event.tokens
  return [
    ...(event.promptTokens === undefined ? [] : [{ label: 'Prompt tokens' as const, value: event.promptTokens }]),
    ...(event.cachedPromptTokens === undefined
      ? []
      : [{ label: 'Cached prompt tokens' as const, value: event.cachedPromptTokens }]),
    ...(completionTokens === undefined ? [] : [{ label: 'Completion tokens' as const, value: completionTokens }]),
    ...(event.totalTokens === undefined ? [] : [{ label: 'Total tokens' as const, value: event.totalTokens }])
  ]
}

export function completionTokenCount(event: TokenUsageEvent): number | undefined {
  return event.completionTokens ?? event.tokens
}

export function formatTokenUsage(event: TokenUsageEvent): string | undefined {
  const entries = tokenUsageEntries(event)
  return entries.length === 0
    ? undefined
    : entries.map(({ label, value }) => `${label}: ${value.toLocaleString()}`).join(' · ')
}

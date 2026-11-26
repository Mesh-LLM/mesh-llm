import { createChatDraftConversationId } from '@/features/chat/api/chat-session-ids'
import type { ComposerQuote } from '@/features/chat/lib/quote-to-ask'

export type ComposerSubmission = {
  prompt: string
  attachments: File[]
  /** What the composer held before the quote merged into `prompt`; failure
   *  paths restore the draft from this so a failed send returns the typed
   *  prompt and the quote chip instead of the merged text. */
  restore?: { prompt: string; quote: ComposerQuote | null }
}
/** The draft carries a pending quote-to-ask selection; it merges into the
 *  prompt as a quoted prefix at submission time. */
export type ConversationComposerDraft = ComposerSubmission & { quote?: ComposerQuote | null }
/** `target` is the node the prompt was submitted to ('' routes as usual), kept so a change
 *  while it waits doesn't reroute it. */
export type QueuedSubmission = ComposerSubmission & {
  id: string
  timestamp: string
  conversationId: string
  target: string
}
export type FailedSubmission = ComposerSubmission & {
  id: string
  timestamp: string
  conversationId: string
  errorMessage: string
  model: string
  includeUserRow: boolean
}
export type DeleteConversationOptions = { returnFocusElement?: HTMLElement | null }

export function hasLastUserTurn(messages: Array<{ role: string }>): boolean {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') return true
  }

  return false
}

export function getMessageTextContent(message: { parts?: Array<{ type: string; content?: unknown }> }): string {
  const textPart = message.parts?.find((part) => part.type === 'text' && typeof part.content === 'string')
  return typeof textPart?.content === 'string' ? textPart.content.trim() : ''
}

export function createQueuedSubmissionId(): string {
  return `queued-${createChatDraftConversationId()}`
}

export function getQueuedSubmissionBody(submission: QueuedSubmission): string {
  const trimmedPrompt = submission.prompt.trim()
  if (trimmedPrompt) return trimmedPrompt

  return `${submission.attachments.length} attachment${submission.attachments.length === 1 ? '' : 's'} queued`
}

export function getSubmissionBody(submission: ComposerSubmission): string {
  const trimmedPrompt = submission.prompt.trim()
  if (trimmedPrompt) return trimmedPrompt

  return `${submission.attachments.length} attachment${submission.attachments.length === 1 ? '' : 's'}`
}

export function createStoppedAssistantThreadMessage(model: string) {
  return {
    id: `stopped-${createChatDraftConversationId()}`,
    messageRole: 'assistant' as const,
    timestamp: new Date().toISOString(),
    body: '',
    model
  }
}

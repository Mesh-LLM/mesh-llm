import { createChatDraftConversationId } from '@/features/chat/api/chat-session-ids'

export type ComposerSubmission = { prompt: string; attachments: File[] }
export type ConversationComposerDraft = ComposerSubmission
export type QueuedSubmission = ComposerSubmission & { id: string; timestamp: string; conversationId: string }
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

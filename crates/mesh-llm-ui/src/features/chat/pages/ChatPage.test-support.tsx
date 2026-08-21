/* eslint-disable react-refresh/only-export-components */
import { cleanup, configure, render, screen, waitFor } from '@testing-library/react'
import type { MultimodalContent } from '@tanstack/ai-client'
import { afterEach, beforeEach, expect, vi } from 'vitest'
import { CHAT_HARNESS } from '@/features/app-tabs/data'
import { APP_STORAGE_KEYS } from '@/features/app-tabs/data'
import { ChatLayout } from '@/features/chat/layouts/ChatLayout'
import { DataModeProvider } from '@/lib/data-mode/DataModeContext'
import { FeatureFlagProvider } from '@/lib/feature-flags'

export { CHAT_HARNESS, APP_STORAGE_KEYS } from '@/features/app-tabs/data'
export { DEFAULT_SYSTEM_PROMPT } from '@/constants/system-prompt'
export { DataModeProvider } from '@/lib/data-mode/DataModeContext'
export { FeatureFlagProvider } from '@/lib/feature-flags'

export const scrollIntoViewMock = vi.fn()
export const createObjectUrlMock = vi.fn((file: File) => `blob:preview/${file.name}`)
export const revokeObjectUrlMock = vi.fn()

configure({ asyncUtilTimeout: 4_000 })

export function installPointerCaptureShim() {
  Object.defineProperty(HTMLElement.prototype, 'hasPointerCapture', {
    configurable: true,
    value: () => false
  })
  Object.defineProperty(HTMLElement.prototype, 'setPointerCapture', {
    configurable: true,
    value: () => undefined
  })
  Object.defineProperty(HTMLElement.prototype, 'releasePointerCapture', {
    configurable: true,
    value: () => undefined
  })
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
    configurable: true,
    value: scrollIntoViewMock
  })
}

export function installImageFallbackShim() {
  class TestImage {
    width = 0
    height = 0
    onload: (() => void) | null = null
    onerror: (() => void) | null = null

    set src(value: string) {
      void value
      window.setTimeout(() => this.onerror?.(), 0)
    }
  }

  Object.defineProperty(globalThis, 'Image', {
    configurable: true,
    value: TestImage
  })
}

export function installObjectUrlShim() {
  Object.defineProperty(URL, 'createObjectURL', {
    configurable: true,
    value: createObjectUrlMock
  })
  Object.defineProperty(URL, 'revokeObjectURL', {
    configurable: true,
    value: revokeObjectUrlMock
  })
}

const chatMock = vi.hoisted(() => {
  function createUiMessage(id: string, role: 'user' | 'assistant', body: string) {
    return {
      id,
      role,
      createdAt: new Date('2026-05-06T00:00:00.000Z'),
      parts: [{ type: 'text' as const, content: body }]
    }
  }

  const state = {
    messagesByConversation: new Map<string, ReturnType<typeof createUiMessage>[]>(),
    statusByConversation: new Map<string, 'ready' | 'submitted' | 'streaming' | 'error'>(),
    errorByConversation: new Map<string, Error | undefined>(),
    sendAssistantText: 'Partial assistant reply',
    sendResponseMetadata: undefined as
      | undefined
      | {
          model?: string
          usage?: { input_tokens: number; output_tokens: number; total_tokens?: number }
          timings?: { decode_time_ms?: number; ttft_ms?: number; total_time_ms?: number }
          servedBy?: string
        },
    sendStatus: 'streaming' as 'ready' | 'submitted' | 'streaming',
    sendErrorMessage: undefined as string | undefined,
    sendErrorResolves: false,
    sendOptimisticStatusBeforeError: false,
    sendOptimisticUserMessageBeforeError: false,
    sendOptimisticAssistantPlaceholderBeforeError: false,
    reloadAssistantText: 'Retried assistant reply',
    reloadStatus: 'ready' as 'ready' | 'submitted' | 'streaming' | 'error',
    reloadErrorMessage: undefined as string | undefined,
    stopCalls: [] as string[],
    sendCalls: [] as Array<{
      conversationId: string
      content: string | MultimodalContent
      model: string
      systemPrompt: string
    }>,
    reloadCalls: [] as string[],
    hookConversationIds: [] as string[],
    hookSystemPrompts: [] as Array<{ conversationId: string; systemPrompt: string }>,
    hookUnmounts: [] as string[],
    reset() {
      state.messagesByConversation.clear()
      state.statusByConversation.clear()
      state.errorByConversation.clear()
      state.sendAssistantText = 'Partial assistant reply'
      state.sendResponseMetadata = undefined
      state.sendStatus = 'streaming'
      state.sendErrorMessage = undefined
      state.sendErrorResolves = false
      state.sendOptimisticStatusBeforeError = false
      state.sendOptimisticUserMessageBeforeError = false
      state.sendOptimisticAssistantPlaceholderBeforeError = false
      state.reloadAssistantText = 'Retried assistant reply'
      state.reloadStatus = 'ready'
      state.reloadErrorMessage = undefined
      state.stopCalls = []
      state.sendCalls = []
      state.reloadCalls = []
      state.hookConversationIds = []
      state.hookSystemPrompts = []
      state.hookUnmounts = []
    },
    createUiMessage
  }

  return state
})

const attachmentPreprocessingMock = vi.hoisted(() => ({
  describeImageForPrompt: vi.fn(),
  extractPdfTextFromFile: vi.fn(),
  describeScannedPdf: vi.fn(),
  isBrowserVisionModelLoaded: vi.fn()
}))

export { attachmentPreprocessingMock, chatMock }

export type TestAttachmentProcessingStage = 'downloading' | 'starting' | 'processing'

export function createDeferred<T>() {
  let resolveDeferred: ((value: T) => void) | undefined
  let rejectDeferred: ((reason?: unknown) => void) | undefined
  const promise = new Promise<T>((resolve, reject) => {
    resolveDeferred = resolve
    rejectDeferred = reject
  })

  return {
    promise,
    resolve(value: T) {
      resolveDeferred?.(value)
    },
    reject(reason?: unknown) {
      rejectDeferred?.(reason)
    }
  }
}

vi.mock('@/features/chat/api/chat-storage', () => ({
  MAX_CHAT_CONVERSATIONS: 80,
  clearChatState: vi.fn(),
  loadChatState: vi.fn(),
  saveChatState: vi.fn(),
  trimThreadMessages: vi.fn((messages: Array<unknown>) => messages)
}))

vi.mock('@/features/network/api/use-models-query', () => ({
  useModelsQuery: vi.fn()
}))

vi.mock('@/features/network/api/use-status-query', () => ({
  useStatusQuery: vi.fn()
}))

vi.mock('@/features/network/api/models-adapter', () => ({
  adaptModelsToSummary: vi.fn(() => CHAT_HARNESS.models)
}))

vi.mock('@/features/chat/api/attachment-preprocessing', () => ({
  describeImageForPrompt: attachmentPreprocessingMock.describeImageForPrompt,
  extractPdfTextFromFile: attachmentPreprocessingMock.extractPdfTextFromFile,
  describeScannedPdf: attachmentPreprocessingMock.describeScannedPdf,
  isBrowserVisionModelLoaded: attachmentPreprocessingMock.isBrowserVisionModelLoaded
}))

vi.mock('@/features/chat/api/use-chat', async () => {
  const React = await import('react')

  function threadToUiMessage(message: {
    id: string
    messageRole: 'user' | 'assistant'
    timestamp: string
    body: string
  }) {
    return {
      id: message.id,
      role: message.messageRole,
      createdAt: new Date(message.timestamp),
      parts: [{ type: 'text' as const, content: message.body }]
    }
  }

  return {
    useMeshChat: vi.fn(
      ({
        conversationId,
        model,
        systemPrompt,
        initialMessages,
        onResponseMetadata
      }: {
        conversationId: string
        model: string
        systemPrompt: string
        initialMessages: Array<{ id: string; messageRole: 'user' | 'assistant'; timestamp: string; body: string }>
        onResponseMetadata?: (metadata: {
          messageId: string
          model?: string
          usage?: { input_tokens: number; output_tokens: number; total_tokens?: number }
          timings?: { decode_time_ms?: number; ttft_ms?: number; total_time_ms?: number }
          servedBy?: string
        }) => void
      }) => {
        React.useEffect(() => {
          chatMock.hookConversationIds.push(conversationId)
          return () => {
            chatMock.hookUnmounts.push(conversationId)
          }
        }, [conversationId])

        React.useEffect(() => {
          chatMock.hookSystemPrompts.push({ conversationId, systemPrompt })
        }, [conversationId, systemPrompt])

        const initialUiMessages = React.useMemo(() => initialMessages.map(threadToUiMessage), [initialMessages])
        const [messages, setMessages] = React.useState(
          () => chatMock.messagesByConversation.get(conversationId) ?? initialUiMessages
        )
        const [status, setStatus] = React.useState<'ready' | 'submitted' | 'streaming' | 'error'>(
          () => chatMock.statusByConversation.get(conversationId) ?? 'ready'
        )
        const [error, setError] = React.useState<Error | undefined>(() =>
          chatMock.errorByConversation.get(conversationId)
        )
        const messagesRef = React.useRef(messages)

        React.useEffect(() => {
          const nextMessages = chatMock.messagesByConversation.get(conversationId) ?? initialUiMessages
          setMessages(nextMessages)
          setStatus(chatMock.statusByConversation.get(conversationId) ?? 'ready')
          setError(chatMock.errorByConversation.get(conversationId))
        }, [conversationId, initialUiMessages])

        React.useEffect(() => {
          messagesRef.current = messages
          chatMock.messagesByConversation.set(conversationId, messages)
        }, [conversationId, messages])

        React.useEffect(() => {
          chatMock.statusByConversation.set(conversationId, status)
        }, [conversationId, status])

        React.useEffect(() => {
          chatMock.errorByConversation.set(conversationId, error)
        }, [conversationId, error])

        return {
          messages,
          sendMessage: vi.fn(async (content: string | MultimodalContent) => {
            chatMock.sendCalls.push({ conversationId, content, model, systemPrompt })
            const body =
              typeof content === 'string'
                ? content
                : (() => {
                    if (typeof content.content === 'string') return content.content
                    const textPart = content.content.find(
                      (part): part is { type: 'text'; content: string } => part.type === 'text'
                    )
                    return textPart?.content ?? ''
                  })()
            if (chatMock.sendErrorMessage) {
              if (chatMock.sendOptimisticUserMessageBeforeError) {
                const optimisticMessages = [
                  ...messagesRef.current,
                  chatMock.createUiMessage(`user-${chatMock.sendCalls.length}`, 'user', body)
                ]
                if (chatMock.sendOptimisticAssistantPlaceholderBeforeError) {
                  optimisticMessages.push(
                    chatMock.createUiMessage(`assistant-${chatMock.sendCalls.length}`, 'assistant', '')
                  )
                }
                chatMock.messagesByConversation.set(conversationId, optimisticMessages)
                setMessages(optimisticMessages)
              }
              if (chatMock.sendOptimisticStatusBeforeError) {
                chatMock.statusByConversation.set(conversationId, chatMock.sendStatus)
                setStatus(chatMock.sendStatus)
                await Promise.resolve()
              }
              const sendError = new Error(chatMock.sendErrorMessage)
              chatMock.errorByConversation.set(conversationId, sendError)
              chatMock.statusByConversation.set(conversationId, 'error')
              setError(sendError)
              setStatus('error')
              if (chatMock.sendErrorResolves) {
                return
              }
              throw sendError
            }
            const userMessageId = `user-${chatMock.sendCalls.length}`
            const assistantMessageId = `assistant-${chatMock.sendCalls.length}`
            const nextMessages = [
              ...messagesRef.current,
              chatMock.createUiMessage(userMessageId, 'user', body),
              chatMock.createUiMessage(assistantMessageId, 'assistant', chatMock.sendAssistantText)
            ]
            chatMock.messagesByConversation.set(conversationId, nextMessages)
            chatMock.statusByConversation.set(conversationId, chatMock.sendStatus)
            chatMock.errorByConversation.set(conversationId, undefined)
            setMessages(nextMessages)
            if (chatMock.sendResponseMetadata) {
              onResponseMetadata?.({ messageId: assistantMessageId, ...chatMock.sendResponseMetadata })
            }
            setStatus(chatMock.sendStatus)
            setError(undefined)
          }),
          reload: vi.fn(async () => {
            chatMock.reloadCalls.push(conversationId)
            const currentMessages = messagesRef.current
            let lastUserIndex = -1
            for (let index = currentMessages.length - 1; index >= 0; index -= 1) {
              if (currentMessages[index]?.role === 'user') {
                lastUserIndex = index
                break
              }
            }
            if (lastUserIndex < 0) return

            const nextMessages = [
              ...currentMessages.slice(0, lastUserIndex + 1),
              chatMock.createUiMessage(
                `assistant-retry-${chatMock.reloadCalls.length}`,
                'assistant',
                chatMock.reloadAssistantText
              )
            ]

            chatMock.messagesByConversation.set(conversationId, nextMessages)
            chatMock.statusByConversation.set(conversationId, chatMock.reloadStatus)
            chatMock.errorByConversation.set(
              conversationId,
              chatMock.reloadErrorMessage ? new Error(chatMock.reloadErrorMessage) : undefined
            )
            setMessages(nextMessages)
            setStatus(chatMock.reloadStatus)
            setError(chatMock.reloadErrorMessage ? new Error(chatMock.reloadErrorMessage) : undefined)
          }),
          stop: vi.fn(() => {
            chatMock.stopCalls.push(conversationId)
            chatMock.statusByConversation.set(conversationId, 'ready')
            setStatus('ready')
          }),
          status,
          error,
          isLoading: status === 'submitted' || status === 'streaming',
          setMessages,
          append: vi.fn(),
          addToolResult: vi.fn(),
          addToolApprovalResponse: vi.fn(),
          isSubscribed: false,
          connectionStatus: 'disconnected',
          sessionGenerating: false,
          clear: vi.fn()
        }
      }
    )
  }
})

const { loadChatState, saveChatState, trimThreadMessages } = await import('@/features/chat/api/chat-storage')
const { adaptModelsToSummary } = await import('@/features/network/api/models-adapter')
const { useModelsQuery } = await import('@/features/network/api/use-models-query')
const { useStatusQuery } = await import('@/features/network/api/use-status-query')
const { ChatPage, ChatPageContent } = await import('@/features/chat/pages/ChatPage')
const { ChatSessionProvider } = await import('@/features/chat/api/chat-session')

export {
  adaptModelsToSummary,
  ChatPage,
  ChatPageContent,
  ChatSessionProvider,
  loadChatState,
  saveChatState,
  trimThreadMessages,
  useModelsQuery,
  useStatusQuery
}

export function renderChatPage({
  transparencyTabEnabled = false,
  systemPromptButtonEnabled = false,
  mode = 'harness'
}: {
  transparencyTabEnabled?: boolean
  systemPromptButtonEnabled?: boolean
  mode?: 'live' | 'harness'
} = {}) {
  if (transparencyTabEnabled || systemPromptButtonEnabled) {
    window.localStorage.setItem(
      APP_STORAGE_KEYS.featureFlagOverrides,
      JSON.stringify({
        chat: { transparencyTab: transparencyTabEnabled, systemPromptButton: systemPromptButtonEnabled }
      })
    )
  }

  render(
    <FeatureFlagProvider>
      <DataModeProvider initialMode={mode} persist={false}>
        <ChatPage />
      </DataModeProvider>
    </FeatureFlagProvider>
  )
}

export function renderPersistentChatRoute(showChat: boolean) {
  return render(
    <FeatureFlagProvider>
      <DataModeProvider initialMode="live" persist={false}>
        <ChatSessionProvider>
          {showChat ? <ChatPageContent /> : <div data-testid="network-route">Network route</div>}
        </ChatSessionProvider>
      </DataModeProvider>
    </FeatureFlagProvider>
  )
}

export function queryAllByTextContent(text: string) {
  return screen.queryAllByText((_, element) => element?.textContent?.includes(text) ?? false, {
    selector: 'span,button'
  })
}

export function shortTimestamp(date: Date) {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  }).format(date)
}

export function setLocalTime(date: Date, hours: number, minutes: number) {
  const timestamp = new Date(date)
  timestamp.setHours(hours, minutes, 0, 0)

  return timestamp
}

export async function expectPartialAssistantReply() {
  await waitFor(() => expect(screen.getByText('Partial assistant reply')).toBeInTheDocument())
}

export function chatLayout(status: string, stickToBottomKey = 'conversation-a:0') {
  return (
    <ChatLayout
      actions={<span>{status}</span>}
      composer={<textarea aria-label="Prompt" />}
      sidebar={<div role="tablist" aria-label="Chat sidebar views" />}
      stickToBottomKey={stickToBottomKey}
      title="Chat"
    >
      <div data-testid="message-content">Messages</div>
    </ChatLayout>
  )
}

export function setMessageListDimensions(messageList: HTMLElement) {
  Object.defineProperty(messageList, 'scrollHeight', { configurable: true, value: 1400 })
  Object.defineProperty(messageList, 'clientHeight', { configurable: true, value: 420 })
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

beforeEach(() => {
  scrollIntoViewMock.mockClear()
  createObjectUrlMock.mockClear()
  revokeObjectUrlMock.mockClear()
  installPointerCaptureShim()
  installImageFallbackShim()
  installObjectUrlShim()
  window.localStorage.removeItem(APP_STORAGE_KEYS.featureFlagOverrides)
  window.localStorage.removeItem(APP_STORAGE_KEYS.chatSystemPrompt)
  vi.mocked(loadChatState).mockResolvedValue(undefined)
  vi.mocked(saveChatState).mockResolvedValue(undefined)
  vi.mocked(trimThreadMessages).mockImplementation((messages) => messages)
  vi.mocked(adaptModelsToSummary).mockReturnValue(CHAT_HARNESS.models)
  vi.mocked(useModelsQuery).mockReturnValue({
    data: { mesh_models: [] },
    isFetching: false,
    isError: false,
    refetch: vi.fn()
  } as unknown as ReturnType<typeof useModelsQuery>)
  vi.mocked(useStatusQuery).mockReturnValue({
    data: undefined,
    isFetching: false,
    isError: false,
    refetch: vi.fn()
  } as unknown as ReturnType<typeof useStatusQuery>)
  attachmentPreprocessingMock.describeImageForPrompt.mockReset()
  attachmentPreprocessingMock.extractPdfTextFromFile.mockReset()
  attachmentPreprocessingMock.describeScannedPdf.mockReset()
  attachmentPreprocessingMock.isBrowserVisionModelLoaded.mockReset()
  attachmentPreprocessingMock.isBrowserVisionModelLoaded.mockReturnValue(false)
  attachmentPreprocessingMock.describeImageForPrompt.mockResolvedValue({
    imageDescription: '[Image description: A tabby cat]'
  })
  attachmentPreprocessingMock.extractPdfTextFromFile.mockResolvedValue({
    text: '',
    pageCount: 1,
    pagesWithText: 0,
    wordCount: 0
  })
  attachmentPreprocessingMock.describeScannedPdf.mockResolvedValue('[Page 1]\n[Image description: A scanned receipt]')
  chatMock.reset()
})
/* eslint-enable react-refresh/only-export-components */

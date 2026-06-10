import { useLayoutEffect } from 'react'
import { render, waitFor } from '@testing-library/react'
import type { UIMessage } from '@tanstack/ai'
import type { ConnectConnectionAdapter } from '@tanstack/ai-client'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { DEFAULT_SYSTEM_PROMPT } from '@/constants/system-prompt'
import { useMeshChat } from '@/features/chat/api/use-chat'
import type { ThreadMessage } from '@/features/app-tabs/types'

type UseChatOptions = {
  id: string
  connection: ConnectConnectionAdapter
  initialMessages: UIMessage[]
}

function createUserMessage(content: string): UIMessage {
  return {
    id: 'user-1',
    role: 'user',
    parts: [{ type: 'text', content }],
    createdAt: new Date('2026-05-13T00:00:00.000Z')
  }
}

async function drainMessageStream(adapter: ConnectConnectionAdapter, messages: UIMessage[]) {
  for await (const chunk of adapter.connect(messages, undefined, undefined)) {
    void chunk
    // Drain the stream so the request body is built and posted.
  }
}

const useChatMockState = vi.hoisted(() => ({
  resetMessages: [] as unknown[][],
  reset() {
    this.resetMessages = []
  }
}))

vi.mock('@tanstack/ai-react', async () => {
  const React = await import('react')

  return {
    useChat: vi.fn(({ connection, initialMessages }: UseChatOptions) => {
      const connectionRef = React.useRef(connection)
      const messagesRef = React.useRef(initialMessages)
      const setMessages = vi.fn((messages: UIMessage[]) => {
        messagesRef.current = messages
        useChatMockState.resetMessages.push(messages)
      })

      return {
        messages: messagesRef.current,
        sendMessage: vi.fn((content: string) =>
          drainMessageStream(connectionRef.current, [...messagesRef.current, createUserMessage(content)])
        ),
        setMessages,
        reload: vi.fn(),
        stop: vi.fn(),
        status: 'ready',
        error: null,
        isLoading: false
      }
    })
  }
})

function createSSEStream(lines: string[]) {
  const encoder = new TextEncoder()

  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const line of lines) {
        controller.enqueue(encoder.encode(line))
      }

      controller.close()
    }
  })
}

function SendFirstMessageOnLayout() {
  const chat = useMeshChat({
    conversationId: 'chat-1',
    model: 'auto',
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    initialMessages: []
  })

  useLayoutEffect(() => {
    void chat.sendMessage('What is MeshLLM?')
  }, [chat])

  return null
}

function SendMessageOnLayout({
  message,
  conversationId = 'chat-1',
  initialMessages = [],
  model,
  systemPrompt
}: {
  message?: string
  conversationId?: string
  initialMessages?: ThreadMessage[]
  model: string
  systemPrompt: string
}) {
  const chat = useMeshChat({
    conversationId,
    model,
    systemPrompt,
    initialMessages
  })

  useLayoutEffect(() => {
    if (message) {
      void chat.sendMessage(message)
    }
  }, [chat, message])

  return null
}

describe('useMeshChat', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    useChatMockState.reset()
  })

  it('sends the default system prompt with the first message in a new chat', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(createSSEStream(['data: [DONE]\n']), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    render(<SendFirstMessageOnLayout />)

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    const request = fetchMock.mock.calls[0]?.[1]
    const body = JSON.parse(String(request?.body)) as { input: Array<{ role: string; content: string }> }

    expect(body.input[0]).toEqual({ role: 'system', content: DEFAULT_SYSTEM_PROMPT })
    expect(body.input[1]).toEqual({ role: 'user', content: 'What is MeshLLM?' })
  })

  it('sends the latest model and system prompt after rerendering the chat hook', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(createSSEStream(['data: [DONE]\n']), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const { rerender } = render(<SendMessageOnLayout model="model-a" systemPrompt="prompt-a" />)

    rerender(<SendMessageOnLayout message="Use latest values" model="model-b" systemPrompt="prompt-b" />)

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    const request = fetchMock.mock.calls[0]?.[1]
    const body = JSON.parse(String(request?.body)) as {
      model: string
      input: Array<{ role: string; content: string }>
    }

    expect(body.model).toBe('model-b')
    expect(body.input[0]).toEqual({ role: 'system', content: 'prompt-b' })
    expect(body.input[1]).toEqual({ role: 'user', content: 'Use latest values' })
  })

  it('resets the underlying chat messages before a fresh conversation can send', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(createSSEStream(['data: [DONE]\n']), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const oldThread: ThreadMessage[] = [
      {
        id: 'old-user',
        messageRole: 'user',
        timestamp: '2026-05-13T00:00:00.000Z',
        body: 'Old Windows via Tailscale question'
      },
      {
        id: 'old-assistant',
        messageRole: 'assistant',
        timestamp: '2026-05-13T00:00:01.000Z',
        body: 'Old answer that must not leak'
      }
    ]

    const { rerender } = render(
      <SendMessageOnLayout conversationId="old-chat" initialMessages={oldThread} model="mesh" systemPrompt="" />
    )

    rerender(
      <SendMessageOnLayout
        conversationId="fresh-chat"
        initialMessages={[]}
        message="Only answer this fresh prompt"
        model="mesh"
        systemPrompt=""
      />
    )

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    const request = fetchMock.mock.calls[0]?.[1]
    const body = JSON.parse(String(request?.body)) as { input: Array<{ role: string; content: string }> }

    expect(body.input).toEqual([{ role: 'user', content: 'Only answer this fresh prompt' }])
    expect(JSON.stringify(body.input)).not.toContain('Windows via Tailscale')
  })
})

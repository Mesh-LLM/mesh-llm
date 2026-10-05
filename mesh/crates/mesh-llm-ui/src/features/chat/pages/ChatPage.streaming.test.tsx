import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import {
  ChatPageContent,
  ChatSessionProvider,
  DataModeProvider,
  FeatureFlagProvider,
  chatMock,
  expectPartialAssistantReply,
  renderChatPage,
  renderPersistentChatRoute,
  saveChatState,
  scrollIntoViewMock,
  setMessageListDimensions
} from './ChatPage.test-support'

describe('ChatPage', () => {
  it('renders streamed thinking separately, formats final markdown, and persists the raw assistant body', async () => {
    const user = userEvent.setup()
    const streamedBody = 'Reasoning text.</think> The capital of France is **Paris**.'
    chatMock.sendAssistantText = streamedBody

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Show final answer formatting')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    const reasoningDisclosure = await screen.findByRole('button', { name: 'Peer consultation Show details' })
    expect(screen.getByText('Reasoning text.')).not.toBeVisible()

    await user.click(reasoningDisclosure)

    expect(screen.getByText('Reasoning text.')).toBeVisible()

    const paris = screen.getByText('Paris')
    expect(paris.tagName.toLowerCase()).toBe('strong')
    expect(paris.closest('.select-text')).toHaveTextContent('The capital of France is Paris.')
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toHaveTextContent('Streaming response...')

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Show final answer formatting',
        streamedBody
      ])
    })
  })

  it('keeps live streaming mounted and marked in the sidebar when the chat route unmounts', async () => {
    const user = userEvent.setup()
    const { rerender } = renderPersistentChatRoute(true)

    await user.type(screen.getByLabelText('Prompt'), 'Continue this while I leave')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Continue this while I leave',
        'Partial assistant reply'
      ])
    })

    const streamingConversationId = vi.mocked(saveChatState).mock.calls.at(-1)?.[1].activeConversationId
    if (!streamingConversationId) throw new Error('Expected streaming conversation id')
    const hookUnmountsBeforeRouteChange = chatMock.hookUnmounts.length

    rerender(
      <FeatureFlagProvider>
        <DataModeProvider initialMode="live" persist={false}>
          <ChatSessionProvider>
            <div data-testid="network-route">Network route</div>
          </ChatSessionProvider>
        </DataModeProvider>
      </FeatureFlagProvider>
    )

    expect(screen.getByTestId('network-route')).toBeInTheDocument()
    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts).toHaveLength(hookUnmountsBeforeRouteChange)

    rerender(
      <FeatureFlagProvider>
        <DataModeProvider initialMode="live" persist={false}>
          <ChatSessionProvider>
            <ChatPageContent />
          </ChatSessionProvider>
        </DataModeProvider>
      </FeatureFlagProvider>
    )

    await expectPartialAssistantReply()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()
    expect(chatMock.hookConversationIds).toContain(streamingConversationId)
  })

  it('keeps a streaming conversation active when selecting another chat in the sidebar', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Write a long story')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Write a long story',
        'Partial assistant reply'
      ])
    })

    const streamingConversationId = vi.mocked(saveChatState).mock.calls.at(-1)?.[1].activeConversationId
    if (!streamingConversationId) throw new Error('Expected streaming conversation id')
    const hookUnmountsBeforeSwitch = chatMock.hookUnmounts.length

    await user.click(screen.getByRole('button', { name: 'New' }))

    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeSwitch)).not.toContain(streamingConversationId)
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()
    expect(screen.getByText('Start Chatting')).toBeInTheDocument()
    expect(screen.queryByText('Partial assistant reply')).not.toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: /Write a long story/i })[0])

    await expectPartialAssistantReply()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toHaveTextContent('Streaming response...')
    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeSwitch)).not.toContain(streamingConversationId)
    expect(chatMock.hookConversationIds).toContain(streamingConversationId)
  })

  it('sends from the newly selected live conversation while another conversation is streaming', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First streaming prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId
      expect(activeConversationId).toBeTruthy()
      expect(chatMock.sendCalls).toHaveLength(1)
      expect(chatMock.sendCalls[0]?.conversationId).toBe(activeConversationId)
      expect(latestState?.threads[activeConversationId ?? ''].map((message) => message.body)).toEqual([
        'First streaming prompt',
        'Partial assistant reply'
      ])
    })

    const streamingConversationId = vi.mocked(saveChatState).mock.calls.at(-1)?.[1].activeConversationId
    if (!streamingConversationId) throw new Error('Expected streaming conversation id')
    const hookUnmountsBeforeNewChat = chatMock.hookUnmounts.length

    await user.click(screen.getByRole('button', { name: 'New' }))

    let newConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId ?? ''
      expect(activeConversationId).toBeTruthy()
      expect(activeConversationId).not.toBe(streamingConversationId)
      expect(latestState?.threads[activeConversationId]).toEqual([])
      newConversationId = activeConversationId
    })

    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeNewChat)).not.toContain(streamingConversationId)
    expect(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Second prompt for the new chat')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.conversationId).toBe(newConversationId)
      expect(chatMock.sendCalls[1]?.conversationId).not.toBe(streamingConversationId)
      expect(chatMock.sendCalls[1]?.content).toBe('Second prompt for the new chat')
    })
  })

  it('does not retarget either hidden lane when creating a third chat while two chats stream', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First stream stays alive')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()

    let firstStreamingConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      firstStreamingConversationId = latestState?.activeConversationId ?? ''
      expect(firstStreamingConversationId).toBeTruthy()
      expect(chatMock.sendCalls).toHaveLength(1)
      expect(chatMock.sendCalls[0]?.conversationId).toBe(firstStreamingConversationId)
    })

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Prompt'), 'Second stream also stays alive')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()

    let secondStreamingConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      secondStreamingConversationId = latestState?.activeConversationId ?? ''
      expect(secondStreamingConversationId).toBeTruthy()
      expect(secondStreamingConversationId).not.toBe(firstStreamingConversationId)
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.conversationId).toBe(secondStreamingConversationId)
    })

    const hookUnmountsBeforeThirdChat = chatMock.hookUnmounts.length

    await user.click(screen.getByRole('button', { name: 'New' }))

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(latestState?.activeConversationId).not.toBe(firstStreamingConversationId)
      expect(latestState?.activeConversationId).not.toBe(secondStreamingConversationId)
    })

    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeThirdChat)).not.toContain(firstStreamingConversationId)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeThirdChat)).not.toContain(secondStreamingConversationId)
    expect(screen.getByText('Start Chatting')).toBeInTheDocument()
  })

  it('keeps the third chat composer draft isolated while both live lanes stream', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First stream stays alive')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()

    let firstStreamingConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      firstStreamingConversationId = latestState?.activeConversationId ?? ''
      expect(firstStreamingConversationId).toBeTruthy()
      expect(chatMock.sendCalls).toHaveLength(1)
    })

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Prompt'), 'Second stream also stays alive')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()

    let secondStreamingConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      secondStreamingConversationId = latestState?.activeConversationId ?? ''
      expect(secondStreamingConversationId).toBeTruthy()
      expect(secondStreamingConversationId).not.toBe(firstStreamingConversationId)
      expect(chatMock.sendCalls).toHaveLength(2)
    })

    await user.click(screen.getByRole('button', { name: 'New' }))

    let thirdConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      thirdConversationId = latestState?.activeConversationId ?? ''
      expect(thirdConversationId).toBeTruthy()
      expect(thirdConversationId).not.toBe(firstStreamingConversationId)
      expect(thirdConversationId).not.toBe(secondStreamingConversationId)
    })

    const thirdDraft = 'Draft belongs only to the third chat'
    await user.type(screen.getByLabelText('Prompt'), thirdDraft)

    expect(screen.getByLabelText('Prompt')).toHaveValue(thirdDraft)
    expect(screen.getByRole('button', { name: 'Queue' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Stop' })).not.toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: /First stream stays alive/i })[0])

    expect(screen.getByLabelText('Prompt')).not.toHaveValue(thirdDraft)
    expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /^New chat/i }))

    expect(screen.getByLabelText('Prompt')).toHaveValue(thirdDraft)
    expect(screen.getByRole('button', { name: 'Queue' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Queue' }))

    expect(chatMock.sendCalls).toHaveLength(2)
    expect(screen.getByText(thirdDraft)).toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: /First stream stays alive/i })[0])

    expect(screen.queryByText(thirdDraft)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    expect(chatMock.sendCalls).toHaveLength(2)

    await user.click(screen.getByRole('button', { name: /^New chat/i }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(3)
      expect(chatMock.sendCalls[2]?.conversationId).toBe(thirdConversationId)
      expect(chatMock.sendCalls[2]?.content).toBe(thirdDraft)
    })
  })

  it('does not clear a hidden streaming conversation when deleting another selected chat', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Write a long story')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
    })

    const streamingConversationId = vi.mocked(saveChatState).mock.calls.at(-1)?.[1].activeConversationId
    if (!streamingConversationId) throw new Error('Expected streaming conversation id')
    const hookUnmountsBeforeDelete = chatMock.hookUnmounts.length

    await user.click(screen.getByRole('button', { name: 'New' }))
    expect(screen.getByText('Start Chatting')).toBeInTheDocument()
    expect(screen.queryByText('Partial assistant reply')).not.toBeInTheDocument()

    await user.click(await screen.findByRole('button', { name: 'Open actions for New chat' }))
    await user.click(await screen.findByRole('menuitem', { name: /delete/i }))
    await user.click(await screen.findByRole('button', { name: 'Delete chat' }))

    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeDelete)).not.toContain(streamingConversationId)
    await expectPartialAssistantReply()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toHaveTextContent('Streaming response...')

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.conversations).toHaveLength(1)
      expect(latestState?.activeConversationId).toBe(streamingConversationId)
      expect(latestState?.threads[streamingConversationId].map((message) => message.body)).toEqual([
        'Write a long story',
        'Partial assistant reply'
      ])
    })
  })

  it('persists an empty stopped assistant turn when a stream is stopped before tokens arrive', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Stop before any token')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(screen.getByRole('button', { name: 'Stop streaming' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stop streaming' }))

    expect(chatMock.stopCalls).toHaveLength(1)
    expect(await screen.findByText('(stopped)')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Stop before any token',
        ''
      ])
    })
  })

  it('returns to the latest message when the reader sends while scrolled up', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const messageList = screen.getByTestId('chat-message-list')
    setMessageListDimensions(messageList)
    messageList.scrollTop = 320
    fireEvent.scroll(messageList)

    await user.type(screen.getByLabelText('Prompt'), 'Follow the latest message')
    scrollIntoViewMock.mockClear()

    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()
    await waitFor(() => expect(scrollIntoViewMock).toHaveBeenCalledWith({ block: 'end' }))

    const scrollTarget = scrollIntoViewMock.mock.contexts.at(-1) as HTMLElement | undefined
    expect(scrollTarget).toHaveAttribute('data-chat-scroll-anchor', 'true')
    expect(messageList.scrollTop).toBe(1400)
  })

  it('renders and persists completed response metadata on live assistant messages', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = 'Response with measured metadata'
    chatMock.sendResponseMetadata = {
      model: 'unsloth/MiniMax-M2.5-GGUF:Q4_K_M',
      usage: { input_tokens: 9, output_tokens: 27, total_tokens: 36 },
      timings: { decode_time_ms: 1765, ttft_ms: 1116, total_time_ms: 2881 },
      servedBy: 'lemony-28'
    }

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Measure this response')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    const userHeader = screen.getByText('You').parentElement

    await waitFor(() => expect(userHeader).toHaveTextContent(chatMock.sendCalls[0]?.model ?? ''))
    expect(userHeader).not.toHaveTextContent('2026-05-06')
    expect(await screen.findByText('Response with measured metadata')).toBeInTheDocument()
    expect(screen.getByText('unsloth/MiniMax-M2.5-GGUF:Q4_K_M')).toBeInTheDocument()
    expect(await screen.findByText((_, element) => element?.textContent === '27 tok')).toBeInTheDocument()
    expect(await screen.findByText((_, element) => element?.textContent === '15.3 tok/s')).toBeInTheDocument()
    expect(await screen.findByText((_, element) => element?.textContent === 'TTFT 1116ms')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeThread = latestState?.activeConversationId
        ? latestState.threads[latestState.activeConversationId]
        : undefined
      expect(activeThread?.at(-1)).toMatchObject({
        id: 'assistant-1',
        messageRole: 'assistant',
        model: 'unsloth/MiniMax-M2.5-GGUF:Q4_K_M',
        route: 'lemony-28',
        routeNode: 'lemony-28',
        tokens: '27 tok',
        tokPerSec: '15.3 tok/s',
        ttft: '1116ms'
      })
    })
  })

  it('shows the streaming placeholder and drains a queued prompt with the latest selected model', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First live prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Streaming response...')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Queue' })).toBeDisabled()

    const messageList = screen.getByTestId('chat-message-list')
    setMessageListDimensions(messageList)
    messageList.scrollTop = 320
    fireEvent.scroll(messageList)
    await user.type(screen.getByLabelText('Prompt'), 'Run this next')
    scrollIntoViewMock.mockClear()
    await user.click(screen.getByRole('button', { name: 'Queue' }))

    expect(screen.getByLabelText('Prompt')).toHaveValue('')
    expect(screen.getByText('Run this next')).toBeInTheDocument()
    expect(screen.getByText('Queued')).toBeInTheDocument()
    expect(messageList.scrollTop).toBe(1400)
    expect(scrollIntoViewMock).toHaveBeenCalledWith({ block: 'end' })

    await user.click(screen.getByRole('combobox', { name: 'Select model' }))
    await user.click(await screen.findByText('Qwen3.5-0.8B-UD'))
    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.content).toBe('Run this next')
      expect(chatMock.sendCalls[1]?.model).toBe('Qwen3.5-0.8B-UD')
    })

    expect(screen.queryByText('Queued')).not.toBeInTheDocument()
    expect(screen.getAllByText('Qwen3.5-0.8B-UD')).not.toHaveLength(0)
  })

  it('removes a queued prompt before the stream drains it', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First live prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    expect(await screen.findByText('Streaming response...')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Do not send this')
    await user.click(screen.getByRole('button', { name: 'Queue' }))

    expect(screen.getByText('Do not send this')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Remove queued message' }))

    expect(screen.queryByText('Do not send this')).not.toBeInTheDocument()
    expect(screen.queryByText('Queued')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(1)
      expect(chatMock.sendCalls[0]?.content).toBe('First live prompt')
    })
  })

  it('keeps multiple queued prompts visible and removes only the selected queued item', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First live prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    expect(await screen.findByText('Streaming response...')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Queued alpha')
    await user.click(screen.getByRole('button', { name: 'Queue' }))
    await user.type(screen.getByLabelText('Prompt'), 'Queued beta')
    await user.click(screen.getByRole('button', { name: 'Queue' }))

    expect(screen.getByText('Queued alpha')).toBeInTheDocument()
    expect(screen.getByText('Queued beta')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Remove queued message' })).toHaveLength(2)

    await user.click(screen.getAllByRole('button', { name: 'Remove queued message' })[0])

    expect(screen.queryByText('Queued alpha')).not.toBeInTheDocument()
    expect(screen.getByText('Queued beta')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.content).toBe('Queued beta')
    })
    expect(chatMock.sendCalls.map((call) => call.content)).not.toContain('Queued alpha')
  })

  it('drains multiple queued prompts one at a time in FIFO order', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First live prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    expect(await screen.findByText('Streaming response...')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Queued alpha')
    await user.click(screen.getByRole('button', { name: 'Queue' }))
    await user.type(screen.getByLabelText('Prompt'), 'Queued beta')
    await user.click(screen.getByRole('button', { name: 'Queue' }))

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.content).toBe('Queued alpha')
    })
    expect(screen.getAllByRole('button', { name: 'Remove queued message' })).toHaveLength(1)
    expect(screen.getByText('Queued beta')).toBeInTheDocument()
    expect(chatMock.sendCalls.map((call) => call.content)).not.toContain('Queued beta')

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(3)
      expect(chatMock.sendCalls[2]?.content).toBe('Queued beta')
    })
    expect(screen.queryByText('Queued')).not.toBeInTheDocument()
  })
})

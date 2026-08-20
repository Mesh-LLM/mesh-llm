import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ChatLayout } from '@/features/chat/layouts/ChatLayout'
import {
  CHAT_HARNESS,
  adaptModelsToSummary,
  chatLayout,
  loadChatState,
  renderChatPage,
  scrollIntoViewMock,
  setMessageListDimensions,
  useModelsQuery,
  useStatusQuery
} from './ChatPage.test-support'

describe('ChatPage', () => {
  it('deselects the inspected message when the message area background is clicked', async () => {
    const user = userEvent.setup()

    renderChatPage({ transparencyTabEnabled: true })

    await user.click(screen.getByRole('button', { name: 'Inspect transparency' }))

    expect(screen.queryByText('No message selected')).not.toBeInTheDocument()

    await user.click(screen.getByTestId('chat-message-list'))

    expect(screen.getByText('No message selected')).toBeInTheDocument()
  })

  it('bounds the chat layout and makes the transcript the styled scroll container', () => {
    render(
      <ChatLayout
        actions={null}
        composer={<textarea aria-label="Prompt" />}
        sidebar={<div role="tablist" aria-label="Chat sidebar views" />}
        title="Chat"
      >
        <div data-testid="message-content">Messages</div>
      </ChatLayout>
    )

    expect(screen.getByTestId('chat-layout')).toHaveStyle({
      height: 'var(--chat-layout-height)',
      maxHeight: 'var(--chat-layout-height)'
    })
    expect(screen.getByTestId('chat-layout').style.getPropertyValue('--chat-layout-height')).toBe(
      'calc(100dvh - 180px)'
    )
    expect(screen.getByTestId('chat-message-list')).toHaveClass(
      'chat-message-scrollbar',
      'overflow-y-auto',
      'overflow-x-hidden'
    )
  })

  it('preserves a manual transcript scroll position across unrelated rerenders', () => {
    const { rerender } = render(chatLayout('1 node'))
    const messageList = screen.getByTestId('chat-message-list')

    setMessageListDimensions(messageList)
    messageList.scrollTop = 320
    fireEvent.scroll(messageList)
    scrollIntoViewMock.mockClear()

    rerender(chatLayout('2 nodes'))

    expect(messageList.scrollTop).toBe(320)
    expect(scrollIntoViewMock).not.toHaveBeenCalled()
  })

  it('resumes following transcript updates when the reader returns near the bottom', () => {
    const { rerender } = render(chatLayout('1 node'))
    const messageList = screen.getByTestId('chat-message-list')

    setMessageListDimensions(messageList)
    messageList.scrollTop = 320
    fireEvent.scroll(messageList)
    rerender(chatLayout('2 nodes'))
    expect(messageList.scrollTop).toBe(320)

    messageList.scrollTop = 940
    fireEvent.scroll(messageList)
    scrollIntoViewMock.mockClear()
    rerender(chatLayout('3 nodes'))

    expect(messageList.scrollTop).toBe(1400)
    expect(scrollIntoViewMock).toHaveBeenCalledWith({ block: 'end' })
  })

  it('uses the sticky-scroll threshold boundary when following transcript updates', () => {
    const { rerender } = render(chatLayout('1 node'))
    const messageList = screen.getByTestId('chat-message-list')

    setMessageListDimensions(messageList)
    messageList.scrollTop = 915
    fireEvent.scroll(messageList)
    scrollIntoViewMock.mockClear()
    rerender(chatLayout('2 nodes'))
    expect(messageList.scrollTop).toBe(915)
    expect(scrollIntoViewMock).not.toHaveBeenCalled()

    messageList.scrollTop = 916
    fireEvent.scroll(messageList)
    rerender(chatLayout('3 nodes'))
    expect(messageList.scrollTop).toBe(1400)
    expect(scrollIntoViewMock).toHaveBeenCalledWith({ block: 'end' })
  })

  it('does not run a queued transcript scroll after the reader scrolls upward', () => {
    const animationFrames: FrameRequestCallback[] = []
    const requestAnimationFrameSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
      animationFrames.push(callback)
      return animationFrames.length
    })
    const cancelAnimationFrameSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined)

    try {
      render(chatLayout('1 node'))
      const messageList = screen.getByTestId('chat-message-list')

      setMessageListDimensions(messageList)
      messageList.scrollTop = 320
      fireEvent.scroll(messageList)
      scrollIntoViewMock.mockClear()

      animationFrames.at(-1)?.(0)

      expect(messageList.scrollTop).toBe(320)
      expect(scrollIntoViewMock).not.toHaveBeenCalled()
    } finally {
      requestAnimationFrameSpy.mockRestore()
      cancelAnimationFrameSpy.mockRestore()
    }
  })

  it('returns to the latest transcript message when the sticky-scroll key changes', () => {
    const { rerender } = render(chatLayout('1 node', 'conversation-a:0'))
    const messageList = screen.getByTestId('chat-message-list')

    setMessageListDimensions(messageList)
    messageList.scrollTop = 320
    fireEvent.scroll(messageList)
    scrollIntoViewMock.mockClear()

    rerender(chatLayout('1 node', 'conversation-b:0'))

    expect(messageList.scrollTop).toBe(1400)
    expect(scrollIntoViewMock).toHaveBeenCalledWith({ block: 'end' })
  })

  it('falls back to harness conversations when persisted chat state is malformed', async () => {
    vi.mocked(loadChatState).mockResolvedValue({} as Awaited<ReturnType<typeof loadChatState>>)

    renderChatPage()

    expect(await screen.findAllByText('Routing latency notes')).not.toHaveLength(0)
  })

  it('does not show harness-scoped persisted conversations when rendering live mode', async () => {
    vi.mocked(loadChatState).mockImplementation(async (scope) => {
      if (scope !== 'harness') return undefined
      return {
        conversations: [{ id: 'persisted-harness', title: 'Harness persisted only', subtitle: '', updatedAt: 'Now' }],
        conversationGroups: [
          { title: 'Today', conversationIds: ['persisted-harness'] },
          { title: 'Earlier', conversationIds: [] }
        ],
        threads: { 'persisted-harness': [] },
        activeConversationId: 'persisted-harness'
      }
    })

    renderChatPage({ mode: 'live' })

    expect(await screen.findByText('Start Chatting')).toBeInTheDocument()
    expect(screen.queryByText('Harness persisted only')).not.toBeInTheDocument()
  })

  it('shows auto as the first chat model selector option by default', async () => {
    const user = userEvent.setup()

    renderChatPage()

    const trigger = screen.getByRole('combobox', { name: 'Select model' })
    expect(trigger).toHaveTextContent('Mesh — automatic')

    await user.click(trigger)

    const options = await screen.findAllByRole('option')
    expect(options[0]).toHaveTextContent('Auto')
  })

  it('renders usable live chat with status-backed models while catalog enrichment is loading', async () => {
    const user = userEvent.setup()
    vi.mocked(useModelsQuery).mockReturnValue({
      data: undefined,
      isFetching: true,
      isError: false,
      refetch: vi.fn()
    } as unknown as ReturnType<typeof useModelsQuery>)
    vi.mocked(useStatusQuery).mockReturnValue({
      data: {
        llama_ready: false,
        node_state: 'client',
        serving_models: [],
        peers: [{ hosted_models_known: false, serving_models: ['peer-model'] }]
      },
      isFetching: false,
      isError: false,
      refetch: vi.fn()
    } as unknown as ReturnType<typeof useStatusQuery>)

    renderChatPage({ mode: 'live' })

    expect(screen.getByText('Start Chatting')).toBeVisible()
    expect(screen.getByLabelText('Prompt')).toBeEnabled()
    const modelSelect = screen.getByRole('combobox', { name: 'Select model' })
    expect(modelSelect).toHaveTextContent('Mesh — automatic')

    await user.click(modelSelect)

    expect(screen.getByRole('option', { name: /peer-model/ })).toBeVisible()
  })

  it('keeps live chat usable when catalog enrichment fails but runtime status is ready', () => {
    vi.mocked(useModelsQuery).mockReturnValue({
      data: undefined,
      isFetching: false,
      isError: true,
      refetch: vi.fn()
    } as unknown as ReturnType<typeof useModelsQuery>)
    vi.mocked(useStatusQuery).mockReturnValue({
      data: { llama_ready: true, serving_models: ['local-model'], peers: [] },
      isFetching: false,
      isError: false,
      refetch: vi.fn()
    } as unknown as ReturnType<typeof useStatusQuery>)

    renderChatPage({ mode: 'live' })

    expect(screen.getByText('Start Chatting')).toBeVisible()
    expect(screen.getByLabelText('Prompt')).toBeEnabled()
  })

  it('uses a warm catalog without waiting for runtime status', () => {
    vi.mocked(adaptModelsToSummary).mockReturnValue([
      { ...CHAT_HARNESS.models[0], name: 'catalog-model', status: 'warm' }
    ])
    vi.mocked(useModelsQuery).mockReturnValue({
      data: { mesh_models: [{}] },
      isFetching: false,
      isError: false,
      refetch: vi.fn()
    } as unknown as ReturnType<typeof useModelsQuery>)
    vi.mocked(useStatusQuery).mockReturnValue({
      data: undefined,
      isFetching: true,
      isError: false,
      refetch: vi.fn()
    } as unknown as ReturnType<typeof useStatusQuery>)

    renderChatPage({ mode: 'live' })

    expect(screen.getByText('Start Chatting')).toBeVisible()
    expect(screen.getByLabelText('Prompt')).toBeEnabled()
  })

  it('keeps warm catalog chat usable if runtime status fails', () => {
    vi.mocked(adaptModelsToSummary).mockReturnValue([
      { ...CHAT_HARNESS.models[0], name: 'catalog-model', status: 'warm' }
    ])
    vi.mocked(useModelsQuery).mockReturnValue({
      data: { mesh_models: [{}] },
      isFetching: false,
      isError: false,
      refetch: vi.fn()
    } as unknown as ReturnType<typeof useModelsQuery>)
    vi.mocked(useStatusQuery).mockReturnValue({
      data: undefined,
      isFetching: false,
      isError: true,
      refetch: vi.fn()
    } as unknown as ReturnType<typeof useStatusQuery>)

    renderChatPage({ mode: 'live' })

    expect(screen.getByText('Start Chatting')).toBeVisible()
    expect(screen.getByLabelText('Prompt')).toBeEnabled()
  })

  it('excludes cold live models from the chat model selector', async () => {
    const user = userEvent.setup()
    vi.mocked(adaptModelsToSummary).mockReturnValue([
      { ...CHAT_HARNESS.models[0], name: 'warm-model', status: 'warm' },
      { ...CHAT_HARNESS.models[1], name: 'cold-model', status: 'offline' }
    ])

    renderChatPage({ mode: 'live' })

    await user.click(screen.getByRole('combobox', { name: 'Select model' }))

    const options = await screen.findAllByRole('option')
    expect(options.map((option) => option.textContent)).toEqual([
      expect.stringContaining('Auto'),
      expect.stringContaining('warm-model')
    ])
    expect(screen.queryByText('cold-model')).not.toBeInTheDocument()
  })
})

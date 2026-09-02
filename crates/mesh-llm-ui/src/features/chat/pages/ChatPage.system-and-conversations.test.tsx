import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import {
  APP_STORAGE_KEYS,
  CHAT_HARNESS,
  DEFAULT_SYSTEM_PROMPT,
  adaptModelsToSummary,
  chatMock,
  loadChatState,
  renderChatPage,
  saveChatState,
  scrollIntoViewMock,
  setLocalTime,
  setMessageListDimensions,
  shortTimestamp
} from './ChatPage.test-support'

describe('ChatPage', () => {
  it('loads the default system prompt when no user override is stored', async () => {
    const user = userEvent.setup()
    localStorage.clear()

    renderChatPage({ systemPromptButtonEnabled: true })

    const systemPromptButton = screen.getByRole('button', { name: 'System prompt' })
    await user.click(systemPromptButton)
    const dialog = screen.getByRole('dialog', { name: 'Set system prompt' })

    const textarea = within(dialog).getByLabelText('System prompt') as HTMLTextAreaElement
    expect(textarea.value).toContain('You are a helpful assistant running inside MeshLLM.')
  })

  it('hides the system prompt button until the chat feature flag is enabled', () => {
    renderChatPage()

    expect(screen.queryByRole('button', { name: 'System prompt' })).not.toBeInTheDocument()
  })

  it('sends the default system prompt while the editor feature flag is disabled', async () => {
    const user = userEvent.setup()

    renderChatPage()

    expect(screen.queryByRole('button', { name: 'System prompt' })).not.toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Tell me about mesh-llm')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({
        content: 'Tell me about mesh-llm',
        model: 'mesh',
        systemPrompt: DEFAULT_SYSTEM_PROMPT
      })
    })
  })

  it('opens, saves, prefills, and sends the chat-wide system prompt for new chats', async () => {
    const user = userEvent.setup()

    renderChatPage({ systemPromptButtonEnabled: true })

    const systemPromptButton = screen.getByRole('button', { name: 'System prompt' })
    await user.click(systemPromptButton)
    const dialog = screen.getByRole('dialog', { name: 'Set system prompt' })
    expect(dialog).toBeInTheDocument()

    const textarea = within(dialog).getByLabelText('System prompt')
    fireEvent.change(textarea, { target: { value: '' } })

    await user.type(textarea, 'Answer as a mesh-llm operator.')
    await user.click(screen.getByRole('button', { name: 'Save prompt' }))

    await waitFor(() => {
      expect(window.localStorage.getItem(APP_STORAGE_KEYS.chatSystemPrompt)).toBe('Answer as a mesh-llm operator.')
      expect(systemPromptButton).toHaveFocus()
    })

    await user.click(systemPromptButton)
    expect(
      within(screen.getByRole('dialog', { name: 'Set system prompt' })).getByLabelText('System prompt')
    ).toHaveValue('Answer as a mesh-llm operator.')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Prompt'), 'Summarize active models')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({
        content: 'Summarize active models',
        model: 'mesh',
        systemPrompt: 'Answer as a mesh-llm operator.'
      })
    })
  })

  it('sends a persisted system prompt while the editor feature flag is disabled', async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(APP_STORAGE_KEYS.chatSystemPrompt, 'Hidden saved instruction')

    renderChatPage()

    expect(screen.queryByRole('button', { name: 'System prompt' })).not.toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Route without hidden instructions')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({
        content: 'Route without hidden instructions',
        model: 'mesh',
        systemPrompt: 'Hidden saved instruction'
      })
    })
  })

  it('treats whitespace-only system prompts as cleared state', async () => {
    const user = userEvent.setup()

    renderChatPage({ systemPromptButtonEnabled: true })

    const systemPromptButton = screen.getByRole('button', { name: 'System prompt' })
    await user.click(systemPromptButton)
    const dialog = screen.getByRole('dialog', { name: 'Set system prompt' })

    const wsTextarea = within(dialog).getByLabelText('System prompt')
    fireEvent.change(wsTextarea, { target: { value: '' } })

    await user.type(wsTextarea, '   {Enter}  ')
    await user.click(screen.getByRole('button', { name: 'Save prompt' }))

    await waitFor(() => {
      expect(systemPromptButton).toHaveFocus()
    })

    await user.click(systemPromptButton)
    expect(
      within(screen.getByRole('dialog', { name: 'Set system prompt' })).getByLabelText('System prompt')
    ).toHaveValue('')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    await user.type(screen.getByLabelText('Prompt'), 'Send without a system prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({
        content: 'Send without a system prompt',
        model: 'mesh',
        systemPrompt: ''
      })
    })
  })

  it('keeps auto selected and sends auto with a single live model', async () => {
    const user = userEvent.setup()
    vi.mocked(adaptModelsToSummary).mockReturnValue(CHAT_HARNESS.models.slice(0, 1))

    renderChatPage({ mode: 'live' })

    expect(screen.getByRole('combobox', { name: 'Select model' })).toHaveTextContent('Mesh — automatic')

    await user.type(screen.getByLabelText('Prompt'), 'Use the router')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({ content: 'Use the router', model: 'mesh' })
    })
  })

  it('keeps Auto highlighted in the dropdown even when other live models are available', async () => {
    // Regression: the chat "Auto" pick routes to `mesh` on the wire,
    // but the dropdown's visible selection must stay on the Auto row.
    // With multiple models present, the buggy version drifted to the
    // first real model option because Radix Select couldn't find an
    // option matching value="mesh".
    const user = userEvent.setup()
    vi.mocked(adaptModelsToSummary).mockReturnValue(CHAT_HARNESS.models)

    renderChatPage({ mode: 'live' })

    const trigger = screen.getByRole('combobox', { name: 'Select model' })
    expect(trigger).toHaveTextContent('Mesh — automatic')

    await user.type(screen.getByLabelText('Prompt'), 'multi-model auto')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({ content: 'multi-model auto', model: 'mesh' })
    })

    // Trigger label must still read the Auto/Mesh label, not whichever
    // real model happened to be options[0].
    expect(trigger).toHaveTextContent('Mesh — automatic')
  })

  it('shows conversation metadata as message count followed by localized timestamp', async () => {
    vi.setSystemTime(new Date(2026, 7, 20, 12, 0, 0))
    const yesterday = new Date()
    yesterday.setDate(yesterday.getDate() - 1)

    renderChatPage()

    expect(
      await screen.findByText(`4 messages · ${shortTimestamp(setLocalTime(new Date(), 9, 42))}`)
    ).toBeInTheDocument()
    expect(screen.getByText(`2 messages · ${shortTimestamp(yesterday)}`)).toBeInTheDocument()
  })

  it('renames and deletes conversations from the row action menu', async () => {
    const user = userEvent.setup()

    renderChatPage()

    await user.click(await screen.findByRole('button', { name: 'Open actions for Routing latency notes' }))
    await user.click(await screen.findByRole('menuitem', { name: /rename/i }))

    const renameInput = screen.getByLabelText('Rename Routing latency notes')
    expect(renameInput).toHaveFocus()
    await user.clear(renameInput)
    await user.type(renameInput, 'Renamed route audit')
    await user.click(screen.getByRole('button', { name: 'Save chat title' }))

    expect(await screen.findAllByText('Renamed route audit')).not.toHaveLength(0)
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.conversations[0]).toMatchObject({ id: 'c1', title: 'Renamed route audit' })
    })

    await user.click(screen.getByRole('button', { name: 'Open actions for Renamed route audit' }))
    await user.click(await screen.findByRole('menuitem', { name: /delete/i }))

    const deleteDialog = await screen.findByRole('alertdialog', { name: 'Delete "Renamed route audit"?' })
    expect(deleteDialog).toHaveTextContent('This permanently removes the selected chat and its message history')
    await user.click(screen.getByRole('button', { name: 'Delete chat' }))

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.conversations.some((conversation) => conversation.id === 'c1')).toBe(false)
      expect(latestState?.threads.c1).toBeUndefined()
      expect(latestState?.activeConversationId).toBe('c2')
    })
    expect(screen.queryByText('Renamed route audit')).not.toBeInTheDocument()
  })

  it('restores the persisted live conversation selection and thread after reload', async () => {
    vi.mocked(loadChatState).mockImplementation(async (scope) => {
      if (scope !== 'live') return undefined

      return {
        conversations: [
          { id: 'live-a', title: 'Live first', subtitle: '', updatedAt: 'Now' },
          { id: 'live-b', title: 'Live restored', subtitle: '', updatedAt: 'Later' }
        ],
        conversationGroups: [
          { title: 'Today', conversationIds: ['live-a', 'live-b'] },
          { title: 'Earlier', conversationIds: [] }
        ],
        threads: {
          'live-a': [{ id: 'msg-a', messageRole: 'assistant', timestamp: 'Now', body: 'Wrong live thread' }],
          'live-b': [{ id: 'msg-b', messageRole: 'assistant', timestamp: 'Later', body: 'Restored live thread body' }]
        },
        activeConversationId: 'live-b'
      }
    })

    renderChatPage({ mode: 'live' })

    expect(await screen.findAllByText(/Live restored/)).not.toHaveLength(0)
    expect(screen.getByText('Restored live thread body')).toBeInTheDocument()
    expect(screen.queryByText('Wrong live thread')).not.toBeInTheDocument()
  })

  it('does not show stale live lane messages while switching between persisted live conversations', async () => {
    const user = userEvent.setup()
    vi.mocked(loadChatState).mockImplementation(async (scope) => {
      if (scope !== 'live') return undefined

      return {
        conversations: [
          { id: 'live-a', title: 'Live first', subtitle: '', updatedAt: 'Now' },
          { id: 'live-b', title: 'Live restored', subtitle: '', updatedAt: 'Later' }
        ],
        conversationGroups: [
          { title: 'Today', conversationIds: ['live-a', 'live-b'] },
          { title: 'Earlier', conversationIds: [] }
        ],
        threads: {
          'live-a': [{ id: 'msg-a', messageRole: 'assistant', timestamp: 'Now', body: 'First persisted body' }],
          'live-b': [{ id: 'msg-b', messageRole: 'assistant', timestamp: 'Later', body: 'Restored persisted body' }]
        },
        activeConversationId: 'live-b'
      }
    })

    renderChatPage({ mode: 'live' })

    await waitFor(() => expect(screen.getByText('Restored persisted body')).toBeInTheDocument())
    const messageList = screen.getByTestId('chat-message-list')
    setMessageListDimensions(messageList)
    messageList.scrollTop = 320
    fireEvent.scroll(messageList)
    scrollIntoViewMock.mockClear()

    await user.click((await screen.findAllByRole('button', { name: /Live first/i }))[0])

    expect(screen.queryByText('Restored persisted body')).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('First persisted body')).toBeInTheDocument())
    expect(messageList.scrollTop).toBe(1400)
    expect(scrollIntoViewMock).toHaveBeenCalledWith({ block: 'end' })
  })

  it('creates a live thread on send, enables Stop while streaming, preserves partial text on stop, and retries with reload semantics', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const retryButton = screen.getByRole('button', { name: 'Retry last' })
    expect(retryButton).toBeDisabled()

    await user.type(screen.getByLabelText('Prompt'), 'Hello from live mode')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(screen.getByLabelText('Prompt')).toHaveValue('')
    expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument()
    expect(screen.getByText(/Generating response/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toHaveTextContent('Streaming response...')

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(chatMock.sendCalls).toHaveLength(1)
      expect(chatMock.sendCalls[0]?.content).toBe('Hello from live mode')
      expect(latestState?.conversations).toHaveLength(1)
      expect(latestState?.conversations[0]?.id).toBe(latestState?.activeConversationId)
      expect(chatMock.sendCalls[0]?.conversationId).toBe(latestState?.activeConversationId)
      expect(chatMock.hookConversationIds).toContain(latestState?.activeConversationId)
      expect(Object.keys(latestState?.threads ?? {})).toEqual([latestState?.activeConversationId])
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Hello from live mode',
        'Partial assistant reply'
      ])
    })

    expect(retryButton).toBeEnabled()

    await user.click(screen.getByRole('button', { name: 'Stop streaming' }))

    expect(chatMock.stopCalls).toHaveLength(1)
    expect(await screen.findByRole('button', { name: 'Send' })).toBeInTheDocument()
    expect(screen.getByText('Partial assistant reply')).toBeInTheDocument()
    expect(screen.getByText('(stopped)')).toBeInTheDocument()

    chatMock.reloadAssistantText = 'Retried assistant reply'
    chatMock.reloadErrorMessage = 'Retry failed after replacing the last assistant reply'
    const messageList = screen.getByTestId('chat-message-list')
    setMessageListDimensions(messageList)
    messageList.scrollTop = 320
    fireEvent.scroll(messageList)
    scrollIntoViewMock.mockClear()

    await user.click(screen.getByRole('button', { name: 'Retry last' }))

    expect(await screen.findByText('Retried assistant reply')).toBeInTheDocument()
    expect(screen.queryByText('Partial assistant reply')).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Retry failed after replacing the last assistant reply')
    expect(messageList.scrollTop).toBe(1400)
    expect(scrollIntoViewMock).toHaveBeenCalledWith({ block: 'end' })

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Hello from live mode',
        'Retried assistant reply'
      ])
    })
  })

  it('keeps retried mesh progress folded before response metadata arrives', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Check this with the mesh')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    await user.click(screen.getByRole('button', { name: 'Stop streaming' }))

    chatMock.reloadAssistantText = 'Routing through mesh…</think>'
    chatMock.reloadStatus = 'streaming'
    await user.click(screen.getByRole('button', { name: 'Retry last' }))

    const disclosure = await screen.findByRole('button', {
      name: 'Consulting peers and corroborating responses… Show details'
    })
    expect(disclosure.closest('[data-thinking-state="active"]')).toBeInTheDocument()
    expect(screen.getByText('Routing through mesh…')).not.toBeVisible()
    expect(screen.queryByText('Thinking')).not.toBeInTheDocument()
  })
})

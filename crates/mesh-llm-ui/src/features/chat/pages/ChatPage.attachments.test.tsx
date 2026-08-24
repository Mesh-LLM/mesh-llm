import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { MultimodalContent } from '@tanstack/ai-client'
import { describe, expect, it, vi } from 'vitest'
import { ChatLayout } from '@/features/chat/layouts/ChatLayout'
import {
  attachmentPreprocessingMock,
  chatMock,
  createDeferred,
  createObjectUrlMock,
  expectPartialAssistantReply,
  queryAllByTextContent,
  renderChatPage,
  revokeObjectUrlMock,
  saveChatState,
  shortTimestamp,
  TestAttachmentProcessingStage
} from './ChatPage.test-support'

describe('ChatPage', () => {
  it('persists submitted live message model labels with the conversation thread', async () => {
    const user = userEvent.setup()
    const submittedModel = 'mesh'

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Persist the submitted model')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId
      expect(activeConversationId).toBeTruthy()
      expect(latestState?.threads[activeConversationId ?? '']).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ body: 'Persist the submitted model', messageRole: 'user', model: submittedModel }),
          expect.objectContaining({ body: 'Partial assistant reply', messageRole: 'assistant', model: submittedModel })
        ])
      )
    })
  })

  it('hides the transparency tab by default', () => {
    renderChatPage()

    expect(screen.queryByRole('tab', { name: /transparency/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Inspect transparency' })).not.toBeInTheDocument()
  })

  it('opens the responsive chat sidebar popover from the floating control', async () => {
    const user = userEvent.setup()

    render(
      <ChatLayout
        actions={null}
        composer={<textarea aria-label="Prompt" />}
        sidebar={<div role="tablist" aria-label="Chat sidebar views" />}
        sidebarMode="compact"
        title="Chat"
      >
        <div data-testid="message-content">Messages</div>
      </ChatLayout>
    )

    expect(screen.queryByRole('tablist', { name: 'Chat sidebar views' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Open chat sidebar' }))

    expect(screen.getAllByRole('tablist', { name: 'Chat sidebar views' })).toHaveLength(1)

    await user.keyboard('{Escape}')

    expect(screen.queryByRole('tablist', { name: 'Chat sidebar views' })).not.toBeInTheDocument()
  })

  it('does not show the floating sidebar control when the sidebar is hidden', () => {
    render(
      <ChatLayout
        actions={null}
        composer={<textarea aria-label="Prompt" />}
        hideSidebar
        sidebar={<div role="tablist" aria-label="Chat sidebar views" />}
        sidebarMode="compact"
        title="Chat"
      >
        <div data-testid="message-content">Messages</div>
      </ChatLayout>
    )

    expect(screen.queryByRole('button', { name: 'Open chat sidebar' })).not.toBeInTheDocument()
  })

  it('hides route disclosure text by default', () => {
    renderChatPage()

    expect(queryAllByTextContent('sent to lemony-28')).toHaveLength(0)
    expect(queryAllByTextContent('sent to carrack')).toHaveLength(0)
    expect(queryAllByTextContent('routed via carrack')).toHaveLength(0)
  })

  it('shows the transparency tab when the feature flag is enabled', () => {
    renderChatPage({ transparencyTabEnabled: true })

    expect(screen.getByRole('tab', { name: /transparency/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Inspect transparency' })).toBeInTheDocument()
    expect(queryAllByTextContent('sent to lemony-28').length).toBeGreaterThan(0)
    expect(queryAllByTextContent('sent to carrack').length).toBeGreaterThan(0)
    expect(queryAllByTextContent('routed via carrack').length).toBeGreaterThan(0)
  })

  it('creates and selects a new empty live conversation without copying previous messages', async () => {
    const user = userEvent.setup()
    chatMock.sendStatus = 'ready'

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Hello!')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await expectPartialAssistantReply()
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId
      expect(activeConversationId).toBeTruthy()
      expect(latestState?.threads[activeConversationId ?? ''].map((message) => message.body)).toEqual([
        'Hello!',
        'Partial assistant reply'
      ])
    })

    await user.click(screen.getByRole('button', { name: /new/i }))

    expect(await screen.findAllByText(/New chat/)).not.toHaveLength(0)
    expect(await screen.findByText('Start Chatting')).toBeInTheDocument()
    expect(screen.queryByText('Partial assistant reply')).not.toBeInTheDocument()
    expect(screen.getByText(`0 messages · ${shortTimestamp(new Date())}`)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByLabelText('Prompt')).toHaveFocus())
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId
      expect(activeConversationId).toBeTruthy()
      expect(latestState?.threads[activeConversationId ?? '']).toEqual([])
    })
  })

  it('opens the hidden file picker from Attach', async () => {
    const user = userEvent.setup()
    const clickSpy = vi.spyOn(HTMLInputElement.prototype, 'click')

    renderChatPage({ mode: 'live' })

    await user.click(screen.getByRole('button', { name: 'Attach' }))

    expect(clickSpy).toHaveBeenCalled()
  })

  it('sends legacy-compatible attachment content for image and scanned pdf attachments', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'cat.png', { type: 'image/png' })
    const pdf = new File(['pdf-bytes'], 'scan.pdf', { type: 'application/pdf' })

    await user.upload(picker, [image, pdf])
    expect(screen.getByText('2 attachments ready')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Summarize these')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(1)
    })

    const content = chatMock.sendCalls[0]?.content
    expect(typeof content).not.toBe('string')
    expect((content as MultimodalContent).content).toEqual([
      { type: 'text', content: 'Summarize these' },
      { type: 'text', content: '[Image description: A tabby cat]' },
      { type: 'text', content: '[Content from scan.pdf]\n\n[Page 1]\n[Image description: A scanned receipt]' }
    ])
    expect(attachmentPreprocessingMock.describeImageForPrompt).toHaveBeenCalledTimes(1)
    expect(attachmentPreprocessingMock.extractPdfTextFromFile).toHaveBeenCalledWith(pdf)
    expect(attachmentPreprocessingMock.describeScannedPdf).toHaveBeenCalledWith(pdf, expect.any(Function))
  })

  it('shows submitted attachment chips on the user message and opens an image preview', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'cat.png', { type: 'image/png' })

    await user.upload(picker, image)
    await user.type(screen.getByLabelText('Prompt'), 'Describe this image to me')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(screen.getByText('Describe this image to me')).toBeInTheDocument()
    })

    await waitFor(() => expect(createObjectUrlMock).toHaveBeenCalledWith(image))
    const chip = await screen.findByRole('button', { name: 'Open cat.png' })
    expect(chip).toHaveTextContent('Image 1')

    await user.click(chip)

    const dialog = await screen.findByRole('dialog', { name: /cat\.png/i })
    expect(within(dialog).getByText('Image 1 · image/png')).toBeInTheDocument()
    expect(within(dialog).getByRole('img', { name: 'cat.png' })).toHaveAttribute('src', 'blob:preview/cat.png')
  })

  it('shows staged attachment preparation feedback before the chat prompt is submitted', async () => {
    const user = userEvent.setup()
    const imageDescription = createDeferred<{ imageDescription?: string }>()
    attachmentPreprocessingMock.describeImageForPrompt.mockImplementation(
      async (_dataUrl: string, onStage?: (stage: TestAttachmentProcessingStage) => void) => {
        onStage?.('starting')
        await Promise.resolve()
        onStage?.('processing')
        return imageDescription.promise
      }
    )

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'slow.png', { type: 'image/png' })

    await user.upload(picker, image)
    await user.type(screen.getByLabelText('Prompt'), 'What is in this image?')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    const status = await screen.findByLabelText('Attachment preparation status')
    expect(status).toHaveTextContent('Preparing attachments')
    expect(status).toHaveTextContent('Downloading')
    expect(status).toHaveTextContent('Starting')
    expect(status).toHaveTextContent('Processing')
    expect(status).toHaveTextContent('Prompt waiting: What is in this image?')
    await waitFor(() => expect(status).toHaveTextContent('Processing attachment content'))
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
    expect(screen.getByText('Processing attachments…')).toBeInTheDocument()
    expect(chatMock.sendCalls).toHaveLength(0)

    imageDescription.resolve({ imageDescription: '[Image description: A slow diagram]' })

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(1)
    })
    expect(screen.queryByLabelText('Attachment preparation status')).not.toBeInTheDocument()
  })

  it('reuses the loaded browser analyzer state for later attachment submissions', async () => {
    const user = userEvent.setup()
    const imageDescription = createDeferred<{ imageDescription?: string }>()
    attachmentPreprocessingMock.isBrowserVisionModelLoaded.mockReturnValue(true)
    attachmentPreprocessingMock.describeImageForPrompt.mockImplementation(
      async (_dataUrl: string, onStage?: (stage: TestAttachmentProcessingStage) => void) => {
        onStage?.('processing')
        return imageDescription.promise
      }
    )

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'cached.png', { type: 'image/png' })

    await user.upload(picker, image)
    await user.type(screen.getByLabelText('Prompt'), 'Use the cached analyzer')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    const status = await screen.findByLabelText('Attachment preparation status')
    expect(status).toHaveTextContent('Processing attachment content')
    expect(status).toHaveTextContent('Cached')
    expect(status).toHaveTextContent('Reusing the browser analyzer already loaded in this tab.')
    expect(status).toHaveTextContent('Ready')
    expect(status).toHaveTextContent('The local vision and document pipeline is already warm.')
    expect(status).not.toHaveTextContent('Downloading browser model')
    expect(status).not.toHaveTextContent('Fetching the browser analyzer and attachment assets.')
    expect(status).not.toHaveTextContent('Warming the local vision and document pipeline.')
    expect(chatMock.sendCalls).toHaveLength(0)

    imageDescription.resolve({ imageDescription: '[Image description: Cached run]' })

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(1)
    })
  })

  it('preserves the prompt and queued attachment when upload fails before send completes', async () => {
    const user = userEvent.setup()
    chatMock.sendErrorMessage = 'Upload failed: 503'
    chatMock.sendErrorResolves = true
    chatMock.sendStatus = 'submitted'
    chatMock.sendOptimisticStatusBeforeError = true
    chatMock.sendOptimisticUserMessageBeforeError = true
    chatMock.sendOptimisticAssistantPlaceholderBeforeError = true

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const audio = new File(['audio-bytes'], 'clip.mp3', { type: 'audio/mpeg' })

    await user.upload(picker, audio)
    await user.type(screen.getByLabelText('Prompt'), 'Keep this prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Upload failed: 503')
    await waitFor(() => expect(createObjectUrlMock).toHaveBeenCalledWith(audio))
    await waitFor(() => expect(revokeObjectUrlMock).toHaveBeenCalledWith('blob:preview/clip.mp3'))
    expect(screen.getByTestId('chat-message-list')).toHaveTextContent('Keep this prompt')
    await waitFor(() => {
      expect(screen.getByLabelText('Prompt')).toHaveValue('Keep this prompt')
    })
    expect(chatMock.sendCalls).toHaveLength(1)

    chatMock.sendErrorMessage = undefined
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
    })

    const retriedContent = chatMock.sendCalls[1]?.content
    expect(typeof retriedContent).not.toBe('string')
    expect((retriedContent as MultimodalContent).content).toEqual([
      { type: 'text', content: 'Keep this prompt' },
      {
        type: 'audio',
        source: { type: 'data', value: 'YXVkaW8tYnl0ZXM=', mimeType: 'audio/mpeg' },
        metadata: { fileName: 'clip.mp3' }
      }
    ])
  })

  it('restores the prompt and attachment when a send request fails generically', async () => {
    const user = userEvent.setup()
    chatMock.sendErrorMessage = 'Network failed'

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'diagram.png', { type: 'image/png' })

    await user.upload(picker, image)
    await user.type(screen.getByLabelText('Prompt'), 'Keep this generic failure draft')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Network failed')
    expect(screen.getByTestId('chat-message-list')).toHaveTextContent('Keep this generic failure draft')
    await waitFor(() => {
      expect(screen.getByLabelText('Prompt')).toHaveValue('Keep this generic failure draft')
    })
    expect(chatMock.sendCalls).toHaveLength(1)

    chatMock.sendErrorMessage = undefined
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
    })
    const retriedContent = chatMock.sendCalls[1]?.content
    expect(typeof retriedContent).not.toBe('string')
    expect((retriedContent as MultimodalContent).content).toEqual([
      { type: 'text', content: 'Keep this generic failure draft' },
      { type: 'text', content: '[Image description: A tabby cat]' }
    ])
  })
})

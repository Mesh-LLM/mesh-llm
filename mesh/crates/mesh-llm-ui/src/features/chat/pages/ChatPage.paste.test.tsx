import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { MultimodalContent } from '@tanstack/ai-client'
import { describe, expect, it } from 'vitest'
import { chatMock, renderChatPage } from './ChatPage.test-support'

function pasteIntoComposer(files: File[]) {
  return fireEvent.paste(screen.getByLabelText('Prompt'), {
    clipboardData: { files, getData: () => '' }
  })
}

describe('ChatPage composer paste', () => {
  it('attaches a pasted image as a chip and sends it through the attachment pipeline', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const image = new File(['image-bytes'], 'screenshot.png', { type: 'image/png' })
    const consumed = pasteIntoComposer([image])

    expect(consumed).toBe(false)
    expect(screen.getByTestId('composer-attachments')).toHaveTextContent('screenshot.png')

    await user.type(screen.getByLabelText('Prompt'), 'Describe this')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(1)
    })
    const content = chatMock.sendCalls[0]?.content
    expect(typeof content).not.toBe('string')
    expect((content as MultimodalContent).content).toEqual([
      { type: 'text', content: 'Describe this' },
      { type: 'text', content: '[Image description: A tabby cat]' }
    ])
  })

  it('removes a pasted image from the composer chips before sending', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    pasteIntoComposer([new File(['image-bytes'], 'screenshot.png', { type: 'image/png' })])
    expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled()

    await user.click(screen.getByRole('button', { name: 'Remove attachment screenshot.png' }))

    expect(screen.queryByTestId('composer-attachments')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
  })

  it('does not prevent plain-text pastes', () => {
    renderChatPage({ mode: 'live' })

    const notConsumed = fireEvent.paste(screen.getByLabelText('Prompt'), {
      clipboardData: { files: [], getData: () => 'plain text' }
    })

    expect(notConsumed).toBe(true)
    expect(screen.queryByTestId('composer-attachments')).not.toBeInTheDocument()
  })

  it('ignores non-image clipboard files', () => {
    renderChatPage({ mode: 'live' })

    const notConsumed = pasteIntoComposer([new File(['pdf-bytes'], 'doc.pdf', { type: 'application/pdf' })])

    expect(notConsumed).toBe(true)
    expect(screen.queryByTestId('composer-attachments')).not.toBeInTheDocument()
  })

  it('attaches every image from a multi-file paste', () => {
    renderChatPage({ mode: 'live' })

    const consumed = pasteIntoComposer([
      new File(['image-bytes'], 'first.png', { type: 'image/png' }),
      new File(['more-image-bytes'], 'second.jpg', { type: 'image/jpeg' })
    ])

    expect(consumed).toBe(false)
    const pendingAttachments = screen.getByTestId('composer-attachments')
    expect(pendingAttachments).toHaveTextContent('first.png')
    expect(pendingAttachments).toHaveTextContent('second.jpg')
  })

  it('ignores a paste event without clipboardData', () => {
    renderChatPage({ mode: 'live' })

    const notConsumed = fireEvent.paste(screen.getByLabelText('Prompt'), {})

    expect(notConsumed).toBe(true)
    expect(screen.queryByTestId('composer-attachments')).not.toBeInTheDocument()
  })

  it('attaches a pasted image while a response is streaming', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Start a response')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toBeInTheDocument()

    const consumed = pasteIntoComposer([new File(['image-bytes'], 'during-stream.png', { type: 'image/png' })])

    expect(consumed).toBe(false)
    expect(screen.getByTestId('composer-attachments')).toHaveTextContent('during-stream.png')
  })

  it('removes only one chip when two pasted files share a name', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    pasteIntoComposer([
      new File(['first-bytes'], 'same.png', { type: 'image/png' }),
      new File(['second-bytes'], 'same.png', { type: 'image/png' })
    ])
    expect(screen.getAllByRole('button', { name: 'Remove attachment same.png' })).toHaveLength(2)

    await user.click(screen.getAllByRole('button', { name: 'Remove attachment same.png' })[0])

    expect(screen.getAllByRole('button', { name: 'Remove attachment same.png' })).toHaveLength(1)
    expect(screen.getByTestId('composer-attachments')).toBeInTheDocument()
  })
})

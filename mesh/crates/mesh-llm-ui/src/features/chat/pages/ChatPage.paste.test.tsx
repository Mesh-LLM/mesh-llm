import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { MultimodalContent } from '@tanstack/ai-client'
import { describe, expect, it, vi } from 'vitest'
import { chatMock, renderChatPage } from './ChatPage.test-support'

vi.mock('@/lib/attachments', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/attachments')>()
  return { ...actual, resolveImageFiles: vi.fn(actual.resolveImageFiles) }
})

const PNG_MAGIC_BYTES = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

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

  it('attaches a pasted image with an empty MIME type after sniffing its bytes', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const image = new File([new Uint8Array(PNG_MAGIC_BYTES)], 'clipboard-image', { type: '' })
    const consumed = pasteIntoComposer([image])

    expect(consumed).toBe(false)
    expect(await screen.findByText('clipboard-image')).toBeInTheDocument()

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

  it('drops a pasted file with an empty MIME type when its bytes are not an image', async () => {
    renderChatPage({ mode: 'live' })

    const notConsumedAsText = pasteIntoComposer([new File(['%PDF-1.7'], 'doc.pdf', { type: '' })])

    expect(notConsumedAsText).toBe(false)
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(screen.queryByTestId('composer-attachments')).not.toBeInTheDocument()
    expect(screen.queryByText('doc.pdf')).not.toBeInTheDocument()
  })

  it('restores the clipboard text when an undetermined file turns out not to be an image', async () => {
    renderChatPage({ mode: 'live' })

    const notConsumedAsText = fireEvent.paste(screen.getByLabelText('Prompt'), {
      clipboardData: {
        files: [new File(['%PDF-1.7'], 'doc.pdf', { type: '' })],
        getData: (type: string) => (type === 'text/plain' ? '/home/alex/report.pdf' : '')
      }
    })

    expect(notConsumedAsText).toBe(false)
    await waitFor(() => {
      expect(screen.getByLabelText('Prompt')).toHaveValue('/home/alex/report.pdf')
    })
    expect(screen.queryByTestId('composer-attachments')).not.toBeInTheDocument()
  })

  it('restores clipboard text at the caret, not at the end of the prompt', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })
    const prompt = screen.getByLabelText('Prompt')
    await user.type(prompt, 'Hello world')
    ;(prompt as HTMLTextAreaElement).setSelectionRange(5, 5)

    fireEvent.paste(prompt, {
      clipboardData: {
        files: [new File(['%PDF-1.7'], 'doc.pdf', { type: '' })],
        getData: (type: string) => (type === 'text/plain' ? '/report.pdf' : '')
      }
    })

    await waitFor(() => {
      expect(prompt).toHaveValue('Hello/report.pdf world')
    })
  })

  it('restores clipboard text over a selection instead of appending it', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })
    const prompt = screen.getByLabelText('Prompt')
    await user.type(prompt, 'Hello world')
    ;(prompt as HTMLTextAreaElement).setSelectionRange(6, 11)

    fireEvent.paste(prompt, {
      clipboardData: {
        files: [new File(['%PDF-1.7'], 'doc.pdf', { type: '' })],
        getData: (type: string) => (type === 'text/plain' ? 'there' : '')
      }
    })

    await waitFor(() => {
      expect(prompt).toHaveValue('Hello there')
    })
  })

  it('attaches the typed image from a mixed paste and drops the undetected file', async () => {
    renderChatPage({ mode: 'live' })

    const consumed = pasteIntoComposer([
      new File(['image-bytes'], 'shot.png', { type: 'image/png' }),
      new File(['%PDF-1.7'], 'doc.pdf', { type: '' })
    ])

    expect(consumed).toBe(false)
    const pendingAttachments = await screen.findByTestId('composer-attachments')
    expect(pendingAttachments).toHaveTextContent('shot.png')
    expect(pendingAttachments).not.toHaveTextContent('doc.pdf')
  })

  it('appends instead of overwriting when the prompt is edited while detection is pending', async () => {
    const user = userEvent.setup()
    const { resolveImageFiles } = await import('@/lib/attachments')
    let releaseSniff!: (files: File[]) => void
    vi.mocked(resolveImageFiles).mockImplementationOnce(
      () =>
        new Promise<File[]>((resolve) => {
          releaseSniff = resolve
        })
    )

    renderChatPage({ mode: 'live' })
    const prompt = screen.getByLabelText('Prompt')
    await user.type(prompt, 'Hello world')
    ;(prompt as HTMLTextAreaElement).setSelectionRange(6, 11)

    const file = new File(['%PDF-1.7'], 'doc.pdf', { type: '' })
    fireEvent.paste(prompt, {
      clipboardData: {
        files: [file],
        getData: (type: string) => (type === 'text/plain' ? '/notes.txt' : '')
      }
    })

    await user.type(prompt, '!')
    expect(prompt).toHaveValue('Hello world!')

    releaseSniff([file])

    await waitFor(() => {
      expect(prompt).toHaveValue('Hello world!/notes.txt')
    })
    expect(screen.queryByTestId('composer-attachments')).not.toBeInTheDocument()
  })

  it('keeps both texts when two undetermined pastes resolve back to back', async () => {
    const { resolveImageFiles } = await import('@/lib/attachments')
    let releaseFirst!: (files: File[]) => void
    let releaseSecond!: (files: File[]) => void
    vi.mocked(resolveImageFiles)
      .mockImplementationOnce(
        () =>
          new Promise<File[]>((resolve) => {
            releaseFirst = resolve
          })
      )
      .mockImplementationOnce(
        () =>
          new Promise<File[]>((resolve) => {
            releaseSecond = resolve
          })
      )

    renderChatPage({ mode: 'live' })
    const prompt = screen.getByLabelText('Prompt')

    const first = new File(['%PDF-1.7'], 'one.pdf', { type: '' })
    fireEvent.paste(prompt, {
      clipboardData: { files: [first], getData: (type: string) => (type === 'text/plain' ? 'FIRST' : '') }
    })
    const second = new File(['%PDF-2.7'], 'two.pdf', { type: '' })
    fireEvent.paste(prompt, {
      clipboardData: { files: [second], getData: (type: string) => (type === 'text/plain' ? 'SECOND' : '') }
    })

    await act(async () => {
      releaseFirst([first])
      releaseSecond([second])
    })

    await waitFor(() => {
      expect(prompt).toHaveValue('FIRSTSECOND')
    })
    expect(screen.queryByTestId('composer-attachments')).not.toBeInTheDocument()
  })

  it('places concurrent restorations at the caret in the middle of a nonempty prompt', async () => {
    const user = userEvent.setup()
    const { resolveImageFiles } = await import('@/lib/attachments')
    let releaseFirst!: (files: File[]) => void
    let releaseSecond!: (files: File[]) => void
    vi.mocked(resolveImageFiles)
      .mockImplementationOnce(
        () =>
          new Promise<File[]>((resolve) => {
            releaseFirst = resolve
          })
      )
      .mockImplementationOnce(
        () =>
          new Promise<File[]>((resolve) => {
            releaseSecond = resolve
          })
      )

    renderChatPage({ mode: 'live' })
    const prompt = screen.getByLabelText('Prompt')
    await user.type(prompt, 'Hello world')
    ;(prompt as HTMLTextAreaElement).setSelectionRange(5, 5)

    const first = new File(['%PDF-1.7'], 'one.pdf', { type: '' })
    fireEvent.paste(prompt, {
      clipboardData: { files: [first], getData: (type: string) => (type === 'text/plain' ? 'FIRST' : '') }
    })
    const second = new File(['%PDF-2.7'], 'two.pdf', { type: '' })
    fireEvent.paste(prompt, {
      clipboardData: { files: [second], getData: (type: string) => (type === 'text/plain' ? 'SECOND' : '') }
    })

    // Both detections resolve after both captures; the restorations must
    // land back to back at the caret, in paste order, regardless of the
    // resolution order.
    await act(async () => {
      releaseSecond([second])
      releaseFirst([first])
    })

    await waitFor(() => {
      expect(prompt).toHaveValue('HelloFIRSTSECOND world')
    })
    expect(screen.queryByTestId('composer-attachments')).not.toBeInTheDocument()
  })

  it('blocks sending while an undetermined paste detection is still running', async () => {
    const { resolveImageFiles } = await import('@/lib/attachments')
    let releaseSniff!: (files: File[]) => void
    vi.mocked(resolveImageFiles).mockImplementationOnce(
      () =>
        new Promise<File[]>((resolve) => {
          releaseSniff = resolve
        })
    )

    renderChatPage({ mode: 'live' })
    const prompt = screen.getByLabelText('Prompt')
    await userEvent.type(prompt, 'Describe this')

    fireEvent.paste(prompt, {
      clipboardData: {
        files: [new File(['%PDF-1.7'], 'doc.pdf', { type: '' })],
        getData: () => ''
      }
    })

    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()

    releaseSniff([new File(['%PDF-1.7'], 'doc.pdf', { type: '' })])
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled()
    })
  })

  it('keeps a queued restoration from shifting later pastes when it becomes an image', async () => {
    const { resolveImageFiles } = await import('@/lib/attachments')
    let releaseFirst!: (files: File[]) => void
    let releaseSecond!: (files: File[]) => void
    const firstFile = new File(['%PDF-1.7'], 'one.pdf', { type: '' })
    const imageFile = new File([new Uint8Array(PNG_MAGIC_BYTES)], 'shot.png', { type: 'image/png' })
    vi.mocked(resolveImageFiles)
      .mockImplementationOnce(
        () =>
          new Promise<File[]>((resolve) => {
            releaseFirst = resolve
          })
      )
      .mockImplementationOnce(
        () =>
          new Promise<File[]>((resolve) => {
            releaseSecond = resolve
          })
      )

    renderChatPage({ mode: 'live' })
    const prompt = screen.getByLabelText('Prompt')
    await userEvent.type(prompt, 'Hello world')
    ;(prompt as HTMLTextAreaElement).setSelectionRange(5, 5)

    // First paste turns out to be an image; the second paste's text must
    // still land at the original caret, unshifted by the skipped restore.
    fireEvent.paste(prompt, {
      clipboardData: { files: [firstFile], getData: (type: string) => (type === 'text/plain' ? 'FIRST' : '') }
    })
    fireEvent.paste(prompt, {
      clipboardData: {
        files: [new File(['%PDF-2.7'], 'two.pdf', { type: '' })],
        getData: (type: string) => (type === 'text/plain' ? 'SECOND' : '')
      }
    })

    await act(async () => {
      releaseFirst([imageFile])
    })
    await screen.findByTestId('composer-attachments')

    const secondFile = new File(['%PDF-2.7'], 'two.pdf', { type: '' })
    await act(async () => {
      releaseSecond([secondFile])
    })

    await waitFor(() => {
      expect(prompt).toHaveValue('HelloSECOND world')
    })
  })
})

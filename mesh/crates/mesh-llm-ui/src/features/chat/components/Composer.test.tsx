import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Composer } from './Composer'

vi.mock('@/lib/attachments', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/attachments')>()
  return { ...actual, resolveImageFiles: vi.fn(actual.resolveImageFiles) }
})

describe('Composer request eligibility', () => {
  it('blocks sending and retry but preserves Stop and Escape for an active response', () => {
    const onStop = vi.fn()
    const onSend = vi.fn()
    render(
      <Composer
        value="Next prompt"
        onChange={vi.fn()}
        onSend={onSend}
        onStop={onStop}
        onRetry={vi.fn()}
        canRetry
        isStreaming
        requestDisabled
      />
    )
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled()
    expect(screen.getByLabelText('Prompt')).toBeEnabled()
    fireEvent.keyDown(screen.getByLabelText('Prompt'), { key: 'Enter' })
    expect(onSend).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    fireEvent.keyDown(screen.getByLabelText('Prompt'), { key: 'Escape' })
    expect(onStop).toHaveBeenCalledTimes(2)
  })
})

describe('Composer pending type detection gate', () => {
  it('keeps the count with the owner when a detection outlives the composer', async () => {
    const { resolveImageFiles } = await import('@/lib/attachments')
    let releaseSniff!: (files: File[]) => void
    vi.mocked(resolveImageFiles).mockImplementationOnce(
      () =>
        new Promise<File[]>((resolve) => {
          releaseSniff = resolve
        })
    )

    const onPendingTypeDetectionChange = vi.fn()
    const { unmount } = render(
      <Composer
        value="Draft"
        onChange={vi.fn()}
        onSend={vi.fn()}
        onAttach={vi.fn()}
        onPendingTypeDetectionChange={onPendingTypeDetectionChange}
      />
    )

    fireEvent.paste(screen.getByLabelText('Prompt'), {
      clipboardData: {
        files: [new File(['%PDF-1.7'], 'doc.pdf', { type: '' })],
        getData: () => 'payload'
      }
    })

    expect(onPendingTypeDetectionChange).toHaveBeenCalledWith(1)

    // A conversation switch remounts the composer; the pending count must
    // survive with the owner, so the completion signal still arrives.
    unmount()
    releaseSniff([new File(['%PDF-1.7'], 'doc.pdf', { type: '' })])
    await vi.waitFor(() => {
      expect(onPendingTypeDetectionChange).toHaveBeenCalledWith(-1)
    })
  })

  it('blocks sending while the owner reports detection pending', () => {
    const onSend = vi.fn()
    const { rerender } = render(
      <Composer value="Draft" onChange={vi.fn()} onSend={onSend} onAttach={vi.fn()} pendingTypeDetection={true} />
    )
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
    fireEvent.keyDown(screen.getByLabelText('Prompt'), { key: 'Enter' })
    expect(onSend).not.toHaveBeenCalled()

    rerender(
      <Composer value="Draft" onChange={vi.fn()} onSend={onSend} onAttach={vi.fn()} pendingTypeDetection={false} />
    )
    expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled()
  })
})

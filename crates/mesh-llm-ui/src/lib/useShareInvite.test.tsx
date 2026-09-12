import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { shareButtonLabel, shareSheetAvailable, useShareInvite } from '@/lib/useShareInvite'

function installShare(share: unknown) {
  Object.defineProperty(navigator, 'share', { configurable: true, value: share })
}

function installClipboard(writeText: unknown) {
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
}

afterEach(() => {
  Object.defineProperty(navigator, 'share', { configurable: true, value: undefined })
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })
})

describe('useShareInvite', () => {
  it('uses the platform share sheet when one exists', async () => {
    const share = vi.fn().mockResolvedValue(undefined)
    installShare(share)
    const { result } = renderHook(() => useShareInvite())

    await act(async () => {
      await expect(result.current.share('join me', 'Join my Mesh')).resolves.toBe('shared')
    })

    expect(share).toHaveBeenCalledWith({ title: 'Join my Mesh', text: 'join me' })
    expect(result.current.outcome).toBe('shared')
  })

  it('falls back to the clipboard when no share sheet exists', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    installClipboard(writeText)
    const { result } = renderHook(() => useShareInvite())

    await act(async () => {
      await expect(result.current.share('join me')).resolves.toBe('copied')
    })

    expect(writeText).toHaveBeenCalledWith('join me')
  })

  it('treats a dismissed share sheet as dismissed and never copies behind the user', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    installClipboard(writeText)
    installShare(vi.fn().mockRejectedValue(new DOMException('cancelled', 'AbortError')))
    const { result } = renderHook(() => useShareInvite())

    await act(async () => {
      await expect(result.current.share('join me')).resolves.toBe('dismissed')
    })

    expect(writeText).not.toHaveBeenCalled()
  })

  it('reports a real share failure distinctly from a dismissal', async () => {
    installShare(vi.fn().mockRejectedValue(new Error('boom')))
    const { result } = renderHook(() => useShareInvite())

    await act(async () => {
      await expect(result.current.share('join me')).resolves.toBe('failed')
    })
  })

  it('fails when neither a share sheet nor a clipboard is available', async () => {
    const { result } = renderHook(() => useShareInvite())

    await act(async () => {
      await expect(result.current.share('join me')).resolves.toBe('failed')
    })
  })
})

describe('shareSheetAvailable / shareButtonLabel', () => {
  it('detects share-sheet support', () => {
    expect(shareSheetAvailable()).toBe(false)
    installShare(vi.fn())
    expect(shareSheetAvailable()).toBe(true)
  })

  it('labels the button for the available mechanism and outcome', () => {
    expect(shareButtonLabel(null, true)).toBe('Share')
    expect(shareButtonLabel(null, false)).toBe('Copy invite')
    expect(shareButtonLabel('shared', true)).toBe('Shared')
    expect(shareButtonLabel('copied', false)).toBe('Copied')
    expect(shareButtonLabel('failed', true)).toBe('Failed')
    expect(shareButtonLabel('dismissed', true)).toBe('Share')
  })
})

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readStoredChatSystemPrompt, writeStoredChatSystemPrompt } from './system-prompt'

const storageKey = 'test-system-prompt'

describe('chat system prompt persistence', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it('defaults to empty without writing a prompt', () => {
    expect(readStoredChatSystemPrompt(storageKey)).toBe('')
    expect(localStorage.getItem(storageKey)).toBeNull()
  })

  it('preserves custom instructions across reads', () => {
    const custom = '  Answer concisely.\nUse examples.  '
    writeStoredChatSystemPrompt(custom, storageKey)
    expect(readStoredChatSystemPrompt(storageKey)).toBe(custom)
  })

  it.each(['', '   \n', '\u0000'])('preserves previously cleared state %j', (stored) => {
    localStorage.setItem(storageKey, stored)
    expect(readStoredChatSystemPrompt(storageKey)).toBe('')
  })

  it('uses an empty prompt when storage is unavailable', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('Storage unavailable')
    })
    expect(readStoredChatSystemPrompt(storageKey)).toBe('')
  })
})

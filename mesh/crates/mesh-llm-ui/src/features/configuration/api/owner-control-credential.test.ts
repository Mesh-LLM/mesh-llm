// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import { getOwnerControlCredential, setOwnerControlCredential } from './owner-control-credential'
import { fetchRuntimeControlConfigSnapshot } from './config-adapter-http'

afterEach(() => {
  setOwnerControlCredential(null)
  document.body.replaceChildren()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('owner-control credential entry', () => {
  it('accepts a token explicitly and retains it only in memory', async () => {
    Object.defineProperty(HTMLDialogElement.prototype, 'showModal', { configurable: true, value: vi.fn() })
    Object.defineProperty(HTMLDialogElement.prototype, 'close', { configurable: true, value: vi.fn() })
    const pending = getOwnerControlCredential()
    const input = document.querySelector('input')!
    expect(input.type).toBe('password')
    input.value = '  test-token  '
    document.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
    expect(await pending).toBe('test-token')
    expect(await getOwnerControlCredential()).toBe('test-token')
    expect(document.querySelector('dialog')).toBeNull()
  })

  it('clears a rejected token and asks for a new one on retry', async () => {
    Object.defineProperty(HTMLDialogElement.prototype, 'showModal', { configurable: true, value: vi.fn() })
    Object.defineProperty(HTMLDialogElement.prototype, 'close', { configurable: true, value: vi.fn() })
    setOwnerControlCredential('old-token')
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response('{"error":"unauthorized"}', { status: 401 }))
      .mockResolvedValueOnce(new Response('{"snapshot":{"revision":1,"config":{}}}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await expect(fetchRuntimeControlConfigSnapshot('control://owner')).rejects.toThrow()
    const retry = fetchRuntimeControlConfigSnapshot('control://owner')
    const input = document.querySelector('input')!
    input.value = 'new-token'
    document.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
    await retry
    expect(fetchMock.mock.calls[1]?.[1]?.headers).toMatchObject({ Authorization: 'Bearer new-token' })
  })
})

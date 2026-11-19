import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { isLocalConsole, resolveOpenAIBaseUrl } from '@/app/layout/shell-adapter'
import { env } from '@/lib/env'
import type { StatusPayload } from '@/lib/api/types'

const originalEnv = { ...env }

describe('Console inference endpoint instructions', () => {
  beforeEach(() => {
    env.isDevelopment = false
    env.apiUrl = 'http://127.0.0.1:9337'
  })

  afterEach(() => {
    Object.assign(env, originalEnv)
    vi.unstubAllGlobals()
  })

  it('never advertises a remote Console origin as an inference endpoint', () => {
    vi.stubGlobal('location', new URL('https://console.example.com'))
    expect(resolveOpenAIBaseUrl({ api_port: 9447 } as StatusPayload)).toBe('http://127.0.0.1:9337/v1')
    expect(isLocalConsole()).toBe(false)
  })

  it('keeps an explicitly configured native endpoint', () => {
    vi.stubGlobal('location', new URL('https://console.example.com'))
    env.apiUrl = 'https://inference.example.com/v1/'
    expect(resolveOpenAIBaseUrl()).toBe('https://inference.example.com/v1')
  })

  it('uses the native listener port for a local Console', () => {
    vi.stubGlobal('location', new URL('http://localhost:3131'))
    expect(resolveOpenAIBaseUrl({ api_port: 9447 } as StatusPayload)).toBe('http://127.0.0.1:9447/v1')
    expect(isLocalConsole()).toBe(true)
  })
})

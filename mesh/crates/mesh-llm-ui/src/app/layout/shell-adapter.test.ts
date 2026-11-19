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

  it.each(['https://console.example.com', 'http://[::1]:3131'])(
    'does not substitute the Console origin or status port at %s',
    (origin) => {
      vi.stubGlobal('location', new URL(origin))
      expect(resolveOpenAIBaseUrl({ api_port: 9447 } as StatusPayload)).toBe('http://127.0.0.1:9337/v1')
      expect(isLocalConsole()).toBe(false)
    }
  )

  it.each(['https://console.example.com', 'http://[::1]:3131'])(
    'keeps an explicitly configured native endpoint at %s',
    (origin) => {
      vi.stubGlobal('location', new URL(origin))
      env.apiUrl = 'https://inference.example.com/v1/'
      expect(resolveOpenAIBaseUrl({ api_port: 9447 } as StatusPayload)).toBe('https://inference.example.com/v1')
    }
  )

  it('uses the native listener port for a local Console', () => {
    vi.stubGlobal('location', new URL('http://localhost:3131'))
    expect(resolveOpenAIBaseUrl({ api_port: 9447 } as StatusPayload)).toBe('http://127.0.0.1:9447/v1')
    expect(isLocalConsole()).toBe(true)
  })
})

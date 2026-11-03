import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ReactNode } from 'react'
import { useModelPaymentsQuery } from './use-model-payments-query'
import { usePaidRoutingQuery } from './use-paid-routing-query'

const clients: QueryClient[] = []
afterEach(() => {
  clients.forEach((client) => client.clear())
  clients.length = 0
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } } })
  clients.push(client)
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
}

describe('payment query refresh', () => {
  it('refreshes model availability and prices across discovery pages without remounting', async () => {
    const paidOnly = (outputMsatPerMillion: number, next?: string) =>
      new Response(
        JSON.stringify({
          data: [
            {
              id: 'model',
              providers: [
                {
                  provider_id: 'peer-a',
                  local: false,
                  available: true,
                  payment: {
                    binding_quote: false,
                    paid: true,
                    pricing: { input_msat_per_million: 1000, output_msat_per_million: outputMsatPerMillion },
                    rate_unit: 'msat_per_million_tokens'
                  }
                }
              ]
            }
          ],
          ...(next ? { next_cursor: next } : {})
        })
      )
    const mixedPage = (next?: string) =>
      new Response(
        JSON.stringify({
          data: [
            {
              id: 'model',
              providers: [
                {
                  provider_id: 'peer-a',
                  local: false,
                  available: true,
                  payment: {
                    binding_quote: false,
                    paid: true,
                    pricing: { input_msat_per_million: 1000, output_msat_per_million: 2000 },
                    rate_unit: 'msat_per_million_tokens'
                  }
                },
                {
                  provider_id: 'peer-b',
                  local: false,
                  available: true,
                  payment: { binding_quote: false, paid: false, pricing: null, rate_unit: 'msat_per_million_tokens' }
                }
              ]
            }
          ],
          ...(next ? { next_cursor: next } : {})
        })
      )
    const secondPage = new Response(
      JSON.stringify({
        data: [
          {
            id: 'other',
            providers: [
              {
                provider_id: 'peer-c',
                local: false,
                available: true,
                payment: { binding_quote: false, paid: false, pricing: null, rate_unit: 'msat_per_million_tokens' }
              }
            ]
          }
        ]
      })
    )
    const fetchMock = vi
      .fn()
      // Initial refresh: a single page listing the model as paid-only.
      .mockResolvedValueOnce(paidOnly(1500))
      // Interval refresh: the model gains a free provider, and a second model
      // moves onto page two of the listing.
      .mockResolvedValueOnce(mixedPage('snap:1'))
      .mockResolvedValueOnce(secondPage)
    vi.stubGlobal('fetch', fetchMock)
    const interval = vi.spyOn(globalThis, 'setInterval')
    const { result } = renderHook(() => useModelPaymentsQuery(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data?.get('model')?.freeAvailable).toBe(false))
    const refresh = interval.mock.calls.find((call) => call[1] === 60_000)?.[0]
    expect(refresh).toBeTypeOf('function')
    await act(async () => {
      ;(refresh as () => void)()
    })
    await waitFor(() =>
      expect(result.current.data?.get('model')).toMatchObject({ freeAvailable: true, outputMsatPerMillion: 2000 })
    )
    // The second page is requested through the advertised cursor and merged in.
    expect(fetchMock.mock.calls[2][0]).toContain('cursor=snap%3A1')
    expect(result.current.data?.has('other')).toBe(true)
  })

  it('refreshes free-only policy to automatic without remounting', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(new Response(JSON.stringify({ mode: 'free_only' })))
        .mockResolvedValue(new Response(JSON.stringify({ mode: 'automatic' })))
    )
    const interval = vi.spyOn(globalThis, 'setInterval')
    const { result } = renderHook(() => usePaidRoutingQuery(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toBe(false))
    const refresh = interval.mock.calls.find((call) => call[1] === 60_000)?.[0]
    expect(refresh).toBeTypeOf('function')
    await act(async () => {
      ;(refresh as () => void)()
    })
    await waitFor(() => expect(result.current.data).toBe(true))
  })
})

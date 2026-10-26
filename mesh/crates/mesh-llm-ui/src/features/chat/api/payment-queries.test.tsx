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
  it('refreshes model availability and prices without remounting', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            data: [
              {
                id: 'model',
                payment: {
                  free_available: false,
                  paid_available: true,
                  offers: [{ paid: true, pricing: { output_msat_per_million: 1500 } }]
                }
              }
            ]
          })
        )
      )
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            data: [
              {
                id: 'model',
                payment: {
                  free_available: true,
                  paid_available: true,
                  offers: [{ paid: true, pricing: { output_msat_per_million: 2000 } }]
                }
              }
            ]
          })
        )
      )
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

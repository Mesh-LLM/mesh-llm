import '@testing-library/jest-dom/vitest'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { MeshPluginUiContributionMountContext } from '@/features/plugins/web-ui/host-contract'
import type { PluginSummaryRaw, PluginWebUiStateRaw } from '@/lib/api/plugin-types'

const bundle = vi.hoisted(() => ({
  importBundle: vi.fn(),
  mount: vi.fn(),
  unmount: vi.fn()
}))

vi.mock('@/features/plugins/web-ui/bundle-loader', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/features/plugins/web-ui/bundle-loader')>()),
  importPluginUiBundle: bundle.importBundle
}))

import { PluginContributionSlot, PluginContributionsProvider } from '@/features/plugins/web-ui/PluginContributionSlot'

function readyWebUi(overrides: Partial<PluginWebUiStateRaw> = {}): PluginWebUiStateRaw {
  return {
    state: 'ready',
    declared: true,
    enabled: true,
    available: true,
    primary_tab_enabled: false,
    asset_base_url: '/api/plugins/notes/web-ui/assets/',
    contributions: [
      { id: 'chat-note', slot: 'chat_message', label: 'Note', bundle_id: 'main', entry_script: 'note.js' },
      { id: 'logs-link', slot: 'logs_request', label: 'Related', bundle_id: 'main', entry_script: 'note.js' }
    ],
    ...overrides
  }
}

function summary(webUi: PluginWebUiStateRaw): PluginSummaryRaw {
  return { name: 'notes', kind: 'external', enabled: true, status: 'running', web_ui: webUi }
}

function renderSlot(summaries: readonly PluginSummaryRaw[], requestId = 'req-1') {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  queryClient.setQueryData(['plugins', 'web-ui-config', 'notes'], { plugin: 'notes', settings: {} })
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ plugin: 'notes', settings: {} }), { status: 200 }))
  )
  const ui = (id: string) => (
    <QueryClientProvider client={queryClient}>
      <PluginContributionsProvider summaries={summaries}>
        <PluginContributionSlot subject={{ slot: 'logs_request', requestId: id, exchangeId: 'exch-1' }} />
      </PluginContributionsProvider>
    </QueryClientProvider>
  )
  const view = render(ui(requestId))
  return { ...view, rerenderWith: (id: string) => view.rerender(ui(id)) }
}

describe('PluginContributionSlot', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    bundle.importBundle.mockReset()
    bundle.mount.mockReset()
    bundle.unmount.mockReset()
  })

  it('renders nothing without a provider', () => {
    const { container } = render(<PluginContributionSlot subject={{ slot: 'logs_request', requestId: 'req-1' }} />)

    expect(container).toBeEmptyDOMElement()
  })

  it.each([
    ['disabled', readyWebUi({ state: 'disabled', enabled: false, available: false })],
    ['not running', readyWebUi({ state: 'plugin_not_running', available: false })],
    ['no contribution for the slot', readyWebUi({ contributions: [] })]
  ])('renders nothing when the plugin is %s', (_label, webUi) => {
    const { container } = renderSlot([summary(webUi)])

    expect(container).toBeEmptyDOMElement()
    expect(bundle.importBundle).not.toHaveBeenCalled()
  })

  it('mounts the slot contribution with host ids only, and remounts when an id changes', async () => {
    bundle.mount.mockImplementation(({ element }: MeshPluginUiContributionMountContext) => {
      element.textContent = 'related page'
      return { unmount: bundle.unmount }
    })
    bundle.importBundle.mockResolvedValue({
      registerMeshPluginUi: () => ({ pages: {}, contributions: { 'logs-link': bundle.mount } })
    })

    const { container, rerenderWith } = renderSlot([summary(readyWebUi())])

    await waitFor(() => expect(bundle.mount).toHaveBeenCalledTimes(1))
    const context = bundle.mount.mock.calls[0][0] as MeshPluginUiContributionMountContext
    expect(context.subject).toEqual({ slot: 'logs_request', requestId: 'req-1', exchangeId: 'exch-1' })
    expect(context.contribution.id).toBe('logs-link')
    expect(String(bundle.importBundle.mock.calls[0][0])).toContain('/api/plugins/notes/web-ui/assets/note.js')
    expect(container.querySelector('[data-plugin-contribution="notes:logs-link"]')).toHaveTextContent('related page')

    rerenderWith('req-2')

    await waitFor(() => expect(bundle.mount).toHaveBeenCalledTimes(2))
    expect(bundle.unmount).toHaveBeenCalledTimes(1)
    expect((bundle.mount.mock.calls[1][0] as MeshPluginUiContributionMountContext).subject).toMatchObject({
      requestId: 'req-2'
    })
  })

  it('leaves the slot empty when the bundle fails to load', async () => {
    bundle.importBundle.mockRejectedValue(new Error('bundle missing'))

    const { container } = renderSlot([summary(readyWebUi())])

    await waitFor(() => expect(bundle.importBundle).toHaveBeenCalled())
    await waitFor(() =>
      expect(container.querySelector('[data-plugin-contribution="notes:logs-link"]')).toBeEmptyDOMElement()
    )
  })

  it('removes partial output when a mount rejects', async () => {
    bundle.mount.mockImplementation(async ({ element }: MeshPluginUiContributionMountContext) => {
      element.textContent = 'partial'
      throw new Error('mount failed')
    })
    bundle.importBundle.mockResolvedValue({
      registerMeshPluginUi: () => ({ pages: {}, contributions: { 'logs-link': bundle.mount } })
    })

    const { container } = renderSlot([summary(readyWebUi())])

    await waitFor(() => expect(bundle.mount).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(container.querySelector('[data-plugin-contribution="notes:logs-link"]')).toBeEmptyDOMElement()
    )
  })

  it.each([
    ['resolves', (settle: Deferred) => settle.resolve({ unmount: bundle.unmount }), 1],
    ['rejects', (settle: Deferred) => settle.reject(new Error('mount failed')), 0]
  ])('keeps the replacement intact when a pending mount %s late', async (_label, settleFirst, staleUnmounts) => {
    const first = deferred()
    const unmountSecond = vi.fn()
    bundle.mount
      .mockImplementationOnce(({ element }: MeshPluginUiContributionMountContext) => {
        element.textContent = 'stale req-1'
        bundle.unmount.mockImplementation(() => {
          element.textContent = ''
        })
        return first.promise
      })
      .mockImplementationOnce(({ element }: MeshPluginUiContributionMountContext) => {
        element.textContent = 'req-2'
        return { unmount: unmountSecond }
      })
    bundle.importBundle.mockResolvedValue({
      registerMeshPluginUi: () => ({ pages: {}, contributions: { 'logs-link': bundle.mount } })
    })

    const { container, rerenderWith } = renderSlot([summary(readyWebUi())])
    const slot = () => container.querySelector('[data-plugin-contribution="notes:logs-link"]')

    await waitFor(() => expect(bundle.mount).toHaveBeenCalledTimes(1))
    rerenderWith('req-2')
    await waitFor(() => expect(bundle.mount).toHaveBeenCalledTimes(2))
    expect(slot()).toHaveTextContent(/^req-2$/)

    settleFirst(first)
    await first.promise.catch(() => undefined)
    await Promise.resolve()

    expect(slot()).toHaveTextContent(/^req-2$/)
    expect(slot()?.childElementCount).toBe(1)
    expect(unmountSecond).not.toHaveBeenCalled()
    const firstElement = (bundle.mount.mock.calls[0][0] as MeshPluginUiContributionMountContext).element
    expect(firstElement.isConnected).toBe(false)
    await waitFor(() => expect(bundle.unmount).toHaveBeenCalledTimes(staleUnmounts))
  })
})

type Deferred = {
  readonly promise: Promise<{ unmount: () => void }>
  readonly resolve: (handle: { unmount: () => void }) => void
  readonly reject: (error: Error) => void
}

function deferred(): Deferred {
  let resolve!: Deferred['resolve']
  let reject!: Deferred['reject']
  const promise = new Promise<{ unmount: () => void }>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

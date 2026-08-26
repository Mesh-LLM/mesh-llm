// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { act, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LogRequestLifecycleStrip } from '@/features/logs/components/LogRequestLifecycleStrip'
import { event } from '@/features/logs/components/LogRequestLifecycleStrip.test-fixtures'

function widenViewport(width: number) {
  vi.spyOn(HTMLDivElement.prototype, 'getBoundingClientRect').mockReturnValue({ width } as DOMRect)
}

function controlTrackResize(): (width: number) => void {
  let callback: ResizeObserverCallback | undefined
  let observer: ResizeObserver | undefined
  vi.spyOn(globalThis, 'ResizeObserver').mockImplementation(function ControlledResizeObserver(nextCallback) {
    callback = nextCallback
    observer = {
      disconnect: vi.fn(),
      observe: vi.fn(),
      unobserve: vi.fn()
    }
    return observer
  })
  return (width) => {
    if (callback === undefined || observer === undefined) throw new Error('Lifecycle track was not observed')
    callback([{ contentRect: { width } } as ResizeObserverEntry], observer)
  }
}

function lifecycleEvents(count: number) {
  return Array.from({ length: count }, (_, index) =>
    event(
      index + 1,
      `2026-08-20T01:${String(53 + Math.floor(index / 60)).padStart(2, '0')}:${String(index % 60).padStart(2, '0')}.112Z`,
      index % 2 === 0 ? 'admitted' : 'route_selected'
    )
  )
}

function expectNeutralPageEdge(edge: Element | null) {
  expect(edge).toHaveClass('h-0.5', 'bg-fg-dim')
  expect(edge).not.toHaveClass('h-px')
  expect(edge).not.toHaveClass(/color-mix/)
}

describe('LogRequestLifecycleStrip', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('offers no pager while every node fits the measured track', () => {
    // Given a track wide enough for all three nodes
    widenViewport(600)
    render(
      <LogRequestLifecycleStrip
        events={[
          event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
          event(2, '2026-08-20T01:53:51.127Z', 'route_selected'),
          event(3, '2026-08-20T01:53:51.169Z', 'completed')
        ]}
      />
    )

    // Then the strip renders in one page with no paging affordance
    const list = screen.getByRole('list', { name: 'Lifecycle events' })
    expect(within(list).getAllByRole('listitem')).toHaveLength(3)
    expect(list.querySelector('[data-lifecycle-edge="incoming"]')).not.toBeInTheDocument()
    expect(list.querySelector('[data-lifecycle-edge="outgoing"]')).not.toBeInTheDocument()
    expect(screen.queryByRole('radiogroup', { name: 'Lifecycle timeline pages' })).not.toBeInTheDocument()
  })

  it('caps a wide track at six nodes per page', () => {
    // Given a desktop-width track with room for far more than six nodes
    widenViewport(1200)
    render(
      <LogRequestLifecycleStrip
        events={[
          event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
          event(2, '2026-08-20T01:53:51.127Z', 'route_selected'),
          event(3, '2026-08-20T01:53:51.169Z', 'attempt_started'),
          event(4, '2026-08-20T01:53:51.557Z', 'stream_started'),
          event(5, '2026-08-20T01:53:51.822Z', 'backend_stream_first_item'),
          event(6, '2026-08-20T01:53:52.100Z', 'stream_chunk'),
          event(7, '2026-08-20T01:53:52.400Z', 'stream_completed'),
          event(8, '2026-08-20T01:54:51.843Z', 'completed')
        ]}
      />
    )

    // Then the page holds the desktop maximum across two pages
    expect(within(screen.getByRole('list', { name: 'Lifecycle events' })).getAllByRole('listitem')).toHaveLength(6)
    expect(screen.getAllByRole('radio')).toHaveLength(2)
  })

  it('caps a narrow track at three nodes per page', () => {
    // Given a mobile-width track that could otherwise seat four nodes
    widenViewport(500)
    render(
      <LogRequestLifecycleStrip
        events={[
          event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
          event(2, '2026-08-20T01:53:51.127Z', 'route_selected'),
          event(3, '2026-08-20T01:53:51.169Z', 'attempt_started'),
          event(4, '2026-08-20T01:53:51.557Z', 'stream_started'),
          event(5, '2026-08-20T01:54:51.843Z', 'completed')
        ]}
      />
    )

    // Then the page holds the mobile maximum
    expect(within(screen.getByRole('list', { name: 'Lifecycle events' })).getAllByRole('listitem')).toHaveLength(3)
    expect(screen.getAllByRole('radio')).toHaveLength(2)
  })

  it('bounds a single node when narrow capacity is exactly one', () => {
    // Given a narrow ResizeObserver measurement that can seat exactly one node
    widenViewport(116)
    render(<LogRequestLifecycleStrip events={[event(1, '2026-08-20T01:53:51.112Z', 'admitted')]} />)

    // Then the one-node page remains centered within its bounded track
    expect(screen.getByRole('list', { name: 'Lifecycle events' })).toHaveStyle({ maxWidth: '116px' })
  })

  it('shows a page-boundary interval once as a continuation on the next page', async () => {
    // Given four milestones split into two pages with one interval crossing the boundary
    const user = userEvent.setup()
    widenViewport(240)
    render(
      <LogRequestLifecycleStrip
        events={[
          event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
          event(2, '2026-08-20T01:53:51.912Z', 'route_selected'),
          event(3, '2026-08-20T01:53:52.912Z', 'attempt_started'),
          event(4, '2026-08-20T01:53:53.112Z', 'completed')
        ]}
      />
    )
    expect(screen.getAllByRole('separator')).toHaveLength(1)
    expect(screen.getByRole('separator', { name: 'Elapsed +800ms' })).toHaveTextContent('+800ms')
    expect(screen.queryByLabelText('Elapsed +1s from previous lifecycle page')).not.toBeInTheDocument()

    // When the reader advances to the second page
    await user.click(screen.getByRole('button', { name: 'Later lifecycle events' }))

    // Then the crossing interval leads the page once and the in-page interval remains once
    const continuation = screen.getByRole('separator', {
      name: 'Elapsed +1s from previous lifecycle page'
    })
    expect(screen.getAllByRole('separator')).toHaveLength(2)
    expect(continuation).toHaveClass('sr-only')
    expect(continuation).toHaveTextContent('+1s')
    expect(screen.queryByText(/continued/i)).not.toBeInTheDocument()
    expect(continuation).not.toHaveAttribute('aria-live')
    expect(screen.getAllByLabelText('Elapsed +1s from previous lifecycle page')).toHaveLength(1)
    expect(screen.getAllByLabelText('Elapsed +200ms')).toHaveLength(1)
    expect(screen.queryByLabelText('Elapsed +800ms')).not.toBeInTheDocument()
  })

  it('omits a page-boundary continuation when the interval is unavailable', async () => {
    // Given an invalid timestamp on the first milestone of the second page
    const user = userEvent.setup()
    widenViewport(240)
    render(
      <LogRequestLifecycleStrip
        events={[
          event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
          event(2, '2026-08-20T01:53:51.912Z', 'route_selected'),
          event(3, 'invalid', 'attempt_started'),
          event(4, '2026-08-20T01:53:53.112Z', 'completed')
        ]}
      />
    )

    // When the reader advances across the unavailable interval
    await user.click(screen.getByRole('button', { name: 'Later lifecycle events' }))

    // Then no continuation or invalid elapsed metadata is exposed
    expect(screen.queryByRole('separator')).not.toBeInTheDocument()
    expect(screen.queryByText(/continued/)).not.toBeInTheDocument()
  })

  it('returns a later page to the full timeline when the track widens', async () => {
    // Given the reader is on the second page of a narrow lifecycle track
    const user = userEvent.setup()
    const resizeTrack = controlTrackResize()
    widenViewport(240)
    render(
      <LogRequestLifecycleStrip
        events={[
          event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
          event(2, '2026-08-20T01:53:51.912Z', 'route_selected'),
          event(3, '2026-08-20T01:53:52.912Z', 'attempt_started'),
          event(4, '2026-08-20T01:53:53.112Z', 'completed')
        ]}
      />
    )
    await user.click(screen.getByRole('button', { name: 'Later lifecycle events' }))
    expect(screen.getByLabelText('Elapsed +1s from previous lifecycle page')).toBeInTheDocument()

    // When the measured track widens enough for the entire lifecycle
    act(() => resizeTrack(1200))

    // Then the active page clamps to the full timeline and ordinary intervals return
    const list = screen.getByRole('list', { name: 'Lifecycle events' })
    expect(within(list).getAllByRole('listitem')).toHaveLength(4)
    expect(screen.queryByRole('radiogroup', { name: 'Lifecycle timeline pages' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/previous lifecycle page/)).not.toBeInTheDocument()
    expect(
      within(list)
        .getAllByRole('separator')
        .map((separator) => separator.getAttribute('aria-label'))
    ).toEqual(['Elapsed +800ms', 'Elapsed +1s', 'Elapsed +200ms'])
  })

  it('pages a long lifecycle instead of wrapping it', async () => {
    // Given more nodes than the measured track can seat
    const user = userEvent.setup()
    widenViewport(240)
    render(
      <LogRequestLifecycleStrip
        events={[
          event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
          event(2, '2026-08-20T01:53:51.127Z', 'route_selected'),
          event(3, '2026-08-20T01:53:51.169Z', 'attempt_started'),
          event(4, '2026-08-20T01:53:51.557Z', 'stream_started'),
          event(5, '2026-08-20T01:53:51.822Z', 'stream_completed'),
          event(6, '2026-08-20T01:54:51.843Z', 'completed')
        ]}
      />
    )

    // Then only the first page is mounted, behind a three-page control
    const list = screen.getByRole('list', { name: 'Lifecycle events' })
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((node) => node.getAttribute('data-event-kind'))
    ).toEqual(['admitted', 'route_selected'])
    expect(screen.getAllByRole('radio')).toHaveLength(3)
    expect(list.querySelector('[data-lifecycle-edge="incoming"]')).not.toBeInTheDocument()
    expectNeutralPageEdge(list.querySelector('[data-lifecycle-edge="outgoing"]'))
    expect(Array.from(list.children).every((child) => child.tagName === 'LI')).toBe(true)

    // When the reader advances a page
    await user.click(screen.getByRole('button', { name: 'Later lifecycle events' }))

    // Then the next pair of nodes replaces the first
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((node) => node.getAttribute('data-event-kind'))
    ).toEqual(['attempt_started', 'stream_started'])
    expectNeutralPageEdge(list.querySelector('[data-lifecycle-edge="incoming"]'))
    expectNeutralPageEdge(list.querySelector('[data-lifecycle-edge="outgoing"]'))

    // When a page is chosen directly from its dot
    await user.click(screen.getByRole('radio', { name: 'Lifecycle events page 3 of 3' }))

    // Then the terminal nodes render
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((node) => node.getAttribute('data-event-kind'))
    ).toEqual(['stream_completed', 'completed'])
    expectNeutralPageEdge(list.querySelector('[data-lifecycle-edge="incoming"]'))
    expect(list.querySelector('[data-lifecycle-edge="outgoing"]')).not.toBeInTheDocument()
  })

  it('keeps a 125-page narrow lifecycle pager bounded and clamps after widening', async () => {
    // Given the maximum retained lifecycle with two nodes per narrow page
    const user = userEvent.setup()
    const resizeViewport = controlTrackResize()
    widenViewport(240)
    render(<LogRequestLifecycleStrip events={lifecycleEvents(250)} />)

    // Then the narrow pager exposes bounded choices without horizontal overflow
    const viewport = screen.getByTestId('lifecycle-viewport')
    expect(screen.getAllByRole('radio')).toHaveLength(7)
    expect(screen.getByRole('radio', { name: 'Lifecycle events page 1 of 125' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Lifecycle events page 125 of 125' })).toBeInTheDocument()
    expect(viewport.parentElement?.className).not.toMatch(/overflow/)

    // When the reader selects the last page and the viewport widens to desktop
    await user.click(screen.getByRole('radio', { name: 'Lifecycle events page 125 of 125' }))
    act(() => resizeViewport(1200))

    // Then the page clamps to the 42-page desktop timeline and remains bounded
    expect(screen.getAllByRole('radio')).toHaveLength(7)
    expect(screen.getByRole('radio', { name: 'Lifecycle events page 42 of 42' })).toBeChecked()
    expect(screen.getByRole('button', { name: 'Later lifecycle events' })).toBeDisabled()
  })
})

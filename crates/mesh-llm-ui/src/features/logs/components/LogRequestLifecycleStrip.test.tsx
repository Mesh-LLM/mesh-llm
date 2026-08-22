// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogRequestLifecycleStrip } from '@/features/logs/components/LogRequestLifecycleStrip'
import { event } from '@/features/logs/components/LogRequestLifecycleStrip.test-fixtures'
import { formatClockTime } from '@/features/logs/components/LogRequestOverviewDerivations'

describe('LogRequestLifecycleStrip', () => {
  const sparseKinds = ['admitted', 'route_selected', 'attempt_started', 'stream_started', 'stream_chunk'] as const

  it('keeps the timeline inset on the track while leaving the Pager at full width', () => {
    // Given a lifecycle strip with a paging affordance
    const events = Array.from({ length: 8 }, (_, index) =>
      event(index + 1, `2026-08-20T01:53:5${index}.112Z`, index % 2 === 0 ? 'admitted' : 'route_selected')
    )
    render(<LogRequestLifecycleStrip events={events} />)

    // Then only the timeline track owns the panel inset and Pager shares the unpadded wrapper
    const viewport = screen.getByTestId('lifecycle-viewport')
    const wrapper = viewport.parentElement
    const pager = screen.getByRole('radiogroup').parentElement
    expect(viewport).toHaveClass('px-[var(--panel-x)]')
    expect(wrapper).toHaveClass('py-5')
    expect(wrapper).not.toHaveClass('px-[var(--panel-x)]')
    expect(pager?.parentElement).toBe(wrapper)
  })

  it.each([
    { count: 1, maxWidth: '116px' },
    { count: 2, maxWidth: '232px' },
    { count: 3, maxWidth: '348px' },
    { count: 5, maxWidth: '580px' }
  ])('centers and bounds a sparse $count-node page', ({ count, maxWidth }) => {
    // Given a wide viewport with only a sparse visible lifecycle page
    const events = Array.from({ length: count }, (_, index) =>
      event(index + 1, `2026-08-20T01:53:5${index}.112Z`, sparseKinds[index] ?? 'admitted')
    )

    // When the lifecycle strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then the measured viewport stays full-width while the visible track is centered and bounded
    const viewport = screen.getByTestId('lifecycle-viewport')
    const list = screen.getByRole('list', { name: 'Lifecycle events' })
    expect(viewport).toHaveClass('min-w-0')
    expect(list).toHaveClass('mx-auto')
    expect(list).toHaveStyle({ maxWidth })
    expect(list.style.paddingInlineStart).toBe('')
  })

  it('lets a full visible page use the available viewport width', () => {
    // Given a wide viewport containing a complete six-node page
    const kinds = [
      'admitted',
      'route_selected',
      'attempt_started',
      'stream_started',
      'stream_chunk',
      'completed'
    ] as const
    const events = kinds.map((kind, index) => event(index + 1, `2026-08-20T01:53:5${index}.112Z`, kind))

    // When the lifecycle strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then the complete page is not artificially width-capped
    const list = screen.getByRole('list', { name: 'Lifecycle events' })
    expect(list).not.toHaveStyle({ maxWidth: '696px' })
    expect(list.style.paddingInlineStart).toBe('')
  })

  it('does not suppress horizontal overflow on the page container', () => {
    // Given a lifecycle strip with retained events
    render(<LogRequestLifecycleStrip events={[event(1, '2026-08-20T01:53:51.112Z', 'admitted')]} />)

    // Then the page container leaves overflow behavior to the normal layout
    const viewport = screen.getByTestId('lifecycle-viewport')
    expect(viewport.parentElement?.className).not.toMatch(/overflow/)
  })

  it('shows distinct labels for adjacent stream start and chunk milestones', () => {
    // Given a stream start immediately followed by its first chunk
    const events = [
      event(1, '2026-08-20T01:53:51.112Z', 'stream_started'),
      event(2, '2026-08-20T01:53:51.127Z', 'stream_chunk')
    ]

    // When the overview strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then each adjacent milestone keeps its own visible name
    const nodes = within(screen.getByRole('list', { name: 'Lifecycle events' })).getAllByRole('listitem')
    expect(within(nodes[0]).getByText('Stream started')).toBeVisible()
    expect(within(nodes[1]).getByText('Chunks')).toBeVisible()
  })

  it('centers every node icon and text within equal cells', () => {
    // Given a lifecycle page with a connector-bearing node and a terminal node
    render(
      <LogRequestLifecycleStrip
        events={[event(1, '2026-08-20T01:53:51.112Z', 'admitted'), event(2, '2026-08-20T01:53:52.112Z', 'completed')]}
      />
    )

    const nodes = screen.getByRole('list', { name: 'Lifecycle events' }).querySelectorAll('li')
    const firstNode = nodes[0]
    const terminalNode = nodes[1]
    expect(firstNode).toBeDefined()
    expect(terminalNode).toBeDefined()
    if (!firstNode || !terminalNode) return

    // Then connector geometry remains unchanged while only the terminal cell centers its content
    const firstRow = firstNode.firstElementChild
    const terminalRow = terminalNode.firstElementChild
    const firstText = firstNode.querySelector('p')
    const terminalText = terminalNode.querySelector('p')
    const connectors = screen
      .getByRole('list', { name: 'Lifecycle events' })
      .querySelectorAll('li > div > span.absolute')
    expect(firstRow).toHaveClass('justify-center')
    expect(firstNode.querySelector('.absolute')).toBeInTheDocument()
    expect(terminalRow).toHaveClass('justify-center')
    expect(connectors).toHaveLength(1)
    expect(firstText?.parentElement).toHaveClass('text-center')
    expect(terminalText?.parentElement).toHaveClass('text-center')
    expect(terminalRow?.querySelectorAll('span.absolute')).toHaveLength(0)
  })

  it('does not add a connector to a single node page', () => {
    // Given a lifecycle page with only its terminal node
    render(<LogRequestLifecycleStrip events={[event(1, '2026-08-20T01:53:51.112Z', 'completed')]} />)

    // Then the centered terminal row has no connector extension or spacer
    const terminalRow = screen.getByRole('listitem').firstElementChild
    expect(terminalRow?.querySelectorAll('span.absolute')).toHaveLength(0)
  })

  it('keeps connector tone transitions across a multi-node page', () => {
    // Given adjacent lifecycle segments with different connector tones
    render(
      <LogRequestLifecycleStrip
        events={[
          event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
          event(2, '2026-08-20T01:53:52.112Z', 'stream_started'),
          event(3, '2026-08-20T01:53:53.112Z', 'completed')
        ]}
      />
    )

    // Then each visible segment keeps its own tone and the terminal lead uses the previous tone
    const nodes = screen.getByRole('list', { name: 'Lifecycle events' }).querySelectorAll('li')
    const firstConnector = nodes[0]?.querySelector('span.absolute')
    const secondConnector = nodes[1]?.querySelector('span.absolute')
    const terminalConnectors = nodes[2]?.querySelectorAll('span.absolute')
    expect(firstConnector).toHaveClass('bg-[color:color-mix(in_oklab,var(--color-good)_55%,transparent)]')
    expect(secondConnector).toHaveClass('bg-[color:color-mix(in_oklab,var(--color-accent)_55%,transparent)]')
    expect(terminalConnectors).toHaveLength(0)
  })

  it('collapses consecutive chunks into one visible counted node', () => {
    // Given a stream that emitted many chunks between two milestones
    const events = [
      event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
      event(2, '2026-08-20T01:53:51.127Z', 'stream_chunk'),
      event(3, '2026-08-20T01:53:51.169Z', 'stream_chunk'),
      event(4, '2026-08-20T01:53:51.557Z', 'stream_chunk'),
      event(5, '2026-08-20T01:54:51.843Z', 'completed')
    ]

    // When the overview strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then the chunk run occupies one node with a visual-only multiplicity
    const nodes = within(screen.getByRole('list', { name: 'Lifecycle events' })).getAllByRole('listitem')
    expect(nodes.map((node) => node.getAttribute('data-event-kind'))).toEqual(['admitted', 'stream_chunk', 'completed'])
    expect(nodes[1]).toHaveTextContent('Chunks ×3')
    expect(nodes[1].querySelector('p > span[aria-hidden="true"]')).toHaveTextContent('×3')
  })

  it('announces a chunk aggregate once with its count and final occurrence time', () => {
    // Given three consecutive chunks with distinct occurrence times
    const lastOccurredAt = '2026-08-20T01:53:51.557Z'
    const events = [
      event(1, '2026-08-20T01:53:51.127Z', 'stream_chunk'),
      event(2, '2026-08-20T01:53:51.169Z', 'stream_chunk'),
      event(3, lastOccurredAt, 'stream_chunk')
    ]

    // When the overview strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then one accessible paragraph uses the aggregate's final occurrence while the visible copy stays compact
    const expansion = `3 chunks, last at ${formatClockTime(lastOccurredAt)}`
    const chunkLabel = within(screen.getByRole('list', { name: 'Lifecycle events' }))
      .getAllByRole('listitem')[0]
      ?.querySelector('p')
    expect(chunkLabel).toHaveAttribute('aria-label', expansion)
    expect(chunkLabel?.querySelector('[aria-hidden="true"]')).toHaveTextContent('×3')
    expect(screen.queryByText(expansion)).not.toBeInTheDocument()
    expect(screen.queryByTitle('Chunks')).not.toBeInTheDocument()
  })

  it('announces the lifecycle page through one polite status without adding focus targets', () => {
    // Given enough lifecycle nodes to create multiple pages
    render(
      <LogRequestLifecycleStrip
        events={Array.from({ length: 8 }, (_, index) =>
          event(index + 1, `2026-08-20T01:53:5${index}.112Z`, index % 2 === 0 ? 'admitted' : 'route_selected')
        )}
      />
    )

    // Then the current page is announced without exposing a separate focusable control
    const status = screen.getByRole('status')
    expect(status).toHaveTextContent('Page 1 of 2')
    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(status).not.toHaveAttribute('tabindex')
  })

  it('orders events by instant and labels each node with a short milestone name', () => {
    // Given out-of-order retained events
    const events = [
      event(2, '2026-08-20T01:53:51.127Z', 'route_selected'),
      event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
      event(3, '2026-08-20T01:53:51.169Z', 'attempt_started')
    ]

    // When the overview strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then nodes read in chronological order using human labels, not event kinds
    const list = screen.getByRole('list', { name: 'Lifecycle events' })
    expect(list.textContent).toMatch(/Received[\s\S]*Routed[\s\S]*Connected/)
    expect(list.textContent).not.toContain('route_selected')
  })

  it('uses warning tones for rejected and cancelled outcomes while failures remain bad', () => {
    // Given warning terminal outcomes followed by actual failures
    const events = [
      event(1, '2026-08-20T01:53:51.112Z', 'rejected'),
      event(2, '2026-08-20T01:53:51.127Z', 'cancelled'),
      event(3, '2026-08-20T01:53:51.169Z', 'failed'),
      event(4, '2026-08-20T01:53:51.557Z', 'stream_error')
    ]

    // When the overview strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then needs-attention outcomes are warning-toned and genuine errors remain bad
    const nodes = within(screen.getByRole('list', { name: 'Lifecycle events' })).getAllByRole('listitem')
    expect(nodes.map((node) => node.querySelector('[data-event-tone]')?.getAttribute('data-event-tone'))).toEqual([
      'warn',
      'warn',
      'bad',
      'bad'
    ])
  })

  it('uses theme-safe semantic text tokens for status nodes', () => {
    // Given good, warning, and bad lifecycle states
    const events = [
      event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
      event(2, '2026-08-20T01:53:51.127Z', 'rejected'),
      event(3, '2026-08-20T01:53:51.169Z', 'failed')
    ]

    // When the overview strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then each semantic node color uses its theme-safe text token
    const nodes = within(screen.getByRole('list', { name: 'Lifecycle events' })).getAllByRole('listitem')
    expect(nodes[0].querySelector('[data-event-tone]')).toHaveClass('text-good-text')
    expect(nodes[1].querySelector('[data-event-tone]')).toHaveClass('text-warn-text')
    expect(nodes[2].querySelector('[data-event-tone]')).toHaveClass('text-bad-text')
  })

  it('exposes each elapsed interval through one accessible representation', () => {
    // Given two milestones separated by fifteen milliseconds
    const events = [
      event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
      event(2, '2026-08-20T01:53:51.127Z', 'stream_started')
    ]

    // When the overview strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then the visible abbreviation has one accurate accessible name and no duplicate sr-only copy
    const elapsed = screen.getAllByLabelText('Elapsed +15ms')
    expect(elapsed).toHaveLength(1)
    expect(elapsed[0]).toHaveRole('separator')
    expect(elapsed[0]).not.toHaveAttribute('aria-live')
    expect(elapsed[0]).toHaveTextContent('+15ms')
    expect(screen.queryByText('Elapsed +15ms')).not.toBeInTheDocument()
  })

  it('uses label-floor typography for timestamps and elapsed metadata', () => {
    // Given two milestones with both timestamp and elapsed metadata
    const events = [
      event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
      event(2, '2026-08-20T01:53:51.127Z', 'stream_started')
    ]

    // When the overview strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then every metadata value uses the 11px label floor without label casing or tracking
    const list = screen.getByRole('list', { name: 'Lifecycle events' })
    const metadata = [screen.getByLabelText('Elapsed +15ms'), ...list.querySelectorAll('time')]
    for (const value of metadata) {
      expect(value).toHaveClass('type-label', 'normal-case!', 'tracking-normal!', 'font-mono', 'tabular-nums')
      expect(value.className).not.toContain('--density-type-micro')
    }
  })

  it('preserves phase tones for ordinary lifecycle milestones', () => {
    // Given a request that streamed and then failed
    const events = [
      event(1, '2026-08-20T01:53:51.112Z', 'admitted'),
      event(2, '2026-08-20T01:53:51.127Z', 'stream_started'),
      event(3, '2026-08-20T01:53:51.169Z', 'failed')
    ]

    // When the overview strip renders
    render(<LogRequestLifecycleStrip events={events} />)

    // Then the existing good, accent, and bad phase meanings remain intact
    const nodes = within(screen.getByRole('list', { name: 'Lifecycle events' })).getAllByRole('listitem')
    expect(nodes.map((node) => node.querySelector('[data-event-tone]')?.getAttribute('data-event-tone'))).toEqual([
      'good',
      'accent',
      'bad'
    ])
  })

  it('renders nothing but an empty list when no events are retained', () => {
    // Given no retained lifecycle events
    render(<LogRequestLifecycleStrip events={[]} />)

    // Then the list renders with no nodes
    expect(within(screen.getByRole('list', { name: 'Lifecycle events' })).queryAllByRole('listitem')).toHaveLength(0)
  })
})

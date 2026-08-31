// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { SsePayloadView } from '@/features/logs/components/SsePayloadView'
import { decodeEventStream } from '@/features/logs/lib/log-payload-content'

const HOSTILE_TEXT = '<img src=x onerror=alert(1)><script>alert(2)</script>'
const FRAMES = decodeEventStream(
  [
    'event: delta',
    'id: frame-1',
    'data: {"delta":"hello"}',
    '',
    `data: ${HOSTILE_TEXT}`,
    '',
    'event: done',
    'data: [DONE]',
    ''
  ].join('\n')
)

describe('SSE payload rendering', () => {
  it('renders exactly one selected frame with compact context and response-frame paging', () => {
    // Given / When
    const { container } = render(<SsePayloadView ariaLabel="Response event stream" format="pretty" frames={FRAMES} />)

    // Then
    const frame = screen.getByRole('region', { name: 'Response event stream frame 1' })
    expect(screen.getAllByRole('region', { name: /^Response event stream frame \d$/ })).toHaveLength(1)
    expect(frame).toHaveTextContent('Frame 1 of 3')
    expect(within(frame).getByText('delta')).toBeInTheDocument()
    expect(within(frame).getByText('frame-1')).toBeInTheDocument()
    expect(within(frame).getByText('"delta"')).toHaveAttribute('data-json-token', 'key')
    expect(within(frame).getByRole('button', { name: 'Copy JSON payload' })).toBeInTheDocument()
    const framePager = screen.getByRole('radiogroup', { name: 'Response frames' })
    expect(framePager).toHaveClass('flex-wrap', 'gap-1.5')
    expect(framePager.parentElement).toHaveClass('grid', 'border-t', 'pt-2', 'sm:border-t-0', 'sm:pt-0')
    const frameChoices = screen.getAllByRole('radio')
    expect(frameChoices.map((choice) => choice.textContent)).toEqual(['1', '2', '3'])
    expect(frameChoices[0]).toBeChecked()
    for (const choice of frameChoices) {
      expect(choice).toHaveClass('size-10', 'border')
    }
    expect(frameChoices[0]).toHaveClass('border-accent', 'bg-accent')
    expect(frameChoices[1]).toHaveClass('border-border', 'bg-panel')
    expect(screen.getByRole('button', { name: 'Previous response frame' })).toHaveClass('size-10', 'border')
    expect(screen.getByRole('button', { name: 'Previous response frame' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Next response frame' })).toBeEnabled()
    expect(screen.getAllByRole('status').some((status) => status.textContent === 'Frame 1 of 3')).toBe(true)
    expect(frame.querySelector('header')).toHaveClass('grid', 'gap-2', 'sm:grid-cols-[minmax(0,1fr)_auto]')
    expect(container.querySelector('ol')).toBeNull()
    expect(container.querySelector('li')).toBeNull()
    expect(container.querySelector('[data-radix-scroll-area-viewport]')).toBeNull()
  })

  it('replaces the selected frame while keeping hostile plaintext and DONE data inert', async () => {
    // Given
    const user = userEvent.setup()
    const { container } = render(<SsePayloadView ariaLabel="Response event stream" format="pretty" frames={FRAMES} />)

    // When
    await user.click(screen.getByRole('radio', { name: 'Response frame 2 of 3' }))

    // Then
    const textFrame = screen.getByRole('region', { name: 'Response event stream frame 2' })
    expect(screen.queryByRole('region', { name: 'Response event stream frame 1' })).not.toBeInTheDocument()
    expect(within(textFrame).getByText(HOSTILE_TEXT)).toBeInTheDocument()
    expect(within(textFrame).queryByText('Event')).not.toBeInTheDocument()
    expect(within(textFrame).queryByText('ID')).not.toBeInTheDocument()
    expect(within(textFrame).queryByRole('button', { name: 'Copy JSON payload' })).not.toBeInTheDocument()
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()

    // When
    await user.click(screen.getByRole('button', { name: 'Next response frame' }))

    // Then
    const doneFrame = screen.getByRole('region', { name: 'Response event stream frame 3' })
    expect(screen.queryByText(HOSTILE_TEXT)).not.toBeInTheDocument()
    expect(within(doneFrame).getByText('done')).toBeInTheDocument()
    expect(within(doneFrame).getByText('[DONE]')).toBeInTheDocument()
    expect(within(doneFrame).queryByRole('button', { name: 'Copy JSON payload' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next response frame' })).toBeDisabled()
  })

  it('uses the controlled JSON format without adding a frame-local format control', () => {
    // Given
    const { container, rerender } = render(
      <SsePayloadView ariaLabel="Response event stream" format="raw" frames={FRAMES} />
    )

    // Then
    expect(container.querySelectorAll('[data-json-line]')).toHaveLength(1)
    expect(screen.queryByRole('radiogroup', { name: 'JSON format' })).not.toBeInTheDocument()

    // When
    rerender(<SsePayloadView ariaLabel="Response event stream" format="pretty" frames={FRAMES} />)

    // Then
    expect(container.querySelectorAll('[data-json-line]')).toHaveLength(3)
  })

  it('clamps the selected frame when the available frame count shrinks', async () => {
    // Given
    const user = userEvent.setup()
    const { rerender } = render(<SsePayloadView ariaLabel="Response event stream" format="pretty" frames={FRAMES} />)
    await user.click(screen.getByRole('radio', { name: 'Response frame 3 of 3' }))

    // When
    rerender(<SsePayloadView ariaLabel="Response event stream" format="pretty" frames={FRAMES.slice(0, 1)} />)

    // Then
    expect(screen.getByRole('region', { name: 'Response event stream frame 1' })).toHaveTextContent('Frame 1 of 1')
    expect(screen.queryByRole('radiogroup', { name: 'Response frames' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy JSON payload' })).toBeInTheDocument()
  })
})

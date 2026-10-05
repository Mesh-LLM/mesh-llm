// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { JsonPayloadView } from '@/features/logs/components/JsonPayloadView'

const RAW_JSON = '{"message":"hello","count":2}'
const PRETTY_JSON = '{\n  "message": "hello",\n  "count": 2\n}'

function installClipboard(writeText: (text: string) => Promise<void>) {
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText }
  })
}

function renderedLines(container: HTMLElement): string {
  return Array.from(container.querySelectorAll('[data-json-line]'))
    .map((line) => line.textContent ?? '')
    .join('\n')
}

afterEach(() => {
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })
  vi.restoreAllMocks()
})

describe('JsonPayloadView', () => {
  it('renders the controlled pretty representation with visible line numbers and no local format control', () => {
    // Given / When
    const { container } = render(<JsonPayloadView format="pretty" prettyText={PRETTY_JSON} text={RAW_JSON} />)

    // Then
    expect(renderedLines(container)).toBe(PRETTY_JSON)
    expect(Array.from(container.querySelectorAll('[data-line-number]')).map((line) => line.textContent)).toEqual([
      '1',
      '2',
      '3',
      '4'
    ])
    expect(screen.queryByRole('radiogroup', { name: 'JSON format' })).not.toBeInTheDocument()
  })

  it('updates the representation when the controlled format changes', () => {
    // Given
    const { container, rerender } = render(<JsonPayloadView format="pretty" prettyText={PRETTY_JSON} text={RAW_JSON} />)

    // When
    rerender(<JsonPayloadView format="raw" prettyText={PRETTY_JSON} text={RAW_JSON} />)

    // Then
    expect(renderedLines(container)).toBe(RAW_JSON)
    expect(screen.getByRole('status')).toHaveTextContent('Raw JSON representation selected.')
  })

  it('copies the current representation', async () => {
    // Given
    const user = userEvent.setup()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    installClipboard(writeText)
    render(<JsonPayloadView format="raw" prettyText={PRETTY_JSON} text={RAW_JSON} />)

    // When
    await user.click(screen.getByRole('button', { name: 'Copy JSON payload' }))

    // Then
    expect(writeText).toHaveBeenLastCalledWith(RAW_JSON)
    expect(screen.getByText('Copied')).toBeInTheDocument()
  })

  it('keeps the copy control keyboard reachable', async () => {
    // Given
    const user = userEvent.setup()
    installClipboard(vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined))
    render(<JsonPayloadView format="pretty" prettyText={PRETTY_JSON} text={RAW_JSON} />)

    // When
    await user.tab()

    // Then
    expect(screen.getByRole('button', { name: 'Copy JSON payload' })).toHaveFocus()
  })

  it('anchors the copy toolbar independently from wide JSON content', () => {
    // Given / When
    render(<JsonPayloadView format="raw" prettyText={PRETTY_JSON} text={RAW_JSON} />)

    // Then
    const copy = screen.getByRole('button', { name: 'Copy JSON payload' })
    expect(copy.closest('section')).toHaveClass('min-w-full', 'w-max')
    const toolbar = copy.parentElement
    expect(toolbar).toHaveClass('sticky', 'left-0', 'w-[100cqw]')
    expect(toolbar).not.toHaveClass('min-w-max', 'right-0', 'ml-auto', 'w-fit')
  })

  it('uses distinct syntax tokens while keeping hostile JSON strings inert', () => {
    // Given
    const rawText = '{"markup":"<img src=x onerror=alert(1)>","enabled":true}'
    const prettyText = '{\n  "markup": "<img src=x onerror=alert(1)>",\n  "enabled": true\n}'

    // When
    const { container } = render(<JsonPayloadView format="pretty" prettyText={prettyText} text={rawText} />)

    // Then
    expect(screen.getByText('"markup"')).toHaveAttribute('data-json-token', 'key')
    expect(screen.getByText('"<img src=x onerror=alert(1)>"')).toHaveAttribute('data-json-token', 'string')
    expect(screen.getByText('true')).toHaveAttribute('data-json-token', 'boolean')
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
  })
})

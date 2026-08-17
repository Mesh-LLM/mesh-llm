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
  it('renders the pretty representation with visible line numbers', () => {
    // Given / When
    const { container } = render(<JsonPayloadView prettyText={PRETTY_JSON} text={RAW_JSON} />)

    // Then
    expect(renderedLines(container)).toBe(PRETTY_JSON)
    expect(Array.from(container.querySelectorAll('[data-line-number]')).map((line) => line.textContent)).toEqual([
      '1',
      '2',
      '3',
      '4'
    ])
    expect(screen.getByRole('radio', { name: 'Pretty' })).toBeChecked()
  })

  it('switches between Pretty and Raw without replacing the payload component', async () => {
    // Given
    const user = userEvent.setup()
    const { container } = render(<JsonPayloadView prettyText={PRETTY_JSON} text={RAW_JSON} />)

    // When
    await user.click(screen.getByRole('radio', { name: 'Raw' }))

    // Then
    expect(renderedLines(container)).toBe(RAW_JSON)
    expect(screen.getByRole('radio', { name: 'Raw' })).toBeChecked()
    expect(screen.getByRole('status')).toHaveTextContent('Raw JSON representation selected.')
  })

  it('copies the current representation', async () => {
    // Given
    const user = userEvent.setup()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    installClipboard(writeText)
    render(<JsonPayloadView prettyText={PRETTY_JSON} text={RAW_JSON} />)

    // When
    await user.click(screen.getByRole('button', { name: 'Copy JSON payload' }))

    // Then
    expect(writeText).toHaveBeenLastCalledWith(PRETTY_JSON)

    // When
    await user.click(screen.getByRole('radio', { name: 'Raw' }))
    await user.click(screen.getByRole('button', { name: 'Copy JSON payload' }))

    // Then
    expect(writeText).toHaveBeenLastCalledWith(RAW_JSON)
    expect(screen.getByText('Copied')).toBeInTheDocument()
  })

  it('keeps format and copy controls keyboard reachable', async () => {
    // Given
    const user = userEvent.setup()
    installClipboard(vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined))
    render(<JsonPayloadView prettyText={PRETTY_JSON} text={RAW_JSON} />)
    const pretty = screen.getByRole('radio', { name: 'Pretty' })

    // When
    await user.tab()

    // Then
    expect(pretty).toHaveFocus()

    // When
    await user.tab()

    // Then
    expect(screen.getByRole('radio', { name: 'Raw' })).toHaveFocus()

    // When
    await user.tab()

    // Then
    expect(screen.getByRole('button', { name: 'Copy JSON payload' })).toHaveFocus()
  })

  it('uses distinct syntax tokens while keeping hostile JSON strings inert', () => {
    // Given
    const rawText = '{"markup":"<img src=x onerror=alert(1)>","enabled":true}'
    const prettyText = '{\n  "markup": "<img src=x onerror=alert(1)>",\n  "enabled": true\n}'

    // When
    const { container } = render(<JsonPayloadView prettyText={prettyText} text={rawText} />)

    // Then
    expect(screen.getByText('"markup"')).toHaveAttribute('data-json-token', 'key')
    expect(screen.getByText('"<img src=x onerror=alert(1)>"')).toHaveAttribute('data-json-token', 'string')
    expect(screen.getByText('true')).toHaveAttribute('data-json-token', 'boolean')
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
  })
})

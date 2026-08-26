// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ScrollArea } from '@/components/ui/scroll-area'

describe('ScrollArea', () => {
  it('renders the vertical scrollbar by default', () => {
    // Given / When
    render(
      <ScrollArea type="always" viewportLabel="Default content">
        Payload
      </ScrollArea>
    )

    // Then
    const viewport = screen.getByRole('region', { name: 'Default content' })
    const scrollArea = viewport.parentElement
    expect(scrollArea?.querySelector('[data-orientation="vertical"]')).toBeInTheDocument()
    expect(scrollArea?.querySelector('[data-orientation="horizontal"]')).not.toBeInTheDocument()
  })

  it('suppresses only the vertical scrollbar when horizontal scrolling is enabled', () => {
    // Given / When
    render(
      <ScrollArea horizontal type="always" vertical={false} viewportLabel="Horizontal payload">
        Payload
      </ScrollArea>
    )

    // Then
    const viewport = screen.getByRole('region', { name: 'Horizontal payload' })
    const scrollArea = viewport.parentElement
    expect(viewport).toHaveAttribute('tabindex', '0')
    expect(viewport).not.toHaveClass('overflow-x-hidden')
    expect(scrollArea?.querySelector('[data-orientation="vertical"]')).not.toBeInTheDocument()
    expect(scrollArea?.querySelector('[data-orientation="horizontal"]')).toBeInTheDocument()
  })
})

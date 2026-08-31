// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogPayloadPane } from '@/features/logs/components/LogPayloadPane'

describe('LogPayloadPane', () => {
  it('keeps one horizontal-only payload viewport in the natural pane flow', () => {
    // Given
    const group = { artifacts: [], primary: undefined }

    // When
    const { container } = render(
      <LogPayloadPane
        displayToolbar={
          <div aria-label="Display controls" role="toolbar">
            Pretty or raw
          </div>
        }
        error={false}
        format="pretty"
        group={group}
        kind="request"
        loading={false}
      />
    )

    // Then
    const pane = screen.getByRole('region', { name: 'Request' })
    const toolbar = screen.getByRole('toolbar', { name: 'Display controls' })
    const viewport = screen.getByRole('region', { name: 'Request payload content' })
    const scrollArea = viewport.parentElement
    const displayRow = toolbar.parentElement
    expect(scrollArea).not.toBeNull()
    expect(displayRow?.previousElementSibling?.tagName).toBe('HEADER')
    expect(displayRow?.nextElementSibling).toBe(scrollArea)
    expect(scrollArea?.querySelector('[data-orientation="horizontal"]')).toBeInTheDocument()
    expect(scrollArea?.querySelector('[data-orientation="vertical"]')).not.toBeInTheDocument()
    expect(scrollArea).toHaveClass(
      '[&>[data-orientation=horizontal]]:bg-border-soft',
      '[&>[data-orientation=horizontal]>div]:bg-fg-dim'
    )
    expect(scrollArea).not.toHaveClass('h-80', 'sm:h-[28rem]', 'lg:h-[32rem]')
    expect(viewport).toHaveClass(
      '[container-type:inline-size]',
      '[scrollbar-gutter:stable]',
      '[&>div]:!block',
      '[&>div]:min-w-full',
      'pb-2.5'
    )
    expect(viewport).not.toHaveClass('h-full', 'min-h-0')
    expect(pane).toContainElement(toolbar)
    expect(container.querySelectorAll('[data-radix-scroll-area-viewport]')).toHaveLength(1)
    expect(viewport.querySelector('[data-radix-scroll-area-viewport]')).toBeNull()
  })
})

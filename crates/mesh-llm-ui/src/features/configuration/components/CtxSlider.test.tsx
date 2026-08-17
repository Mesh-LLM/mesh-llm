import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { CtxSlider } from '@/features/configuration/components/CtxSlider'

describe('CtxSlider', () => {
  it('hides the exact token readout by default', () => {
    render(<CtxSlider maxCtx={262144} onChange={vi.fn()} value={49152} />)

    expect(screen.queryByText('Exact')).not.toBeInTheDocument()
    expect(screen.queryByText('49,152')).not.toBeInTheDocument()
  })

  it('shows the exact token readout when configured', () => {
    render(<CtxSlider exactValueVisibility="shown" maxCtx={262144} onChange={vi.fn()} value={49152} />)

    expect(screen.queryByText(/Exact/)).not.toBeInTheDocument()
    expect(screen.getByText('49,152')).toBeInTheDocument()
    expect(screen.getByText(/tokens/)).toBeInTheDocument()
  })

  it('positions the exact token readout above the slider on the left', () => {
    const { container } = render(
      <CtxSlider
        exactValuePosition="top-left"
        exactValueVisibility="shown"
        maxCtx={262144}
        onChange={vi.fn()}
        value={49152}
      />
    )

    const exactBlock = screen.getByText('49,152').closest('div')
    const slider = screen.getByRole('slider', { name: 'Context' })
    expect(exactBlock).toHaveClass('justify-start')
    expect(container.firstElementChild?.children[1]).toContainElement(exactBlock)
    expect((exactBlock as HTMLElement).compareDocumentPosition(slider) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('positions the exact token readout inline on the right', () => {
    const { container } = render(
      <CtxSlider
        exactValuePosition="inline-right"
        exactValueVisibility="shown"
        maxCtx={262144}
        onChange={vi.fn()}
        value={49152}
      />
    )

    const exactBadge = screen.getByText('49,152').closest('span')?.parentElement
    const inlineRow = exactBadge?.closest('div')
    expect(inlineRow).toHaveClass('items-center')
    expect(container.querySelector('.flex-1')).toBeInTheDocument()
    expect(inlineRow?.lastElementChild).toBe(exactBadge)
  })
})

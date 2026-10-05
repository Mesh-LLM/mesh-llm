import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Slider } from '@/components/ui/Slider'

describe('Slider', () => {
  it('emits string values from the native range input', () => {
    const handleValueChange = vi.fn()

    render(
      <Slider
        ariaLabel="Memory margin"
        max={4}
        min={0}
        name="memory-margin"
        onValueChange={handleValueChange}
        step={0.5}
        unit="GB"
        value="2"
      />
    )

    fireEvent.change(screen.getByRole('slider', { name: 'Memory margin' }), { target: { value: '2.5' } })

    expect(handleValueChange).toHaveBeenCalledWith('2.5')
  })

  it('renders caller-provided value labels including zero', () => {
    render(
      <Slider
        ariaLabel="Draft tokens"
        max={128}
        min={0}
        name="draft-max-tokens"
        onValueChange={vi.fn()}
        value="0"
        valueLabel={0}
      />
    )

    expect(screen.getByText('0')).toBeInTheDocument()
  })

  it('supports labels, units, bottom value placement, alignment, and open or closed boundary labels', () => {
    render(
      <Slider
        ariaLabel="Acceptance threshold"
        formatValue={(value) => Number(value).toFixed(2)}
        label="Draft acceptance"
        lowerBound={{ inclusive: false, value: '0.00' }}
        max={1}
        min={0}
        name="draft-acceptance-threshold"
        onValueChange={vi.fn()}
        step={0.05}
        unit="ratio"
        upperBound={{ inclusive: true, value: '1.00' }}
        value="0.7"
        valueLabelAlign="center"
        valueLabelPlacement="bottom"
      />
    )

    expect(screen.getByText('Draft acceptance')).toBeInTheDocument()
    expect(screen.getByText('0.70')).toHaveClass('font-mono')
    expect(screen.getByText('ratio')).not.toHaveClass('font-mono')
    expect(screen.getByText('0.70').parentElement).toHaveClass('justify-self-center')
    expect(screen.getByText('0.00').parentElement).toHaveTextContent('(0.00')
    expect(screen.getByText('1.00').parentElement).toHaveTextContent('1.00]')
    expect(screen.getByRole('slider', { name: 'Acceptance threshold' })).toHaveAttribute('aria-valuetext', '0.70 ratio')
  })

  it('renders lower and upper schema guidance labels with inclusive boundary markers', () => {
    render(
      <Slider
        ariaLabel="Safety margin"
        lowerBound={{ inclusive: true, value: 'Min 0.0 GB' }}
        max={8}
        min={0}
        name="safety-margin"
        onValueChange={vi.fn()}
        upperBound={{ inclusive: true, value: 'Max 8.0 GB' }}
        unit="GB"
        value="2"
      />
    )

    expect(screen.getByText('Min 0.0 GB').parentElement).toHaveTextContent('[Min 0.0 GB')
    expect(screen.getByText('Max 8.0 GB').parentElement).toHaveTextContent('Max 8.0 GB]')
  })

  it('does not set aria-invalid when invalid is false or omitted', () => {
    const { rerender } = render(
      <Slider ariaLabel="Test slider" max={100} min={0} name="test-slider" onValueChange={vi.fn()} value="50" />
    )

    const slider = screen.getByRole('slider', { name: 'Test slider' })
    expect(slider).not.toHaveAttribute('aria-invalid')

    rerender(
      <Slider
        ariaLabel="Test slider"
        invalid={false}
        max={100}
        min={0}
        name="test-slider"
        onValueChange={vi.fn()}
        value="50"
      />
    )

    expect(slider).not.toHaveAttribute('aria-invalid')
  })

  it('sets aria-invalid to true when invalid is true', () => {
    render(
      <Slider ariaLabel="Test slider" invalid max={100} min={0} name="test-slider" onValueChange={vi.fn()} value="50" />
    )

    const slider = screen.getByRole('slider', { name: 'Test slider' })
    expect(slider).toHaveAttribute('aria-invalid', 'true')
  })

  it('applies error ring on the wrapper when invalid is true', () => {
    const { container, rerender } = render(
      <Slider ariaLabel="Test slider" max={100} min={0} name="test-slider" onValueChange={vi.fn()} value="50" />
    )

    const wrapper = container.firstElementChild as HTMLElement
    expect(wrapper.className).not.toContain('ring-bad')

    rerender(
      <Slider ariaLabel="Test slider" invalid max={100} min={0} name="test-slider" onValueChange={vi.fn()} value="50" />
    )

    const wrapperAfter = container.firstElementChild as HTMLElement
    expect(wrapperAfter.className).toContain('ring-bad')
  })
})

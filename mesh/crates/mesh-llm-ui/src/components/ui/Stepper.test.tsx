import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Stepper } from '@/components/ui/Stepper'

describe('Stepper', () => {
  it('renders minus button, value input, and plus button', () => {
    render(<Stepper value={5} onChange={vi.fn()} aria-label="Threshold" />)

    expect(screen.getByRole('group', { name: 'Threshold' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Decrease Threshold' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Increase Threshold' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Threshold value' })).toHaveValue('5')
  })

  it('calls onChange with decremented value on minus click', () => {
    const handleChange = vi.fn()
    render(<Stepper value={5} onChange={handleChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'Decrease value' }))

    expect(handleChange).toHaveBeenCalledWith(4)
  })

  it('calls onChange with incremented value on plus click', () => {
    const handleChange = vi.fn()
    render(<Stepper value={5} onChange={handleChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'Increase value' }))

    expect(handleChange).toHaveBeenCalledWith(6)
  })

  it('respects min and max bounds', () => {
    const handleChange = vi.fn()
    render(<Stepper value={0} min={0} max={10} onChange={handleChange} />)

    expect(screen.getByRole('button', { name: 'Decrease value' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Increase value' })).toBeEnabled()

    fireEvent.click(screen.getByRole('button', { name: 'Increase value' }))
    expect(handleChange).toHaveBeenCalledWith(1)

    fireEvent.click(screen.getByLabelText('Increase value'))
    expect(handleChange).toHaveBeenCalledWith(1)
  })

  it('disables both buttons when at both bounds', () => {
    render(<Stepper value={5} min={5} max={5} onChange={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Decrease value' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Increase value' })).toBeDisabled()
  })

  it('uses custom step size', () => {
    const handleChange = vi.fn()
    render(<Stepper value={0} step={10} onChange={handleChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'Increase value' }))
    expect(handleChange).toHaveBeenCalledWith(10)
  })

  it('handles input change by typing a number', () => {
    const handleChange = vi.fn()
    render(<Stepper value={0} onChange={handleChange} />)

    fireEvent.change(screen.getByRole('textbox'), { target: { value: '42' } })

    expect(handleChange).toHaveBeenCalledWith(42)
  })

  it('ignores non-numeric input', () => {
    const handleChange = vi.fn()
    render(<Stepper value={0} onChange={handleChange} />)

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'abc' } })

    expect(handleChange).not.toHaveBeenCalled()
  })

  it('clamps typed value to bounds', () => {
    const handleChange = vi.fn()
    render(<Stepper value={5} min={0} max={10} onChange={handleChange} />)

    fireEvent.change(screen.getByRole('textbox'), { target: { value: '100' } })

    expect(handleChange).toHaveBeenCalledWith(10)
  })

  it('increments on ArrowUp key', () => {
    const handleChange = vi.fn()
    render(<Stepper value={5} onChange={handleChange} />)

    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'ArrowUp' })

    expect(handleChange).toHaveBeenCalledWith(6)
  })

  it('decrements on ArrowDown key', () => {
    const handleChange = vi.fn()
    render(<Stepper value={5} onChange={handleChange} />)

    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'ArrowDown' })

    expect(handleChange).toHaveBeenCalledWith(4)
  })

  it('disables all controls when disabled is true', () => {
    render(<Stepper value={5} disabled onChange={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Decrease value' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Increase value' })).toBeDisabled()
    expect(screen.getByRole('textbox')).toBeDisabled()
  })

  it('applies custom class names', () => {
    const { container } = render(<Stepper value={5} className="custom-class" onChange={vi.fn()} />)

    expect(container.firstChild).toHaveClass('custom-class')
  })
})

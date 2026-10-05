import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { NativeSelect } from '@/components/ui/NativeSelect'

const options = [
  { value: 'a', label: 'Option A' },
  { value: 'b', label: 'Option B' },
  { value: 'c', label: 'Option C' }
]

describe('NativeSelect', () => {
  it('renders a select element with the given options', () => {
    render(
      <NativeSelect ariaLabel="Test select" name="test-select" onValueChange={vi.fn()} options={options} value="a" />
    )

    const select = screen.getByRole('combobox', { name: 'Test select' })
    expect(select).toBeInTheDocument()
    expect(select).toHaveValue('a')
  })

  it('does not set aria-invalid when invalid is false or omitted', () => {
    const { rerender } = render(
      <NativeSelect ariaLabel="Test select" name="test-select" onValueChange={vi.fn()} options={options} value="a" />
    )

    const select = screen.getByRole('combobox', { name: 'Test select' })
    expect(select).not.toHaveAttribute('aria-invalid')

    rerender(
      <NativeSelect
        ariaLabel="Test select"
        name="test-select"
        onValueChange={vi.fn()}
        options={options}
        value="a"
        invalid={false}
      />
    )

    expect(select).not.toHaveAttribute('aria-invalid')
  })

  it('sets aria-invalid to true when invalid is true', () => {
    render(
      <NativeSelect
        ariaLabel="Test select"
        invalid
        name="test-select"
        onValueChange={vi.fn()}
        options={options}
        value="a"
      />
    )

    const select = screen.getByRole('combobox', { name: 'Test select' })
    expect(select).toHaveAttribute('aria-invalid', 'true')
  })

  it('applies error border and shadow classes when invalid is true', () => {
    const { rerender } = render(
      <NativeSelect ariaLabel="Test select" name="test-select" onValueChange={vi.fn()} options={options} value="a" />
    )

    const select = screen.getByRole('combobox', { name: 'Test select' })
    expect(select.className).not.toContain('border-bad')

    rerender(
      <NativeSelect
        ariaLabel="Test select"
        invalid
        name="test-select"
        onValueChange={vi.fn()}
        options={options}
        value="a"
      />
    )

    expect(select.className).toContain('border-bad')
    expect(select.className).toContain('shadow-[var(--shadow-surface-error-inset)]')
  })
})

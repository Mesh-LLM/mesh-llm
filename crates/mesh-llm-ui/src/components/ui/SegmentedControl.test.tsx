import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { SegmentedControl } from '@/components/ui/SegmentedControl'

const options = [
  { value: 'on', label: 'On' },
  { value: 'off', label: 'Off' },
  { value: 'auto', label: 'Auto' }
]

describe('SegmentedControl', () => {
  it('renders radio group with the given options', () => {
    render(
      <SegmentedControl
        ariaLabel="Test segmented"
        name="test-segmented"
        onValueChange={vi.fn()}
        options={options}
        value="on"
      />
    )

    const group = screen.getByRole('radiogroup', { name: 'Test segmented' })
    expect(group).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'On' })).toBeChecked()
  })

  it('does not set aria-invalid when invalid is false or omitted', () => {
    const { rerender } = render(
      <SegmentedControl
        ariaLabel="Test segmented"
        name="test-segmented"
        onValueChange={vi.fn()}
        options={options}
        value="on"
      />
    )

    const group = screen.getByRole('radiogroup', { name: 'Test segmented' })
    expect(group).not.toHaveAttribute('aria-invalid')

    rerender(
      <SegmentedControl
        ariaLabel="Test segmented"
        name="test-segmented"
        onValueChange={vi.fn()}
        options={options}
        value="on"
        invalid={false}
      />
    )

    expect(group).not.toHaveAttribute('aria-invalid')
  })

  it('sets aria-invalid to true when invalid is true', () => {
    render(
      <SegmentedControl
        ariaLabel="Test segmented"
        invalid
        name="test-segmented"
        onValueChange={vi.fn()}
        options={options}
        value="on"
      />
    )

    const group = screen.getByRole('radiogroup', { name: 'Test segmented' })
    expect(group).toHaveAttribute('aria-invalid', 'true')
  })

  it('applies error border and shadow on pill variant when invalid is true', () => {
    const { rerender } = render(
      <SegmentedControl
        ariaLabel="Test segmented"
        name="test-segmented"
        onValueChange={vi.fn()}
        options={options}
        value="on"
        variant="pill"
      />
    )

    const group = screen.getByRole('radiogroup', { name: 'Test segmented' })
    expect(group.className).not.toContain('border-bad')

    rerender(
      <SegmentedControl
        ariaLabel="Test segmented"
        invalid
        name="test-segmented"
        onValueChange={vi.fn()}
        options={options}
        value="on"
        variant="pill"
      />
    )

    expect(group.className).toContain('border-bad')
    expect(group.className).toContain('shadow-[var(--shadow-surface-error-inset)]')
  })

  it('does not apply error border classes on buttons variant when invalid is true', () => {
    render(
      <SegmentedControl
        ariaLabel="Test segmented"
        invalid
        name="test-segmented"
        onValueChange={vi.fn()}
        options={options}
        value="on"
        variant="buttons"
      />
    )

    const group = screen.getByRole('radiogroup', { name: 'Test segmented' })
    expect(group).toHaveAttribute('aria-invalid', 'true')
    expect(group.className).not.toContain('border-bad')
  })
})

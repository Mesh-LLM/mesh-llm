import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { NumberField } from './NumberField'

describe('NumberField', () => {
  it('renders the unit beneath the input and right aligned', () => {
    const { container } = render(<NumberField aria-label="Draft tokens" type="number" unit="tokens" />)

    const input = screen.getByRole('spinbutton', { name: 'Draft tokens' })
    const unit = screen.getByText('tokens')
    const wrapper = container.firstElementChild

    expect(wrapper).toHaveClass('grid', 'justify-items-end')
    expect(Array.from(wrapper?.childNodes ?? [])).toEqual([input, unit])
    expect(unit).toHaveClass('block', 'text-right')
  })

  it('uses invalid and disabled states on the input', () => {
    render(<NumberField aria-label="Timeout" disabled invalid type="number" unit="ms" />)

    const input = screen.getByRole('spinbutton', { name: 'Timeout' })

    expect(input).toBeDisabled()
    expect(input).toHaveAttribute('aria-invalid', 'true')
    expect(input).toHaveClass('border-bad')
  })
})

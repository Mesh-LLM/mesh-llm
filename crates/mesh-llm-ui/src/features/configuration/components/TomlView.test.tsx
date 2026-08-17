import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { TomlView } from '@/features/configuration/components/TomlView'

describe('TomlView', () => {
  it('formats validation diagnostics with a readable path and message', () => {
    render(
      <TomlView
        nodes={[]}
        assigns={[]}
        reviewMode
        validationWarnings={[
          {
            kind: 'warn',
            text: 'defaults.throughput.continuous_batching: defaults.throughput.continuous_batching must be one of: auto'
          }
        ]}
      />
    )

    const validationPanel = screen.getByRole('heading', { name: 'Validation' }).closest('section')
    expect(validationPanel).not.toBeNull()

    const warning = within(validationPanel as HTMLElement).getByText(
      'defaults.throughput.continuous_batching must be one of: auto'
    )
    expect(warning).toHaveClass('toml-warning-message')
    expect(within(validationPanel as HTMLElement).getByText('defaults.throughput.continuous_batching')).toHaveClass(
      'toml-warning-path'
    )
  })
})

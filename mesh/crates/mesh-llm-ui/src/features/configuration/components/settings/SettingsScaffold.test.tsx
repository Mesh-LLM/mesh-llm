import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SettingsRow } from '@/features/configuration/components/settings/SettingsScaffold'

describe('SettingsRow', () => {
  it('renders label and children', () => {
    render(
      <SettingsRow hint="A description" hintId="setting-desc" label="My Setting">
        <input data-testid="my-input" />
      </SettingsRow>
    )

    expect(screen.getByText('My Setting')).toBeInTheDocument()
    expect(screen.getByTestId('my-input')).toBeInTheDocument()
  })

  it('renders an error message when errorMessage is provided', () => {
    render(
      <SettingsRow
        errorMessage="Value is required"
        errorMessageId="setting-error"
        hint="A description"
        hintId="setting-desc"
        label="My Setting"
      >
        <input data-testid="my-input" />
      </SettingsRow>
    )

    expect(screen.getByText('Value is required')).toBeInTheDocument()
    expect(screen.getByText('Value is required')).toHaveAttribute('id', 'setting-error')
  })

  it('does not apply border-bad to the row wrapper when errorMessage is present', () => {
    const { container } = render(
      <SettingsRow
        errorMessage="Value is required"
        errorMessageId="setting-error"
        hint="A description"
        hintId="setting-desc"
        label="My Setting"
      >
        <input data-testid="my-input" />
      </SettingsRow>
    )

    const row = container.querySelector('[data-settings-row="true"]') as HTMLElement
    expect(row).toBeInTheDocument()
    expect(row.className).not.toContain('border-bad')
  })

  it('applies opacity-55 when disabled is true', () => {
    const { container } = render(
      <SettingsRow disabled hint="A description" hintId="setting-desc" label="My Setting">
        <input data-testid="my-input" />
      </SettingsRow>
    )

    const row = container.querySelector('[data-settings-row="true"]') as HTMLElement
    expect(row.className).toContain('opacity-55')
  })

  it('uses steady row spacing and top-aligned controls', () => {
    const { container } = render(
      <SettingsRow hint="A description" hintId="setting-desc" label="My Setting">
        <input data-testid="my-input" />
      </SettingsRow>
    )

    const row = container.querySelector('[data-settings-row="true"]') as HTMLElement
    const controlColumn = screen.getByTestId('my-input').parentElement as HTMLElement

    expect(row).toHaveClass('min-h-[68px]', 'md:items-start')
    expect(controlColumn).toHaveClass('md:pt-0.5')
  })

  it('keeps label text top-aligned when an accessory appears', () => {
    render(
      <SettingsRow
        hint="A description"
        hintId="setting-desc"
        label="My Setting"
        labelAccessory={<button type="button">Reset</button>}
      >
        <input data-testid="my-input" />
      </SettingsRow>
    )

    const labelRow = screen.getByText('My Setting').parentElement as HTMLElement
    const accessory = screen.getByRole('button', { name: 'Reset' }).parentElement as HTMLElement

    expect(labelRow).toHaveClass('min-h-6', 'items-start')
    expect(accessory).toHaveClass('-mt-0.5')
  })
})

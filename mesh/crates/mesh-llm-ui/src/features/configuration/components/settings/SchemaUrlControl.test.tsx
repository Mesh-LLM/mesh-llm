import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'
import { SchemaUrlControl } from './SchemaUrlControl'

function makeSetting(
  overrides: Partial<ConfigurationDefaultsSetting> & Pick<ConfigurationDefaultsSetting, 'label'>
): ConfigurationDefaultsSetting {
  return {
    id: 'test-setting',
    categoryId: 'runtime',
    icon: 'cog',
    description: 'Test setting',
    inheritedLabel: overrides.label,
    canonicalPath: 'test.setting',
    control: { kind: 'text', name: 'test-setting', value: '', placeholder: '' },
    ...overrides
  }
}

describe('SchemaUrlControl', () => {
  it('renders with aria-invalid="true" when invalid prop is true', () => {
    render(
      <SchemaUrlControl
        invalid={true}
        onChange={() => {}}
        setting={makeSetting({ label: 'Endpoint' })}
        value="http://localhost:4317"
      />
    )
    const input = screen.getByLabelText('Endpoint')
    expect(input).toHaveAttribute('aria-invalid', 'true')
  })

  it('renders without aria-invalid when invalid prop is not set', () => {
    render(
      <SchemaUrlControl
        onChange={() => {}}
        setting={makeSetting({ label: 'Endpoint' })}
        value="http://localhost:4317"
      />
    )
    const input = screen.getByLabelText('Endpoint')
    expect(input).not.toHaveAttribute('aria-invalid')
  })

  it('renders without aria-invalid when invalid is false', () => {
    render(
      <SchemaUrlControl
        invalid={false}
        onChange={() => {}}
        setting={makeSetting({ label: 'Endpoint' })}
        value="http://localhost:4317"
      />
    )
    const input = screen.getByLabelText('Endpoint')
    expect(input).not.toHaveAttribute('aria-invalid')
  })

  it('sets aria-invalid when validation fails on the current value', () => {
    render(
      <SchemaUrlControl
        onChange={() => {}}
        setting={makeSetting({ label: 'Endpoint', valueSchema: { kind: 'url' } })}
        value="not-a-url"
      />
    )
    const input = screen.getByLabelText('Endpoint')
    expect(input).toHaveAttribute('aria-invalid', 'true')
  })

  it('applies error classes when invalid', () => {
    render(
      <SchemaUrlControl
        invalid={true}
        onChange={() => {}}
        setting={makeSetting({ label: 'Endpoint' })}
        value="http://localhost:4317"
      />
    )
    const input = screen.getByLabelText('Endpoint')
    // The input should have border-bad and shadow-surface-error-inset classes
    expect(input.className).toMatch(/border-bad/)
    expect(input.className).toMatch(/shadow-surface-error-inset/)
  })

  it('does not apply error classes when valid', () => {
    render(
      <SchemaUrlControl
        onChange={() => {}}
        setting={makeSetting({ label: 'Endpoint' })}
        value="http://localhost:4317"
      />
    )
    const input = screen.getByLabelText('Endpoint')
    expect(input.className).not.toMatch(/border-bad/)
    expect(input.className).not.toMatch(/shadow-surface-error-inset/)
  })
})

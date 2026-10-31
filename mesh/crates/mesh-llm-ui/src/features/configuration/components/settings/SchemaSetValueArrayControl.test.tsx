import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ConfigurationDefaultsSetting, ConfigurationSettingObjectSchema } from '@/features/app-tabs/types'
import { SchemaArrayControl } from './SchemaArrayControl'

const itemSchema: ConfigurationSettingObjectSchema = {
  kind: 'object',
  properties: [
    { label: 'Endpoint', name: 'endpoint', required: true, value_schema: { kind: 'url' } },
    { label: 'Public key', name: 'public_key', required: false, value_schema: { kind: 'string' } }
  ]
}

const setting: ConfigurationDefaultsSetting = {
  id: 'witness',
  categoryId: 'runtime',
  icon: 'cog',
  label: 'Witnesses',
  description: 'Witness services',
  inheritedLabel: 'Witnesses',
  canonicalPath: 'plugin.capsules.witness',
  rendererId: 'set-value',
  restartScope: 'process_restart',
  control: { kind: 'text', name: 'witness', value: '', placeholder: 'https://witness.example.org' },
  valueSchema: { kind: 'array', items: itemSchema }
}

function Harness({ initial = '' }: { initial?: string }) {
  const [value, setValue] = useState(initial)
  return (
    <>
      <SchemaArrayControl onChange={setValue} setting={setting} value={value} />
      <output data-testid="value">{value}</output>
    </>
  )
}

describe('SchemaSetValueArrayControl', () => {
  it('shows the placeholder greyed and stores nothing until Set is clicked', () => {
    render(<Harness />)
    const input = screen.getByLabelText('Witnesses (not set)')
    expect(input).toHaveValue('https://witness.example.org')
    expect(input).toHaveAttribute('data-set', 'false')
    expect(screen.getByTestId('value')).toHaveTextContent('')
    expect(screen.getByText('Not set: no witness until you click Set.')).toBeInTheDocument()
  })

  it('Set stores the value, turns it set, and shows a brief checkmark', async () => {
    render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'Set' }))
    expect(screen.getByTestId('value')).toHaveTextContent('[{"endpoint":"https://witness.example.org"}]')
    expect(screen.getByLabelText('Witnesses 1')).toHaveAttribute('data-set', 'true')
    expect(screen.getByText('Saved. Takes effect when mesh-llm restarts.')).toBeInTheDocument()
    expect(screen.queryByText(/Not set/)).not.toBeInTheDocument()
  })

  it('edits a set value and sets it again; Set is disabled until the value changes', async () => {
    render(<Harness initial='[{"endpoint":"https://a.example"}]' />)
    const set = screen.getByRole('button', { name: 'Set' })
    expect(set).toBeDisabled()
    fireEvent.change(screen.getByLabelText('Witnesses 1'), { target: { value: 'https://b.example' } })
    expect(screen.getByTestId('value')).toHaveTextContent('https://a.example')
    await userEvent.click(set)
    expect(screen.getByTestId('value')).toHaveTextContent('[{"endpoint":"https://b.example"}]')
  })

  it('adds another value with its own Set, and Clear removes one', async () => {
    render(<Harness initial='[{"endpoint":"https://a.example","public_key":"ab"}]' />)
    await userEvent.click(screen.getByRole('button', { name: /Add your own/ }))
    fireEvent.change(screen.getByLabelText('Witnesses 2'), { target: { value: 'https://c.example' } })
    await userEvent.click(screen.getAllByRole('button', { name: 'Set' })[1]!)
    expect(screen.getByTestId('value')).toHaveTextContent(
      '[{"endpoint":"https://a.example","public_key":"ab"},{"endpoint":"https://c.example"}]'
    )
    await userEvent.click(screen.getAllByRole('button', { name: 'Clear' })[0]!)
    expect(screen.getByTestId('value')).toHaveTextContent('[{"endpoint":"https://c.example"}]')
  })

  it('clearing the first of two rows leaves the second row showing its own value', async () => {
    render(<Harness initial='[{"endpoint":"https://a.example"},{"endpoint":"https://b.example"}]' />)
    fireEvent.change(screen.getByLabelText('Witnesses 1'), { target: { value: 'https://a-edited.example' } })
    await userEvent.click(screen.getAllByRole('button', { name: 'Clear' })[0]!)
    expect(screen.getByLabelText('Witnesses 1')).toHaveValue('https://b.example')
    expect(screen.getByTestId('value')).toHaveTextContent('[{"endpoint":"https://b.example"}]')
  })

  it('Set on the first of two pending rows keeps the second row and its draft', async () => {
    render(<Harness initial='[{"endpoint":"https://a.example"}]' />)
    const add = screen.getByRole('button', { name: /Add your own/ })
    await userEvent.click(add)
    await userEvent.click(add)
    fireEvent.change(screen.getByLabelText('Witnesses 2'), { target: { value: 'https://one.example' } })
    fireEvent.change(screen.getByLabelText('Witnesses 3'), { target: { value: 'https://two.example' } })
    await userEvent.click(screen.getAllByRole('button', { name: 'Set' })[1]!)
    expect(screen.getByTestId('value')).toHaveTextContent('https://one.example')
    expect(screen.getByLabelText('Witnesses 3')).toHaveValue('https://two.example')
  })

  it('edits a nested secondary property at its own path', () => {
    const nested: ConfigurationSettingObjectSchema = {
      kind: 'object',
      properties: [
        { label: 'Endpoint', name: 'endpoint', required: true, value_schema: { kind: 'url' } },
        {
          label: 'Auth',
          name: 'auth',
          required: false,
          value_schema: {
            kind: 'object',
            properties: [{ label: 'Token', name: 'token', required: false, value_schema: { kind: 'string' } }]
          }
        }
      ]
    }
    function NestedHarness() {
      const [value, setValue] = useState('[{"endpoint":"https://a.example"}]')
      return (
        <>
          <SchemaArrayControl
            onChange={setValue}
            setting={{ ...setting, valueSchema: { kind: 'array', items: nested } }}
            value={value}
          />
          <output data-testid="value">{value}</output>
        </>
      )
    }
    render(<NestedHarness />)
    fireEvent.change(screen.getByLabelText(/Token/), { target: { value: 'abc' } })
    expect(screen.getByTestId('value')).toHaveTextContent('[{"endpoint":"https://a.example","auth":{"token":"abc"}}]')
  })

  it('keeps the plain object-array rows when the primary property is not a string', () => {
    const numeric: ConfigurationSettingObjectSchema = {
      kind: 'object',
      properties: [{ label: 'Weight', name: 'weight', required: true, value_schema: { kind: 'integer' } }]
    }
    render(
      <SchemaArrayControl
        onChange={() => {}}
        setting={{ ...setting, valueSchema: { kind: 'array', items: numeric } }}
        value='[{"weight":3}]'
      />
    )
    expect(screen.queryByRole('button', { name: 'Set' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Add witness/ })).toBeInTheDocument()
  })
})

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
  })

  it('Set stores the value, turns it set, and shows a brief checkmark', async () => {
    render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'Set' }))
    expect(screen.getByTestId('value')).toHaveTextContent('[{"endpoint":"https://witness.example.org"}]')
    expect(screen.getByLabelText('Witnesses 1')).toHaveAttribute('data-set', 'true')
    expect(screen.getByLabelText('Saved')).toBeInTheDocument()
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
})

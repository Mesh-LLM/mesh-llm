import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'
import { SchemaArrayControl } from './SchemaArrayControl'

function listSetting(key: string): ConfigurationDefaultsSetting {
  return {
    id: `plugin.demo.${key}`,
    categoryId: 'plugin-host',
    icon: 'cog',
    description: key,
    inheritedLabel: key,
    label: key,
    canonicalPath: `plugin.demo.${key}`,
    valueSchema: { kind: 'array', items: { kind: 'string' } },
    control: { kind: 'text', name: `plugin.demo.${key}`, value: '', placeholder: '' }
  }
}

/** Renders the control the way the settings form does, keeping its value. */
function renderControl(key: string, initial: string) {
  const values: string[] = []
  function Form() {
    const [value, setValue] = useState(initial)
    return (
      <SchemaArrayControl
        onChange={(next) => {
          values.push(next)
          setValue(next)
        }}
        setting={listSetting(key)}
        value={value}
      />
    )
  }
  render(<Form />)
  const textbox = screen.getByRole('textbox', { name: key }) as HTMLTextAreaElement
  return { textbox, values }
}

describe('SchemaArrayControl', () => {
  it('shows other list settings as written', () => {
    expect(renderControl('args', 'sold, also-sold').textbox.value).toBe('sold\nalso-sold')
  })

  it('does not decode JSON for other list settings', () => {
    expect(renderControl('args', '["x"]').textbox.value).toBe('["x"]')
  })
})

describe('SchemaArrayControl for a plugin model allowlist', () => {
  it('shows one model ID per line, commas included', () => {
    expect(renderControl('models', '["a,b","c"]').textbox.value).toBe('a,b\nc')
  })

  it('shows an explicit empty list as [] and an unset one as nothing', () => {
    expect(renderControl('models', '[]').textbox.value).toBe('[]')
  })

  it('keeps a model ID with a comma whole when another line is edited', () => {
    const { textbox, values } = renderControl('models', '["a,b","c"]')
    fireEvent.change(textbox, { target: { value: 'a,b\nc\nd' } })
    expect(values.at(-1)).toBe('["a,b","c","d"]')
    expect(textbox.value).toBe('a,b\nc\nd')
  })

  it('keeps a new empty line while typing and saves plain IDs comma-separated', () => {
    const { textbox, values } = renderControl('models', 'sold')
    fireEvent.change(textbox, { target: { value: 'sold\n' } })
    expect(values.at(-1)).toBe('sold')
    expect(textbox.value).toBe('sold\n')
    fireEvent.change(textbox, { target: { value: 'sold\nalso-sold' } })
    expect(values.at(-1)).toBe('sold,also-sold')
  })

  it('sets the empty list with [] and unsets the allowlist when cleared', () => {
    const { textbox, values } = renderControl('models', 'sold')
    fireEvent.change(textbox, { target: { value: '[]' } })
    expect(values.at(-1)).toBe('[]')
    fireEvent.change(textbox, { target: { value: '' } })
    expect(values.at(-1)).toBe('')
  })
})

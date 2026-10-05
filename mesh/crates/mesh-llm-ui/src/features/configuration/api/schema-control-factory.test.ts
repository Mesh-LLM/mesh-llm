import { describe, expect, it } from 'vitest'
import type { ConfigurationRuntimeControlStateEntry } from '@/features/app-tabs/types'
import { createSchemaControl, type SchemaControlFactoryEntry } from './schema-control-factory'

const selectionEntry: SchemaControlFactoryEntry = {
  canonical_path: 'runtime.native_runtime.selection',
  value_schema: {
    kind: 'one_of',
    variants: [{ kind: 'enum', values: ['recommended'] }, { kind: 'string' }]
  },
  control_behavior: { options_source: 'runtime_native_backends' },
  presentation: {
    control_hint: 'select',
    choices: [
      {
        value: 'recommended',
        label: 'Recommended (auto-detect)',
        description: 'Auto-detect the best backend for this host.'
      }
    ]
  }
}

const backendControlState: ConfigurationRuntimeControlStateEntry = {
  enabled: true,
  source: 'runtime',
  write_policy: 'preserve_existing',
  options: [
    { value: { kind: 'string', value: 'cpu' }, label: 'CPU', disabled: false, source: 'runtime_native_backends' },
    { value: { kind: 'string', value: 'metal' }, label: 'Metal', disabled: false, source: 'runtime_native_backends' }
  ]
}

describe('createSchemaControl for the native runtime backend selection', () => {
  it('renders a select with Recommended ahead of the host backends', () => {
    const control = createSchemaControl({
      entry: selectionEntry,
      name: 'selection',
      runtimeControlState: backendControlState
    })

    expect(control).toMatchObject({ kind: 'choice', presentation: 'select' })
    if (control.kind !== 'choice') throw new Error('expected a choice control')

    expect(control.options.map((option) => option.value)).toEqual(['', 'recommended', 'cpu', 'metal'])
    expect(control.options[1]).toEqual({
      value: 'recommended',
      label: 'Recommended (auto-detect)',
      description: 'Auto-detect the best backend for this host.'
    })
  })

  it('does not duplicate a static value that the runtime already reports', () => {
    const control = createSchemaControl({
      entry: {
        ...selectionEntry,
        value_schema: {
          kind: 'one_of',
          variants: [{ kind: 'enum', values: ['recommended', 'cpu'] }, { kind: 'string' }]
        }
      },
      name: 'selection',
      runtimeControlState: backendControlState
    })

    if (control.kind !== 'choice') throw new Error('expected a choice control')
    expect(control.options.map((option) => option.value)).toEqual(['', 'recommended', 'cpu', 'metal'])
  })
})

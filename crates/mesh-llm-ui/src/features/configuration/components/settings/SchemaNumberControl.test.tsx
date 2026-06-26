import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'
import { SchemaNumberControl } from '@/features/configuration/components/settings/SchemaNumberControl'

function contextSetting(): ConfigurationDefaultsSetting {
  return {
    id: 'ctx-size',
    categoryId: 'memory',
    icon: 'gauge',
    label: 'Context window size',
    description: 'Set the default context window size in tokens.',
    inheritedLabel: 'Applied when a placement does not override context size',
    canonicalPath: 'defaults.model_fit.ctx_size',
    valueSchema: { kind: 'integer' },
    control: {
      kind: 'range',
      name: 'ctx_size',
      value: '2048',
      min: 2048,
      max: 262144,
      step: 512,
      unit: 'tokens'
    }
  }
}

describe('SchemaNumberControl', () => {
  it('uses the common context slider for context window settings', () => {
    const handleChange = vi.fn()

    render(<SchemaNumberControl onChange={handleChange} setting={contextSetting()} value="49152" />)

    const slider = screen.getByRole('slider', { name: 'Context window size' })
    expect(slider).toHaveAttribute('aria-valuemin', '2048')
    expect(slider).toHaveAttribute('aria-valuemax', '262144')
    expect(screen.queryByRole('button', { name: '512' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '1K' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '2K' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '64K' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '256K' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '512K' })).not.toBeInTheDocument()
    expect(screen.queryByText(/Exact/)).not.toBeInTheDocument()
    expect(screen.getByText('49,152')).toBeInTheDocument()
    expect(screen.getByText(/tokens/).closest('div')).toHaveClass('justify-end')
    expect(screen.getByText(/tokens/).compareDocumentPosition(slider) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: '64K' }))
    expect(handleChange).toHaveBeenCalledWith('65536')
  })
})

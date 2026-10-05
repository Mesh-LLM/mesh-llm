import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'
import { SchemaPathControl } from '@/features/configuration/components/settings/SchemaPathControl'

function pathSetting(): ConfigurationDefaultsSetting {
  return {
    id: 'model-directory',
    categoryId: 'runtime',
    rendererId: 'host-directory-picker',
    icon: 'folder',
    label: 'Model directory',
    description: 'Host directory scanned for local model files.',
    inheritedLabel: 'Inherited',
    control: { kind: 'text', name: 'model-directory', value: '' }
  }
}

describe('SchemaPathControl', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('disables the path input while the directory picker is pending so typing cannot race a stale result', async () => {
    const user = userEvent.setup()
    let resolvePick: ((response: Response) => void) | undefined
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            resolvePick = resolve
          })
      )
    )
    const onChange = vi.fn()

    render(<SchemaPathControl onChange={onChange} setting={pathSetting()} value="./models" />)

    const input = screen.getByLabelText('Model directory')
    expect(input).toBeEnabled()

    await user.click(screen.getByRole('button', { name: 'Browse' }))
    expect(input).toBeDisabled()

    await resolvePick?.(new Response(JSON.stringify({ path: '/data/models' }), { status: 200 }))
    expect(await screen.findByRole('button', { name: 'Browse' })).toBeEnabled()
    expect(input).toBeEnabled()
    expect(onChange).toHaveBeenCalledWith('/data/models')
  })
})

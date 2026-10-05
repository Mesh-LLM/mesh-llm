import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const EXEMPLAR_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../../../../docs/plugins/exemplars/web-ui'
)

describe('plugin web UI exemplar contract', () => {
  it('keeps the sample bundle tied to the typed host registration contract', () => {
    const source = readFileSync(resolve(EXEMPLAR_ROOT, 'bundle/register-mesh-plugin-ui.ts'), 'utf8')

    expect(source).toContain('MeshPluginUiBundleModule')
    expect(source).toContain('satisfies MeshPluginUiBundleModule')
    expect(source).toContain('registerMeshPluginUi(host)')
    expect(source).toContain('overview: mountOverviewPage')
    expect(source).toContain('"page-actions": mountPageActionsSection')
    expect(source.match(/unmount\(\)/g)).toHaveLength(2)
    expect(source).toContain('host.navigation.navigateTo')
    expect(source).toContain('host.navigation.openPluginPage("overview")')
    expect(source).toContain('host.config.visible.settings.retention_days')
    expect(source).toContain('interactionStatus.setAttribute("role", "status")')
  })

  it('documents all lifecycle states while preserving non-UI capability samples', () => {
    const lifecycle = readFileSync(resolve(EXEMPLAR_ROOT, 'lifecycle-states.json'), 'utf8')

    expect(JSON.parse(lifecycle)).toMatchObject({
      plugin: 'web-ui-exemplar',
      non_ui_capabilities: ['exemplar.notes.v1'],
      states: {
        none: { state: 'none', non_ui_capabilities: ['exemplar.notes.v1'] },
        ready: { state: 'ready', non_ui_capabilities: ['exemplar.notes.v1'] },
        disabled: { state: 'disabled', non_ui_capabilities: ['exemplar.notes.v1'] },
        invalid: {
          state: 'invalid',
          unavailable_reason: 'web UI bundle root `bundle` is missing from the installed package',
          non_ui_capabilities: ['exemplar.notes.v1']
        },
        plugin_not_running: { state: 'plugin_not_running', non_ui_capabilities: ['exemplar.notes.v1'] }
      }
    })
  })
})

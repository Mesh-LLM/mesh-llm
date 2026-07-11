import type {
  MeshPluginUiBundleModule,
  MeshPluginUiConfigMountContext,
  MeshPluginUiMountContext
} from '../../../../../crates/mesh-llm-ui/src/features/plugins/web-ui/host-contract'

const moduleRegistration = {
  async registerMeshPluginUi(host) {
    host.state.update({ loadedAt: 'exemplar' })

    return {
      pages: {
        overview: mountOverviewPage
      },
      configSections: {
        retention: mountRetentionSection
      }
    }
  }
} satisfies MeshPluginUiBundleModule

export const registerMeshPluginUi = moduleRegistration.registerMeshPluginUi

function mountOverviewPage({ element, host, page }: MeshPluginUiMountContext) {
  element.replaceChildren()

  const heading = document.createElement('h2')
  heading.textContent = page.label

  const status = document.createElement('p')
  status.textContent = `${host.plugin.name} UI is ${host.webUi.state}; non-UI capability stays available when projection is disabled.`

  element.append(heading, status)

  const unsubscribe = host.state.subscribe((snapshot) => {
    status.dataset.snapshotKeys = Object.keys(snapshot).sort().join(',')
  })

  return {
    unmount() {
      unsubscribe()
      element.replaceChildren()
    }
  }
}

function mountRetentionSection({ element, host, section }: MeshPluginUiConfigMountContext) {
  element.replaceChildren()

  const label = document.createElement('label')
  label.textContent = section.title


  const input = document.createElement('input')
  input.name = 'retention_days'
  input.type = 'number'
  input.min = '1'
  input.max = '365'
  input.value = String(host.config.visible.settings.retention_days ?? 30)

  const save = document.createElement('button')
  save.type = 'button'
  save.textContent = 'Save retention'
  save.addEventListener('click', async () => {
    try {
      const nextConfig = await host.config.requestMutation({
        plugin: host.plugin.name,
        settings: {
          retention_days: Number(input.value)
        }
      })
      input.value = String(nextConfig.settings.retention_days ?? input.value)
      host.notifications.show({ title: 'Retention saved', tone: 'success' })
    } catch (error) {
      host.notifications.show({
        title: 'Retention save failed',
        description: error instanceof Error ? error.message : 'Unknown config mutation error',
        tone: 'error'
      })
    }
  })

  element.append(label, input, save)

  return {
    unmount() {
      save.replaceWith(save.cloneNode(true))
      element.replaceChildren()
    }
  }
}

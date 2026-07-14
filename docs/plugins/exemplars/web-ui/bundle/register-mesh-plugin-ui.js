/** @type {import('./host-contract').MeshPluginUiBundleModule} */
const moduleRegistration = {
  async registerMeshPluginUi(host) {
    host.state.update({ loadedAt: 'exemplar' })
    return {
      pages: { overview: mountOverviewPage },
      configSections: { retention: mountRetentionSection }
    }
  }
}
export const registerMeshPluginUi = moduleRegistration.registerMeshPluginUi

function mountOverviewPage({ element, host, page }) {
  const heading = document.createElement('h2')
  heading.textContent = page.label
  const status = document.createElement('p')
  status.textContent = `${host.plugin.name} UI is ${host.webUi.state}; non-UI capability exemplar.notes.v1 remains available.`
  element.replaceChildren(heading, status)
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

function mountRetentionSection({ element, host, section }) {
  const label = document.createElement('label')
  label.textContent = section.title
  const input = document.createElement('input')
  input.name = 'retention_days'
  input.type = 'number'
  input.min = '1'
  input.max = '365'
  input.value = String(host.config.visible.settings.retention_days ?? 14)
  const save = document.createElement('button')
  save.type = 'button'
  save.textContent = 'Save retention'
  const saveRetention = async () => {
    try {
      const config = await host.config.requestMutation({
        plugin: host.plugin.name,
        settings: { retention_days: Number(input.value) }
      })
      input.value = String(config.settings.retention_days ?? input.value)
      host.notifications.show({ title: 'Retention saved', tone: 'success' })
    } catch (error) {
      host.notifications.show({
        title: 'Retention save failed',
        description: error instanceof Error ? error.message : 'Unknown config mutation error',
        tone: 'error'
      })
    }
  }
  save.addEventListener('click', saveRetention)
  element.replaceChildren(label, input, save)
  return {
    unmount() {
      save.removeEventListener('click', saveRetention)
      element.replaceChildren()
    }
  }
}

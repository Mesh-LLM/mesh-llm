// DEV HARNESS ONLY. Mounts the page the way the console does: import the
// bundle entry, call `registerMeshPluginUi(host)`, and mount `pages.evidence`
// into an element -- with a standalone host in place of the console's.
import './console-styles.css'
import { registerMeshPluginUi } from '@/register-mesh-plugin-ui'
import { createStandaloneHost } from '@/plugin-host/standalone-host'

const harness = document.getElementById('harness')
if (!harness) throw new Error('dev harness: #harness missing')

const banner = document.createElement('p')
banner.style.cssText = 'margin:0;padding:6px 16px;font:12px/1.4 ui-monospace,monospace;opacity:.7;border-bottom:1px solid var(--color-border)'
const fixtures = import.meta.env.VITE_EVIDENCE_FIXTURES as string | undefined
banner.textContent = `dev harness — /plugins/capsules/evidence — ${fixtures ? `fixtures: ${fixtures}` : 'live: /api proxied to MESH_UI_API_ORIGIN'}`
const pageElement = document.createElement('main')
pageElement.style.cssText = 'padding:16px 24px'
harness.replaceChildren(banner, pageElement)

const host = createStandaloneHost({
  navigate: (path) => {
    console.info(`[dev harness] host navigation requested: ${path}`)
  }
})
const registration = await registerMeshPluginUi(host)
const mount = registration.pages.evidence
if (!mount) throw new Error('dev harness: bundle registered no `evidence` page')
await mount({ element: pageElement, host, page: host.webUi.pages![0]! })

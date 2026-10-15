// The capsules plugin's mesh-llm plugin web UI bundle.
//
// The console imports this module from the installed package (manifest
// `web_ui` block: bundle `main` rooted at `bundle/`, page `evidence`) and calls
// `registerMeshPluginUi(host)`; see docs/plugins/README.md "Plugin Web UI
// Projection Contract" in Mesh-LLM/mesh-llm. Each mount returns
// `{ unmount() }`, which tears down the React root, the injected stylesheet,
// and the mounted host.
import { createRoot } from 'react-dom/client'
import evidenceCss from '@/styles/evidence.css?inline'
import { EvidencePage, focusExchangeKeyFrom } from '@/features/capsules/pages/EvidencePage'
import { setPluginHost } from '@/plugin-host/host'
import { SCOPE_ATTRIBUTE } from '@/lib/scope'
import { EvidenceChip } from '@/features/capsules/contributions/EvidenceChip'
import type {
  MeshPluginUiBundleModule,
  MeshPluginUiContributionMountContext,
  MeshPluginUiMountContext,
  MeshPluginUiMountHandle
} from '@/plugin-host/host-contract'

export const EVIDENCE_PAGE_ID = 'evidence'
/** The manifest's contribution ids (`web_ui_manifest.rs`). */
export const CHAT_CONTRIBUTION_ID = 'evidence-chat'
export const LOGS_CONTRIBUTION_ID = 'evidence-logs'
const STYLE_ELEMENT_ID = 'capsules-evidence-styles'

function injectStyles(): HTMLStyleElement {
  const existing = document.getElementById(STYLE_ELEMENT_ID)
  existing?.remove()
  const style = document.createElement('style')
  style.id = STYLE_ELEMENT_ID
  style.textContent = evidenceCss
  document.head.appendChild(style)
  return style
}

export function mountEvidencePage({ element, host }: MeshPluginUiMountContext): MeshPluginUiMountHandle {
  setPluginHost(host)
  const style = injectStyles()
  const container = document.createElement('div')
  container.setAttribute(SCOPE_ATTRIBUTE, '')
  element.replaceChildren(container)
  const root = createRoot(container)
  root.render(<EvidencePage focusExchangeKey={focusExchangeKeyFrom(window.location.search)} />)
  return {
    unmount() {
      root.unmount()
      container.remove()
      style.remove()
      setPluginHost(null)
    }
  }
}

/** One line under a chat answer or in a Logs request. It uses the host it is
 *  handed, never the page's: the page need not be open. */
export function mountEvidenceChip({ element, host, subject }: MeshPluginUiContributionMountContext): MeshPluginUiMountHandle {
  const container = document.createElement('span')
  container.setAttribute(SCOPE_ATTRIBUTE, '')
  element.replaceChildren(container)
  const root = createRoot(container)
  root.render(<EvidenceChip host={host} subject={subject} />)
  return {
    unmount() {
      root.unmount()
      container.remove()
    }
  }
}

const moduleRegistration: MeshPluginUiBundleModule = {
  registerMeshPluginUi() {
    return {
      pages: { [EVIDENCE_PAGE_ID]: mountEvidencePage },
      contributions: { [CHAT_CONTRIBUTION_ID]: mountEvidenceChip, [LOGS_CONTRIBUTION_ID]: mountEvidenceChip }
    }
  }
}

export const registerMeshPluginUi = moduleRegistration.registerMeshPluginUi

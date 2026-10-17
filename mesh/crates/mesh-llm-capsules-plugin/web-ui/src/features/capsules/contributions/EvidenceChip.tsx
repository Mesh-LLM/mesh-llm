// The one line this plugin adds under a chat answer and in a Logs request:
// whether this node sealed that exchange, with a link to the record.
import { useEffect, useState } from 'react'
import type { MeshPluginUiContributionSubject, MeshPluginUiHost } from '@/plugin-host/host-contract'
import {
  lookupPath,
  recordPath,
  SEALED_LINE,
  type LookupResult
} from '@/features/capsules/lib/evidence-lookup'

type State = { kind: 'loading' } | { kind: 'done'; result: LookupResult } | { kind: 'none' }

export function EvidenceChip({ host, subject }: { host: MeshPluginUiHost; subject: MeshPluginUiContributionSubject }) {
  const path = lookupPath(subject)
  const [state, setState] = useState<State>(path ? { kind: 'loading' } : { kind: 'none' })

  useEffect(() => {
    if (!path) return
    let live = true
    host.network
      .fetchPlugin(path)
      .then(async (response) => {
        // A plugin without the route (older) or any failure: say nothing.
        if (!response.ok) throw new Error(String(response.status))
        return (await response.json()) as LookupResult
      })
      .then((result) => live && setState({ kind: 'done', result }))
      .catch(() => live && setState({ kind: 'none' }))
    return () => {
      live = false
    }
  }, [host, path])

  if (state.kind !== 'done') return null
  const to = recordPath(state.result)
  // A miss is not shown: a miss does not mean this node sealed nothing. For an
  // exchange this node served itself, the host gives the plugin no client
  // nonce and a different exchange id from the one the chat and Logs views
  // hold, so the record is there and still cannot be matched.
  if (!state.result.found || !to) return null
  const style = { fontSize: '12px', color: 'var(--color-fg-dim, inherit)' }
  return (
    <a
      data-testid="evidence-chip"
      href={to}
      onClick={(event) => {
        event.preventDefault()
        host.navigation.navigateTo(to)
      }}
      style={{ ...style, textDecoration: 'underline' }}
    >
      {SEALED_LINE}
    </a>
  )
}

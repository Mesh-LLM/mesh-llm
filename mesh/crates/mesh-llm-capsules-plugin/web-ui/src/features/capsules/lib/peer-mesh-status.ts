// Joins a Pane B peer_id (capsule-emit-mesh's own counterparty label --
// see `capsule_mesh_view.label_counterparty`) against the Network feature's
// live mesh state (`Peer`/`ModelSummary`, the "mesh
// status" side of the join: model, quant, native_context_length, latency,
// online). The join key formats don't share a schema across the two
// subsystems, so the match is best-effort with a safe "not available"
// fallback -- never a fabricated status when nothing matches.
import { useMemo } from 'react'
import { useDataMode } from '@/lib/data-mode'
import { adaptModelsToSummary } from '@/features/network/api/models-adapter'
import { adaptStatusToDashboard } from '@/features/network/api/status-adapter'
import { useModelsQuery } from '@/features/network/api/use-models-query'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { LatencySource } from '@/lib/api/types'
import type { ModelSummary, Peer } from '@/features/app-tabs/types'
import type { PaneBRow } from '@/features/capsules/api/sidecarTypes'
import { peerDisplayId, peerIdentityAliases } from '@/features/capsules/lib/peer-row-view'

export type PeerMeshStatus = {
  modelName: string | null
  quant: string | null
  contextLengthK: number | null
  latencyMs: number | null
  /** chooser-v2 §3: "latency with provenance" -- DIRECT means this node
   *  measured it itself, ESTIMATED means the figure was reported by
   *  another node/gossip. Never rendered as a bare number without this. */
  latencySource: LatencySource | null
  online: boolean
}

function stripPeerRefPrefix(peerId: string): string {
  const idx = peerId.indexOf(':')
  return idx === -1 ? peerId : peerId.slice(idx + 1)
}

/** Best-effort match: pane-b's `node:<served_by_node_id[:16]>` /
 *  `initiator:<ref>` / `counterparty:<ref>` forms against the Network
 *  feature's `Peer.id` (itself `node_id ?? id ?? hostname` per
 *  `status-adapter.resolvePeerId`). Exact match first, then the bare ref
 *  with/without the pane-b label prefix, then a short-id or id-prefix
 *  match for the truncated `node:` form. */
export function findMeshPeer(peerId: string, peers: readonly Peer[]): Peer | undefined {
  const bare = stripPeerRefPrefix(peerId)
  return (
    peers.find((peer) => peer.id === peerId) ??
    peers.find((peer) => peer.id === bare) ??
    peers.find((peer) => peer.shortId === bare) ??
    (bare.length >= 6 ? peers.find((peer) => peer.id.startsWith(bare)) : undefined) ??
    // The reverse: a FULL id alias (e.g. the row's 64-hex node id,
    // D3) against a mesh entry that
    // only carries a truncated id -- same >= 6-char guard on both sides.
    (bare.length >= 6 ? peers.find((peer) => peer.id.length >= 6 && bare.startsWith(peer.id)) : undefined)
  )
}

export function deriveMeshStatus(
  peerId: string,
  peers: readonly Peer[],
  models: readonly ModelSummary[]
): PeerMeshStatus | null {
  const peer = findMeshPeer(peerId, peers)
  if (!peer) return null
  const modelName = peer.hostedModels[0] ?? null
  const model = modelName ? models.find((candidate) => candidate.name === modelName) : undefined
  return {
    modelName,
    quant: model?.quant ?? null,
    contextLengthK: model?.ctxMaxK ?? null,
    latencyMs: peer.latencyMs,
    latencySource: peer.latencySource ?? null,
    online: peer.status === 'online'
  }
}

/** Every id string a Pane B row is known by: its display id plus the alias
 *  evidence (D3 -- signing key,
 *  endpoint id, FULL node id). The full node id matters here: the display id
 *  truncates to 16 chars, and a truncation-vs-full prefix heuristic is
 *  exactly how the same peer showed up a THIRD time under "advertised but
 *  unused" in the live shots. */
function rowAliasCandidates(row: PaneBRow): string[] {
  const { signingKeyId, endpointId, nodeId } = peerIdentityAliases(row)
  return [peerDisplayId(row), nodeId, endpointId, signingKeyId].filter((value): value is string => Boolean(value))
}

/** T7 §3-F -- the Peers table's second row group ("advertised but unused")
 *  is every mesh-known peer that Pane B carries no row for at all, i.e. no
 *  exchange has ever happened. Reuses `findMeshPeer`'s own best-effort
 *  matching (the same logic `deriveMeshStatus` already applies per row),
 *  widened over every alias a row carries, so a mesh peer that aliases a
 *  dealt-with row can never re-appear as "advertised but unused" -- one
 *  peer, one appearance. Excludes this node's own self entry
 *  (`status-adapter`'s `adaptSelfPeer`, `role: 'you'`) -- a node never
 *  exchanges with itself, so it can never be "unused" in the counterparty
 *  sense this group means. */
export function advertisedOnlyPeers(paneBRows: readonly PaneBRow[], peers: readonly Peer[]): Peer[] {
  const dealtWithMeshIds = new Set(
    paneBRows
      .flatMap((row) => rowAliasCandidates(row))
      .map((candidate) => findMeshPeer(candidate, peers)?.id)
      .filter((id): id is string => Boolean(id))
  )
  return peers.filter((peer) => peer.role !== 'you' && !dealtWithMeshIds.has(peer.id))
}

export type PeerMeshStatusIndex = {
  statusFor: (peerId: string) => PeerMeshStatus | null
  peers: readonly Peer[]
}

/** Live mode: reuses the same `useStatusQuery`/`useModelsQuery` +
 *  adapters the Network Dashboard already calls -- no new endpoint, no
 *  re-derivation of mesh state. Harness mode: uses the fixtures the caller
 *  supplies (kept in `peer-fixtures.ts`, co-located with the matching
 *  pane-b row fixtures so peer_ids line up by construction). */
export function usePeerMeshStatusIndex(
  harnessPeers: readonly Peer[],
  harnessModels: readonly ModelSummary[]
): PeerMeshStatusIndex {
  const { mode } = useDataMode()
  const liveMode = mode === 'live'
  const statusQuery = useStatusQuery({ enabled: liveMode })
  const modelsQuery = useModelsQuery({ enabled: liveMode })

  const liveModels = useMemo(
    () => (modelsQuery.data ? adaptModelsToSummary(modelsQuery.data.mesh_models) : []),
    [modelsQuery.data]
  )
  const liveDashboard = useMemo(
    () => (statusQuery.data ? adaptStatusToDashboard(statusQuery.data) : undefined),
    [statusQuery.data]
  )

  const peers = useMemo(
    () => (liveMode ? (liveDashboard?.peers ?? []) : harnessPeers),
    [liveMode, liveDashboard, harnessPeers]
  )
  const models = useMemo(() => (liveMode ? liveModels : harnessModels), [liveMode, liveModels, harnessModels])

  return useMemo(() => {
    const cache = new Map<string, PeerMeshStatus | null>()
    return {
      statusFor(peerId: string) {
        if (!cache.has(peerId)) cache.set(peerId, deriveMeshStatus(peerId, peers, models))
        return cache.get(peerId) ?? null
      },
      peers
    }
  }, [peers, models])
}

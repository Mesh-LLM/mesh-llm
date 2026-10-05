import type { ModelSummary } from '@/features/app-tabs/types'
import type { PeerInfo, ServingModelEntry, StatusPayload } from '@/lib/api/types'

const VIRTUAL_MESH_MODEL = 'mesh'

function addModelName(names: Set<string>, name: string | undefined) {
  const normalized = name?.trim()
  if (normalized && normalized !== VIRTUAL_MESH_MODEL) names.add(normalized)
}

function localRoutableModelNames(status: StatusPayload): string[] {
  if (status.node_state === 'client') return []

  return (status.serving_models ?? []).flatMap((model: ServingModelEntry) => {
    if (typeof model === 'string') return [model]
    return model.status === 'warm' ? [model.name] : []
  })
}

function peerRoutableModelNames(peer: PeerInfo): string[] {
  const hosted = peer.hosted_models?.filter(Boolean) ?? []
  if (hosted.length > 0 || peer.hosted_models_known === true) return hosted
  return peer.serving_models?.filter(Boolean) ?? []
}

export function statusBackedChatModels(status: StatusPayload | undefined): ModelSummary[] {
  if (!status) return []

  const names = new Set<string>()
  for (const model of localRoutableModelNames(status)) addModelName(names, model)
  for (const peer of status.peers ?? []) {
    for (const model of peerRoutableModelNames(peer)) addModelName(names, model)
  }

  return [...names].sort().map((name) => ({
    name,
    family: name.split('/')[0] ?? name,
    size: 'Unknown',
    context: 'Unknown',
    status: 'warm',
    tags: []
  }))
}

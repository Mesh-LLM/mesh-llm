// The user's routing intent for console chat, kept across reloads. Availability never rewrites it:
// a picked model that disappears stays picked (shown as unavailable) until the user changes it.
const STORAGE_KEY = 'mesh-llm.chat.routing-preferences'

export type ChatRoutingPreferences = {
  /** '' means "Mesh — automatic". */
  model: string
  /** Send `mesh_payment: { mode: 'free_only' }`: the node may only narrow, never widen, spending. */
  freeOnly: boolean
}

const DEFAULTS: ChatRoutingPreferences = { model: '', freeOnly: false }

export function loadChatRoutingPreferences(): ChatRoutingPreferences {
  try {
    const raw = globalThis.localStorage?.getItem(STORAGE_KEY)
    if (!raw) return DEFAULTS
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return DEFAULTS
    const { model, freeOnly } = parsed as Record<string, unknown>
    return {
      model: typeof model === 'string' ? model : '',
      freeOnly: freeOnly === true
    }
  } catch {
    return DEFAULTS
  }
}

export function saveChatRoutingPreferences(preferences: ChatRoutingPreferences): void {
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(preferences))
  } catch {
    // Private mode / quota: the pick still holds for this session.
  }
}

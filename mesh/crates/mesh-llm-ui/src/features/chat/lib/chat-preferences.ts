// The user's routing intent for console chat, kept across reloads. Availability never rewrites it:
// a picked model that disappears stays picked (shown as unavailable) until the user changes it.
const STORAGE_KEY = 'mesh-llm.chat.routing-preferences'

export type ChatRoutingPreferences = {
  /** '' means "Mesh — automatic". */
  model: string
}

const DEFAULTS: ChatRoutingPreferences = { model: '' }

export function loadChatRoutingPreferences(): ChatRoutingPreferences {
  try {
    const raw = globalThis.localStorage?.getItem(STORAGE_KEY)
    if (!raw) return DEFAULTS
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return DEFAULTS
    const { model } = parsed as Record<string, unknown>
    return { model: typeof model === 'string' ? model : '' }
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

/** The chat route's search params. `target` is a node's endpoint id (64 hex
 *  characters); while it is set, chat requests carry it as `x-mesh-target`. */
export type ChatSearch = {
  readonly target?: string
}

const ENDPOINT_ID = /^[0-9a-f]{64}$/i

export function parseChatSearch(search: Record<string, unknown>): ChatSearch {
  const target = typeof search.target === 'string' ? search.target.trim() : ''
  return ENDPOINT_ID.test(target) ? { target: target.toLowerCase() } : {}
}

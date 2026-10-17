// Ported unchanged from the mesh-llm console fork's `lib/local-time.ts`.
/** "Sep 27, 9:56 PM" in the viewer's own time zone, the way Chat shows it. */
const LOCAL_TIME = new Intl.DateTimeFormat(undefined, {
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit'
})

/** An exchange's time for the row face: local, never an ISO/UTC stamp. */
export function formatExchangeTimestamp(timestamp: string | null): string {
  if (!timestamp) return 'timestamp unavailable'
  const date = new Date(timestamp)
  return Number.isNaN(date.getTime()) ? timestamp : LOCAL_TIME.format(date)
}

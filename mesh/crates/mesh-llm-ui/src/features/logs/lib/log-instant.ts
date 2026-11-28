export function compareLogInstants(left: string, right: string): number {
  return Date.parse(left) - Date.parse(right)
}

export function sortByOccurredAt<T extends { readonly occurredAt: string }>(entries: readonly T[]): T[] {
  return entries
    .map((entry, index) => ({ entry, index }))
    .sort(
      (left, right) => compareLogInstants(left.entry.occurredAt, right.entry.occurredAt) || left.index - right.index
    )
    .map(({ entry }) => entry)
}

type FormatInstantOptions = {
  /** IANA zone to say the instant in; defaults to the reader's own zone. */
  readonly timeZone?: string
}

/**
 * Says a recorded instant with its date and zone ("Oct 27, 2026, 5:09:03 PM CDT", seconds kept so retries within one minute stay ordered), the receipt
 * presentation rule R6. The text locale is fixed to en-US so only the zone comes from the reader;
 * the sealed instant stays on the caller's `<time dateTime>` attribute.
 */
export function formatInstant(value: string, options: FormatInstantOptions = {}): string {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    timeZoneName: 'short',
    timeZone: options.timeZone
  }).format(parsed)
}

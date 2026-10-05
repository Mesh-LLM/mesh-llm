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

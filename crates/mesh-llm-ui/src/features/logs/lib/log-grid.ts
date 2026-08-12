export type GridBreakpoint = 'sm' | 'lg' | 'xl'

/**
 * Literal Tailwind classes keyed by breakpoint and span count so the JIT
 * scanner can detect them in source. `col-span-1` never needs to be emitted
 * because it is the implicit default.
 */
const SPAN_CLASS_BY_BREAKPOINT: Record<GridBreakpoint, Readonly<Record<number, string>>> = {
  sm: { 2: 'sm:col-span-2', 3: 'sm:col-span-3', 4: 'sm:col-span-4' },
  lg: { 2: 'lg:col-span-2', 3: 'lg:col-span-3', 4: 'lg:col-span-4' },
  xl: { 2: 'xl:col-span-2', 3: 'xl:col-span-3', 4: 'xl:col-span-4' }
}

/**
 * Returns the breakpoint-prefixed `col-span-N` class a cell needs so the last
 * (partial) row of a `columns`-wide grid is filled edge to edge.
 *
 * Cells before the trailing row and cells that would naturally occupy a single
 * column return `undefined`. When the trailing row has more cells than fit in
 * one column each, the earlier trailing cells absorb the extra columns so the
 * row still spans exactly `columns` columns.
 */
export function trailingRowSpanClass(
  count: number,
  index: number,
  columns: number,
  breakpoint: GridBreakpoint
): string | undefined {
  if (columns < 2 || columns > 4 || index < 0 || index >= count) return undefined
  const remainder = count % columns
  if (remainder === 0) return undefined
  const trailingStart = count - remainder
  if (index < trailingStart) return undefined
  const position = index - trailingStart
  const base = Math.floor(columns / remainder)
  const extra = columns % remainder
  const span = position < extra ? base + 1 : base
  if (span < 2) return undefined
  return SPAN_CLASS_BY_BREAKPOINT[breakpoint][span]
}

export type SummaryRawKind = 'backend' | 'mode' | 'none'

export type SummaryDescriptor = {
  readonly path: readonly string[]
  readonly booleans: readonly string[]
  readonly redacted: readonly string[]
  readonly conflicts: readonly (readonly string[])[]
  readonly hasPort: boolean
  readonly raw: SummaryRawKind
}

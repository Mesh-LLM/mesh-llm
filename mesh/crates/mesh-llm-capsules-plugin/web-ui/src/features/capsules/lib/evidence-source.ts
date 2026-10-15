// Where the Evidence tab's records are coming from, for the hero chip.
// Look finding 6: in fixture mode (`VITE_EVIDENCE_FIXTURES`, the dev server
// replaying a captured run -- src/lib/dev/evidence-fixtures-plugin.ts) the API
// answers, so the chip read "Live" over a saved sample. Replay is sample data;
// record mode (`VITE_EVIDENCE_FIXTURES_RECORD`) passes through to a real node,
// so it stays live.
export type EvidenceSource = 'sample' | 'live' | 'local'

export function evidenceSource(apiConnected: boolean, fixtureReplayRun: string | null | undefined): EvidenceSource {
  if (fixtureReplayRun) return 'sample'
  return apiConnected ? 'live' : 'local'
}

export const EVIDENCE_SOURCE_LABEL: Record<EvidenceSource, string> = {
  sample: 'Sample data',
  live: 'Live',
  local: 'Local'
}

// Harness fixtures for the Exchanges view -- so
// `pnpm dev`/`pnpm preview` (harness data mode) renders the
// balance/coverage header and the exchanges table without a live sidecar.
// Field shapes mirror `capsule_accountability_tab.build_served_summary_
// block`/`served_summary.ServedSummary.to_value` and `capsule_exchange_tab.
// build_exchange_list_payload` verbatim -- see `sidecarTypes.ts`.
import type { PaneAJson } from '@/features/capsules/api/sidecarTypes'
import type { PaneCListJson, PaneCRow } from '@/features/capsules/api/sidecarTypes'

export const HARNESS_PANE_A_PAYLOAD: PaneAJson = {
  operator: null,
  witness_checkpoint_supplied: true,
  rows: [],
  card: {
    served_summary: {
      state: 'verified',
      text: 'llama-2-7b: 24 served',
      source: 'self_derived',
      served_summary: {
        schema: 'mesh-served-summary/1',
        node_id: 'node:aa11bb22cc33dd44',
        selection: { from_entry: 1, to_entry: 40, covered_entries: 40, note: 'witnessed range only' },
        derivation: {
          kind: 'served_summary_fold',
          definition_digest: 'sha256:harness-definition-digest',
          by_model: {
            'llama-2-7b': {
              served: 24,
              completed: 22,
              failed: 2,
              refused: 0,
              refused_note: null,
              latency_p50_ms: '812.500',
              latency_p95_ms: '1420.000',
              latency_max_ms: '1980.000',
              weights_digest: { present: 0, absent: 24, note: 'no capsule field named weights_digest exists' },
              quantizations: ['q4_0'],
              floor_applied: false
            }
          },
          note: 'counts + latency distribution per model, over the selected witnessed range'
        },
        coverage: {
          checkpoint_root: 'a1b2c3d4e5f60718293a4b5c6d7e8f90',
          mmr_size: 64,
          log_id: 'harness-log',
          timestamp: '2026-09-08T16:58:05Z',
          witnesses: ['https://witness-a.example', 'https://witness-b.example'],
          witnessed: true,
          note: 'cross-check handle'
        },
        adjudications_received: { value: 8, source: 'self_held', note: 'verdicts received about served exchanges' },
        no_requester_identifiers: 'this summary reads and reports no requester-identity field',
        not_a_score: 'An account of facts, with a witness handle to verify them.'
      }
    }
  }
}

const CLEAN_EXCHANGE_PROPERTIES = {
  content_binding: { state: 'PASS' },
  producer_signature: { state: 'PASS' },
  local_inclusion: { state: 'PASS' },
  checkpoint_signature: { state: 'PASS' },
  external_registration: { state: 'PASS' },
  continuity: { state: 'PASS' },
  identity_authority: { state: 'NOT_PRESENT' },
  capture_coverage: { state: 'PASS' },
  outcome_corroboration: { state: 'PASS' }
}

// the artifact-disagrees case (v3 §2):
// `outcome_corroboration` itself must be FAIL -- a checkpoint-signature
// exception with no outcome disagreement would still render CLOSED, not
// CONTRADICTED (the six-state model and the nine-property Checks column
// are independent axes).
const CONTRADICTED_EXCHANGE_PROPERTIES = {
  content_binding: { state: 'PASS' },
  producer_signature: { state: 'PASS' },
  local_inclusion: { state: 'PASS' },
  checkpoint_signature: { state: 'PASS' },
  external_registration: { state: 'PASS' },
  continuity: { state: 'PASS' },
  identity_authority: { state: 'NOT_PRESENT' },
  capture_coverage: { state: 'PASS' },
  outcome_corroboration: { state: 'FAIL', text: 'reported outcomes disagree' }
}

// the seven curated rows below (one per
// right-cell state) are too few to page; `pnpm dev`'s harness mode needs a
// realistic-sized ledger to actually exercise windowed paging, the
// bounded-window banner, and the sticky header. This tail is deterministic
// (index-derived, no Math.random/Date.now) so a screenshot or a snapshot is
// reproducible run to run, and spans back to 3 Sep to match the v3 §2a
// worked example ("Showing 50 of 1,284 exchanges · 3 Sep – 11 Sep")
// verbatim once appended to the 7 curated rows below.
export const HARNESS_LEDGER_FILLER_ROW_COUNT = 1277
const FILLER_SPAN_START = new Date('2026-09-03T00:00:00Z').getTime()
const FILLER_SPAN_END = new Date('2026-09-11T08:00:00Z').getTime() // just before the earliest curated row

function fillerRowState(index: number): 'closed' | 'contradicted' | 'served' | 'refused' | 'absent' | 'not_asked' {
  const cycle = index % 12
  if (cycle === 0) return 'contradicted'
  if (cycle === 1) return 'served'
  if (cycle === 2) return 'refused'
  if (cycle === 3) return 'absent'
  if (cycle === 4) return 'not_asked'
  return 'closed'
}

function buildFillerExchangeRows(count: number): PaneCRow[] {
  const spanMs = FILLER_SPAN_END - FILLER_SPAN_START
  return Array.from({ length: count }, (_, i) => {
    // Newest filler row (i = count - 1) sits just before the curated rows;
    // oldest (i = 0) sits at the span start -- keeps the whole payload in
    // newest-first order once the Ledger's own time sort runs.
    const timestamp = new Date(FILLER_SPAN_START + Math.round((spanMs * i) / (count - 1))).toISOString()
    const key = `exch-filler-${String(i).padStart(4, '0')}`
    const state = fillerRowState(i)
    // Three consecutive ASKED rows share a session -- enough to render a
    // few multi-row rails without every row looking identical.
    const sessionId = state === 'served' ? undefined : `session-filler-${Math.floor(i / 3)}`

    const base = {
      exchange_key: key,
      role_tag: state === 'served' ? 'SERVED' : 'ASKED',
      mine: { state: 'present' as const, capsule_id: `mine_${key}` },
      timestamp,
      ...(sessionId ? { session_id: sessionId } : {})
    }

    switch (state) {
      case 'contradicted':
        return {
          ...base,
          header_state: 'issue',
          properties: CONTRADICTED_EXCHANGE_PROPERTIES,
          has_issue: true,
          theirs: { state: 'present', capsule_id: `theirs_${key}` },
          unilateral: false
        }
      case 'served':
        return {
          ...base,
          header_state: 'ok',
          properties: null,
          has_issue: false,
          theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'not_asked' as const },
          unilateral: true
        }
      case 'refused':
        return {
          ...base,
          header_state: 'ok',
          properties: null,
          has_issue: false,
          theirs: {
            state: 'absent',
            capsule_id: null,
            evidence_outcome: 'signed_refusal' as const,
            evidence_outcome_date: '4 Sep'
          },
          unilateral: true
        }
      case 'absent':
        return {
          ...base,
          header_state: 'ok',
          properties: null,
          has_issue: false,
          theirs: {
            state: 'absent',
            capsule_id: null,
            evidence_outcome: 'recorded_absence' as const,
            evidence_outcome_date: '4 Sep'
          },
          unilateral: true
        }
      case 'not_asked':
        return {
          ...base,
          header_state: 'ok',
          properties: null,
          has_issue: false,
          theirs: { state: 'absent', capsule_id: null },
          unilateral: true
        }
      case 'closed':
      default:
        return {
          ...base,
          header_state: 'ok',
          properties: CLEAN_EXCHANGE_PROPERTIES,
          has_issue: false,
          theirs: { state: 'present', capsule_id: `theirs_${key}` },
          unilateral: false
        }
    }
  })
}

// one row per right-cell state (v3 §2's
// six-row table), plus a served row and a second session, so `pnpm dev`
// shows a real mix: the append-only stream, the session rail (L-N/L-O),
// and every CLOSED/CONTRADICTED/OPEN·* status at once.
// 7 curated + the deterministic filler tail
// above, so harness mode actually pages.
// an 8th curated row (`exch-deleted-08`)
// so the content toggle's deletion state is reachable in harness mode too;
// `exch-closed-00` carries `mine.text`/`reply_text` matching v3 §3's Case A
// worked example verbatim, so the content toggle has real quoted text to
// show rather than the '—' fallback.
// two adjacent curated rows sharing a real
// `twin_bracket_id`, sorted newest-first ahead of everything else, so the
// twin bracket + its sentence + Compare diff + "Twins only" filter are all
// reachable in harness mode, as a client-marked pair would make them.

export const HARNESS_PANE_C_PAYLOAD: PaneCListJson = {
  row_count: 10 + HARNESS_LEDGER_FILLER_ROW_COUNT,
  default_sort: 'timestamp',
  filters: [],
  next_after_seq: null,
  archived_segments: [],
  rows: [
    {
      exchange_key: 'exch-twin-a-00',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: CLEAN_EXCHANGE_PROPERTIES,
      has_issue: false,
      mine: {
        state: 'present',
        capsule_id: 'mine_exch-twin-a-00',
        text: 'What is the capital of France?'
      },
      theirs: { state: 'present', capsule_id: 'theirs_exch-twin-a-00' },
      unilateral: false,
      timestamp: '2026-09-11T16:58:11Z',
      twin_bracket_id: 'twin-fixture-1',
      twin_comparison: {
        temperature: 0,
        seed: 1,
        model_identity_hash: 'd41d8cd98f00b204e9800998ecf8427e',
        settings_label: 'KV F16/F16'
      }
    },
    {
      exchange_key: 'exch-twin-b-00',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: CLEAN_EXCHANGE_PROPERTIES,
      has_issue: false,
      mine: {
        state: 'present',
        capsule_id: 'mine_exch-twin-b-00',
        text: 'What is the capital of France?'
      },
      theirs: { state: 'present', capsule_id: 'theirs_exch-twin-b-00' },
      unilateral: false,
      timestamp: '2026-09-11T16:58:10Z',
      twin_bracket_id: 'twin-fixture-1',
      twin_comparison: {
        temperature: 0,
        seed: 1,
        model_identity_hash: 'd41d8cd98f00b204e9800998ecf8427e',
        settings_label: 'KV F16/F16'
      }
    },
    {
      exchange_key: 'exch-closed-00',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: CLEAN_EXCHANGE_PROPERTIES,
      has_issue: false,
      mine: {
        state: 'present',
        capsule_id: 'mine_exch-closed-00',
        text: 'Summarise this thread…',
        reply_text: 'The thread covers three…',
        reply_capsule_id: 'theirs_reply_exch-closed-00'
      },
      theirs: { state: 'present', capsule_id: 'theirs_exch-closed-00' },
      unilateral: false,
      timestamp: '2026-09-11T16:58:05Z',
      session_id: 'session-refactor-notes'
    },
    {
      exchange_key: 'exch-contradicted-01',
      role_tag: 'ASKED',
      header_state: 'issue',
      properties: CONTRADICTED_EXCHANGE_PROPERTIES,
      has_issue: true,
      mine: { state: 'present', capsule_id: 'mine_exch-contradicted-01' },
      theirs: { state: 'present', capsule_id: 'theirs_exch-contradicted-01' },
      unilateral: false,
      timestamp: '2026-09-11T16:57:40Z',
      session_id: 'session-refactor-notes'
    },
    {
      exchange_key: 'exch-served-02',
      role_tag: 'SERVED',
      header_state: 'ok',
      properties: null,
      has_issue: false,
      mine: { state: 'present', capsule_id: 'mine_exch-served-02' },
      theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'not_asked' },
      unilateral: true,
      timestamp: '2026-09-11T16:55:12Z'
      // No session_id -- served rows never carry one (L-O), even in a
      // fixture built to demonstrate every other state.
    },
    {
      exchange_key: 'exch-refused-03',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: null,
      has_issue: false,
      mine: { state: 'present', capsule_id: 'mine_exch-refused-03' },
      theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'signed_refusal', evidence_outcome_date: '4 Sep' },
      unilateral: true,
      timestamp: '2026-09-11T09:30:00Z',
      session_id: 'session-trip-planning'
    },
    {
      exchange_key: 'exch-absent-04',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: null,
      has_issue: false,
      mine: { state: 'present', capsule_id: 'mine_exch-absent-04' },
      theirs: {
        state: 'absent',
        capsule_id: null,
        evidence_outcome: 'recorded_absence',
        evidence_outcome_date: '4 Sep'
      },
      unilateral: true,
      timestamp: '2026-09-11T09:15:00Z',
      session_id: 'session-trip-planning'
    },
    {
      exchange_key: 'exch-asked-05',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: null,
      has_issue: false,
      mine: { state: 'present', capsule_id: 'mine_exch-asked-05' },
      theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'unanswered', evidence_outcome_date: '3 Sep' },
      unilateral: true,
      timestamp: '2026-09-11T09:00:00Z',
      session_id: 'session-trip-planning'
    },
    {
      exchange_key: 'exch-not-asked-06',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: null,
      has_issue: false,
      mine: { state: 'present', capsule_id: 'mine_exch-not-asked-06' },
      theirs: { state: 'absent', capsule_id: null },
      unilateral: true,
      timestamp: '2026-09-11T08:03:00Z'
    },
    {
      exchange_key: 'exch-deleted-08',
      role_tag: 'ASKED',
      header_state: 'ok',
      properties: CLEAN_EXCHANGE_PROPERTIES,
      has_issue: false,
      mine: {
        state: 'present',
        capsule_id: 'mine_exch-deleted-08',
        deleted: true,
        deleted_date: '5 Sep'
      },
      theirs: { state: 'present', capsule_id: 'theirs_exch-deleted-08' },
      unilateral: false,
      timestamp: '2026-09-11T08:01:00Z'
    },
    ...buildFillerExchangeRows(HARNESS_LEDGER_FILLER_ROW_COUNT)
  ]
}

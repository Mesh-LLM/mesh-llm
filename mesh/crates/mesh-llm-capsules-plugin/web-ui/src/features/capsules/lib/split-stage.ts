// Split-inference stage records on the Evidence page
// (docs/DESIGN-split-stage-records.md §7). Pure, so every honesty rule
// below is unit-testable without mounting anything:
//   - stage 0 is the coordinator's own slice and never gets a ✓;
//   - a missing end, or a required lane empty on both ends, is a gap,
//     never agreement;
//   - the request's terminal state always sits beside the hand-offs line,
//     because agreement is not completion;
//   - `agree` says two stage records agree with each other, never that the
//     stages are independent or computed their slice correctly.
//
// `verifySplit` is the third implementation of the requester's check (the
// Rust producer's `stage_verify.rs` and the Python reference
// `mesh_split_stage.py` are the other two). All three run over
// `vectors/split-stage/`.

export type LaneFold = { frames: number; digest: string }
export type UpstreamHop = { received: LaneFold; replies_sent: LaneFold }
export type DownstreamHop = { sent: LaneFold; replies_received: LaneFold }
export type DirectReturn = { sent?: LaneFold; received?: LaneFold }

export type StageBlock = {
  v: number
  side: 'stage' | 'coordinator'
  coordinator_node_id: string
  run_id: string
  request_id: string
  topology_hash: string
  stage_index: number
  stage_count: number
  layer_start: number
  layer_end: number
  package_id: string
  manifest_sha256: string
  source_model_sha256: string
  return_mode: 'direct' | 'relayed'
  upstream?: UpstreamHop
  downstream?: DownstreamHop
  direct_return?: DirectReturn | null
  tokens?: { prefill_received: number; decode: number }
  terminal_state: string
  coordinator_term?: number
  data_path_observed?: boolean
  coordinator_observed?: { downstream?: DownstreamHop; direct_return?: DirectReturn }
}

export type StageAssignment = { node_id: string; layer_start: number; layer_end: number; package_id: string }
export type BundleRef = { type: string; digest_alg: string; digest: string }

export type SplitReceipt = {
  v: number
  kind: string
  run_id: string
  topology: { seq: number; hop_id: string; role: string; observation_point: string | null; assignment: StageAssignment }[]
  stages: { hop_id: string; bundle: string; bundle_ref?: BundleRef; bundle_refs?: BundleRef[] }[]
  coordinator_node_id: string
  request_id: string
  stage_seal_deadline_ms: number
}

export type CellState =
  | 'coordinator_slice'
  | 'ok'
  | 'disagrees'
  | 'not_received'
  | 'not_requested'
  | 'conflict'
  | 'rejected'
export type HopState = 'agree' | 'gap' | 'break' | 'malformed' | 'not_applicable'
export type Lane = 'forward' | 'reply' | 'direct_return'

export type SplitVerdict = {
  cells: { stage_index: number; state: CellState }[]
  lanes: { hop_index: number; lane: Lane; required: boolean; state: HopState }[]
  handoffs_agree: boolean
  terminal_state: string
}

export type CarriedStageRecord = { capsule_id: string; block: unknown }

export const TERMINAL_STATES = new Set([
  'completed',
  'policy_denied',
  'request_invalid',
  'backend_error',
  'transport_error',
  'client_cancelled',
  'timed_out',
  'evidence_unavailable'
])

const BLOCK_FIELDS = new Set([
  'v', 'side', 'coordinator_node_id', 'run_id', 'request_id', 'topology_hash', 'stage_index', 'stage_count',
  'layer_start', 'layer_end', 'package_id', 'manifest_sha256', 'source_model_sha256', 'return_mode', 'upstream',
  'downstream', 'direct_return', 'tokens', 'terminal_state', 'coordinator_term', 'data_path_observed',
  'coordinator_observed'
])
const REQUIRED_FIELDS = [
  'v', 'side', 'coordinator_node_id', 'run_id', 'request_id', 'topology_hash', 'stage_index', 'stage_count',
  'layer_start', 'layer_end', 'package_id', 'manifest_sha256', 'source_model_sha256', 'return_mode', 'terminal_state'
]
const EXCHANGE_ONLY = ['coordinator_term', 'data_path_observed', 'coordinator_observed']
const U64_MAX = 18446744073709551615n

type Obj = Record<string, unknown>

function isObj(value: unknown): value is Obj {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function present(block: Obj, key: string): boolean {
  return block[key] !== undefined && block[key] !== null
}

function isHex(value: unknown, length: number): boolean {
  return typeof value === 'string' && value.length === length && /^[0-9a-f]*$/.test(value)
}

function isPrefixed(value: unknown): boolean {
  return typeof value === 'string' && value.startsWith('sha256:') && isHex(value.slice(7), 64)
}

function isCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0
}

export function isCanonicalRequestId(raw: unknown): boolean {
  return typeof raw === 'string' && /^(0|[1-9][0-9]*)$/.test(raw) && BigInt(raw) <= U64_MAX
}

function exactKeys(value: unknown, keys: readonly string[]): value is Obj {
  return isObj(value) && Object.keys(value).length === keys.length && keys.every((k) => k in value)
}

function checkFold(value: unknown): string | null {
  if (!exactKeys(value, ['frames', 'digest'])) return 'a lane fold is exactly {frames, digest}'
  return isCount(value.frames) && isPrefixed(value.digest) ? null : 'a lane fold is a frame count and a sha256: digest'
}

function checkPair(value: unknown, keys: readonly [string, string]): string | null {
  if (!exactKeys(value, keys)) return `a hop is exactly {${keys.join(', ')}}`
  return checkFold(value[keys[0]]) ?? checkFold(value[keys[1]])
}

function checkDirect(value: unknown, want: 'sent' | 'received' | null): string | null {
  if (want === null) return value === undefined || value === null ? null : 'direct_return must be null here'
  if (!exactKeys(value, [want])) return `direct_return must carry ${want} only`
  return checkFold(value[want])
}

function checkTokens(value: unknown): string | null {
  if (!exactKeys(value, ['prefill_received', 'decode'])) return 'tokens is exactly {prefill_received, decode}'
  return isCount(value.prefill_received) && isCount(value.decode) ? null : 'token counts are non-negative integers'
}

/** `null` when `block` is a valid x-mesh-stage-v1 block, else why not. */
export function stageBlockError(block: unknown): string | null {
  if (!isObj(block)) return 'block must be an object'
  if (Object.keys(block).some((k) => !BLOCK_FIELDS.has(k))) return 'unknown member'
  if (REQUIRED_FIELDS.some((k) => !(k in block))) return 'missing member'
  if (block.v !== 1) return 'v must be 1'
  for (const k of ['coordinator_node_id', 'run_id', 'topology_hash']) {
    if (typeof block[k] !== 'string' || block[k] === '') return `${k} must be non-empty`
  }
  if (!isCanonicalRequestId(block.request_id)) return 'request_id must be a canonical decimal u64 string'
  for (const k of ['stage_index', 'stage_count', 'layer_start', 'layer_end']) {
    if (!isCount(block[k])) return `${k} must be a non-negative integer`
  }
  const k = block.stage_index as number
  const n = block.stage_count as number
  if (n < 2 || k >= n) return 'a split has stage_count >= 2 and stage_index < stage_count'
  if ((block.layer_end as number) <= (block.layer_start as number)) return 'layer_end is exclusive'
  if (!isPrefixed(block.package_id)) return 'package_id must be sha256: + 64 hex'
  if (!isHex(block.manifest_sha256, 64) || !isHex(block.source_model_sha256, 64)) return 'digests must be 64 hex'
  if (typeof block.terminal_state !== 'string' || !TERMINAL_STATES.has(block.terminal_state)) {
    return 'terminal_state is not in the closed set'
  }
  if (block.return_mode !== 'direct' && block.return_mode !== 'relayed') return 'return_mode must be direct or relayed'
  // Relayed return has no direct lane anywhere, so the position rules below
  // reject a non-null direct_return on every relayed block.
  const direct = block.return_mode === 'direct'
  const final = k + 1 === n
  const noExchangeFields = () => (EXCHANGE_ONLY.some((f) => present(block, f)) ? 'stage-exchange fields here' : null)
  if (block.side === 'stage') {
    if (k === 0) return 'a stage-side block starts at stage 1'
    return (
      noExchangeFields() ??
      (present(block, 'upstream') ? checkPair(block.upstream, ['received', 'replies_sent']) : 'upstream is required') ??
      (final
        ? present(block, 'downstream')
          ? 'the final stage has no downstream'
          : null
        : present(block, 'downstream')
          ? checkPair(block.downstream, ['sent', 'replies_received'])
          : 'downstream is required') ??
      checkDirect(block.direct_return, direct && final ? 'sent' : null) ??
      checkTokens(block.tokens)
    )
  }
  if (block.side === 'coordinator' && k === 0) {
    return (
      noExchangeFields() ??
      (present(block, 'upstream') ? 'stage 0 has no upstream' : null) ??
      (present(block, 'downstream') ? checkPair(block.downstream, ['sent', 'replies_received']) : 'downstream is required') ??
      checkDirect(block.direct_return, direct ? 'received' : null) ??
      checkTokens(block.tokens)
    )
  }
  if (block.side === 'coordinator') {
    if (['upstream', 'downstream', 'direct_return', 'tokens'].some((f) => present(block, f))) {
      return 'a stage-exchange record carries no hop fields or tokens'
    }
    if (!isCount(block.coordinator_term)) return 'coordinator_term is required'
    if (typeof block.data_path_observed !== 'boolean') return 'data_path_observed is required'
    const canObserve = k === 1 || (direct && final)
    if (block.data_path_observed !== canObserve) return 'data_path_observed is wrong for this position'
    const observed = block.coordinator_observed
    if (!canObserve) return present(block, 'coordinator_observed') ? 'coordinator_observed without observation' : null
    if (!isObj(observed) || Object.keys(observed).some((key) => key !== 'downstream' && key !== 'direct_return')) {
      return 'data_path_observed requires coordinator_observed'
    }
    if (k === 1) {
      if (!('downstream' in observed)) return "stage 1's exchange carries the coordinator's downstream"
      const error = checkPair(observed.downstream, ['sent', 'replies_received'])
      if (error) return error
    } else if ('downstream' in observed) {
      return "only stage 1's exchange carries the coordinator's downstream"
    }
    return checkDirect(observed.direct_return, direct && final ? 'received' : null)
  }
  return 'side must be stage or coordinator'
}

/** `null` when `receipt` is a valid split coordinator receipt, else why not. */
export function receiptError(receipt: unknown): string | null {
  const fields = ['v', 'kind', 'run_id', 'topology', 'stages', 'coordinator_node_id', 'request_id', 'stage_seal_deadline_ms']
  if (!exactKeys(receipt, fields)) return 'receipt members'
  if (receipt.v !== 1 || receipt.kind !== 'mesh-coordinator-receipt') return 'v / kind'
  if (!receipt.run_id || !receipt.coordinator_node_id) return 'run_id / coordinator_node_id'
  if (!isCanonicalRequestId(receipt.request_id)) return 'request_id'
  if (!isCount(receipt.stage_seal_deadline_ms)) return 'stage_seal_deadline_ms'
  const { topology, stages } = receipt
  if (!Array.isArray(topology) || !Array.isArray(stages) || topology.length < 2 || stages.length !== topology.length) {
    return 'a split has >= 2 stages and one stages[] entry per hop'
  }
  const cited: string[] = []
  for (let k = 0; k < topology.length; k++) {
    const hop = topology[k]
    const stage = stages[k]
    if (!isObj(hop) || !isObj(stage)) return 'entries must be objects'
    if (Object.keys(hop).some((key) => !['seq', 'hop_id', 'role', 'observation_point', 'assignment'].includes(key))) {
      return 'unknown topology member'
    }
    if (hop.seq !== k || hop.hop_id !== `stage-${k}` || hop.role !== (k === 0 ? 'coordinator' : 'stage')) {
      return `topology[${k}] order`
    }
    if (hop.observation_point !== undefined && hop.observation_point !== null && typeof hop.observation_point !== 'string') {
      return 'observation_point'
    }
    const a = hop.assignment
    if (!exactKeys(a, ['node_id', 'layer_start', 'layer_end', 'package_id'])) return `topology[${k}] assignment`
    if (typeof a.node_id !== 'string' || !a.node_id || !isCount(a.layer_start) || !isCount(a.layer_end)) {
      return 'assignment fields'
    }
    if (a.layer_end <= a.layer_start || !isPrefixed(a.package_id)) return 'assignment range / package'
    if (k === 0 && a.node_id !== receipt.coordinator_node_id) return 'stage 0 is the coordinator'
    if (stage.hop_id !== hop.hop_id) return 'stages[].hop_id'
    if (Object.keys(stage).some((key) => !['hop_id', 'bundle', 'bundle_ref', 'bundle_refs'].includes(key))) {
      return 'unknown stages member'
    }
    const refs = stageEntryRefs(k, stage)
    if (typeof refs === 'string') return refs
    cited.push(...refs)
  }
  return new Set(cited).size === cited.length ? null : 'one record cited under more than one hop'
}

function refOk(ref: unknown): ref is BundleRef {
  return (
    exactKeys(ref, ['type', 'digest_alg', 'digest']) &&
    typeof ref.type === 'string' &&
    ref.type.trim() !== '' &&
    ref.digest_alg === 'SHA-256' &&
    isHex(ref.digest, 64)
  )
}

/** The digests one stages[] entry cites, or why the entry is malformed. */
function stageEntryRefs(k: number, stage: Obj): string[] | string {
  const hasRef = present(stage, 'bundle_ref')
  const hasRefs = present(stage, 'bundle_refs')
  switch (stage.bundle) {
    case 'not_requested':
      return hasRef || hasRefs ? 'not_requested cites nothing' : []
    case 'absent':
      if (k === 0) return "stage 0's bundle is not_requested"
      return hasRef || hasRefs ? 'absent cites nothing' : []
    case 'present':
      if (k === 0) return "stage 0's bundle is not_requested"
      return !hasRefs && refOk(stage.bundle_ref) ? [stage.bundle_ref.digest] : 'present cites one record'
    case 'conflict': {
      if (k === 0) return "stage 0's bundle is not_requested"
      const refs = stage.bundle_refs
      if (hasRef || !Array.isArray(refs) || refs.length < 2 || !refs.every(refOk)) return 'conflict cites two or more'
      const digests = refs.map((r) => r.digest)
      return new Set(digests).size === digests.length ? digests : 'conflict cites distinct records'
    }
    default:
      return 'unknown bundle state'
  }
}

function compare(a: LaneFold | undefined, b: LaneFold | undefined, required: boolean): HopState {
  if (!a || !b) return 'gap'
  if (a.frames === 0 && b.frames === 0) return required ? 'gap' : 'not_applicable'
  if (a.digest === b.digest) return a.frames === b.frames ? 'agree' : 'malformed'
  return 'break'
}

const splitKeyOf = (b: { coordinator_node_id: string; run_id: string; request_id: string }) =>
  `${b.coordinator_node_id}\u0000${b.run_id}\u0000${b.request_id}`

/** The requester's check: stage cells, per-lane hand-offs, and the run line.
 *  Throws when the main record's own blocks are malformed. */
export function verifySplit(own: StageBlock, receipt: SplitReceipt, carried: readonly CarriedStageRecord[]): SplitVerdict {
  const ownError = stageBlockError(own) ?? receiptError(receipt)
  if (ownError) throw new Error(ownError)
  const key = splitKeyOf(receipt)
  if (own.side !== 'coordinator' || own.stage_index !== 0 || splitKeyOf(own) !== key) {
    throw new Error("own slice is not stage 0 of this receipt's split")
  }
  const n = receipt.topology.length
  const valid = new Map<string, StageBlock>()
  const rejected = new Set<string>()
  const idsByStage = new Map<number, Set<string>>()
  for (const record of carried) {
    if (stageBlockError(record.block) !== null) {
      rejected.add(record.capsule_id)
      continue
    }
    const block = record.block as StageBlock
    valid.set(record.capsule_id, block)
    if (block.side === 'stage' && splitKeyOf(block) === key) {
      const ids = idsByStage.get(block.stage_index) ?? new Set<string>()
      ids.add(record.capsule_id)
      idsByStage.set(block.stage_index, ids)
    }
  }
  const conflicted = new Set([...idsByStage].filter(([, ids]) => ids.size > 1).map(([k]) => k))

  const cells: SplitVerdict['cells'] = [{ stage_index: 0, state: 'coordinator_slice' }]
  const blocks: (StageBlock | null)[] = [own]
  for (let k = 1; k < n; k++) {
    const [state, block] = cellFor(k, receipt, own, valid, conflicted, rejected, key)
    cells.push({ stage_index: k, state })
    blocks.push(block)
  }

  const completed = own.terminal_state === 'completed'
  const relayed = own.return_mode === 'relayed'
  const lanes: SplitVerdict['lanes'] = []
  for (let h = 0; h < n - 1; h++) {
    const up = blocks[h]?.downstream
    const down = blocks[h + 1]?.upstream
    lanes.push({ hop_index: h, lane: 'forward', required: true, state: compare(up?.sent, down?.received, true) })
    const replyRequired = relayed && completed
    lanes.push({
      hop_index: h,
      lane: 'reply',
      required: replyRequired,
      state: compare(down?.replies_sent, up?.replies_received, replyRequired)
    })
  }
  if (!relayed) {
    lanes.push({
      hop_index: n - 1,
      lane: 'direct_return',
      required: completed,
      state: compare(blocks[n - 1]?.direct_return?.sent, own.direct_return?.received, completed)
    })
  }
  const handoffs_agree =
    lanes.every((l) => l.state === 'agree' || l.state === 'not_applicable') && lanes.some((l) => l.state === 'agree')
  return { cells, lanes, handoffs_agree, terminal_state: own.terminal_state }
}

function cellFor(
  k: number,
  receipt: SplitReceipt,
  own: StageBlock,
  valid: Map<string, StageBlock>,
  conflicted: Set<number>,
  rejected: Set<string>,
  key: string
): [CellState, StageBlock | null] {
  if (conflicted.has(k)) return ['conflict', null]
  const entry = receipt.stages[k]
  if (entry.bundle === 'not_requested') return ['not_requested', null]
  if (entry.bundle === 'conflict') return ['conflict', null]
  if (entry.bundle !== 'present' || !entry.bundle_ref) return ['not_received', null]
  const digest = entry.bundle_ref.digest
  if (rejected.has(digest)) return ['rejected', null]
  const block = valid.get(digest)
  if (!block) return ['not_received', null]
  if (block.side !== 'stage' || block.stage_index !== k || splitKeyOf(block) !== key || block.stage_count !== receipt.topology.length) {
    return ['rejected', null]
  }
  const assigned = receipt.topology[k].assignment
  const matches =
    block.layer_start === assigned.layer_start &&
    block.layer_end === assigned.layer_end &&
    block.package_id === assigned.package_id &&
    block.return_mode === own.return_mode
  return [matches ? 'ok' : 'disagrees', block]
}

// ---------------------------------------------------------------------------
// What the page shows
// ---------------------------------------------------------------------------

/** How a split row reaches the page: the records it needs for the check.
 *  `requester` and `coordinator` hold the coordinator's main record and the
 *  stage records it cites; a `stage` holds only its own block. */
export type SplitRowJson =
  | { viewer: 'requester' | 'coordinator'; main: Obj; stage_records: Obj[] }
  | { viewer: 'stage'; stage_block: StageBlock }

/** The console's bracket key, `split:<coordinator>/<run_id>/<request_id>`. */
export function splitBracketKey(b: { coordinator_node_id: string; run_id: string; request_id: string }): string {
  return `split:${b.coordinator_node_id}/${b.run_id}/${b.request_id}`
}

/** One role word per row, relative to the viewer (design §3). */
export function splitRoleWord(viewer: SplitRowJson['viewer']): string {
  switch (viewer) {
    case 'requester':
      return 'you asked'
    case 'coordinator':
      return 'you coordinated'
    case 'stage':
      return 'you ran a stage'
  }
}

/** Layers as a person reads them: upstream's end is exclusive, the page
 *  shows an inclusive range. */
export function layerRange(start: number, endExclusive: number): string {
  return `${start}–${endExclusive - 1}`
}

export type StripCell = {
  stageIndex: number
  nodeId: string
  layers: string
  state: CellState
  /** The one mark the cell carries. Stage 0 carries none. */
  mark: '' | '✓' | '✕' | '◌' | '—' | '⚠'
  words: string
}

const CELL_WORDS: Record<CellState, [StripCell['mark'], string]> = {
  coordinator_slice: ['', "coordinator's own slice"],
  ok: ['✓', 'matches its assignment'],
  disagrees: ['✕', 'disagrees with its assignment'],
  not_received: ['◌', 'not received'],
  not_requested: ['—', 'not part of the run'],
  conflict: ['⚠', 'two records for one stage'],
  rejected: ['✕', 'not a valid record of this stage']
}

export function stripCells(verdict: SplitVerdict, receipt: SplitReceipt): StripCell[] {
  return verdict.cells.map(({ stage_index, state }) => {
    const a = receipt.topology[stage_index].assignment
    const [mark, words] = CELL_WORDS[state]
    return { stageIndex: stage_index, nodeId: a.node_id, layers: layerRange(a.layer_start, a.layer_end), state, mark, words }
  })
}

export type HandoffItem = { label: string; state: 'agree' | 'gap' | 'break' | 'malformed'; words: string }

const HOP_RANK: Record<HopState, number> = { not_applicable: 0, agree: 1, gap: 2, malformed: 3, break: 4 }

/** One item per hop (its lanes folded to the worst state), then the direct
 *  return lane when the request used it. */
export function handoffItems(verdict: SplitVerdict, receipt: SplitReceipt): HandoffItem[] {
  const node = (k: number) => receipt.topology[k].assignment.node_id
  const byHop = new Map<string, HopState>()
  for (const lane of verdict.lanes) {
    const key = lane.lane === 'direct_return' ? 'direct' : String(lane.hop_index)
    const worst = byHop.get(key)
    if (worst === undefined || HOP_RANK[lane.state] > HOP_RANK[worst]) byHop.set(key, lane.state)
  }
  const items: HandoffItem[] = []
  for (const [key, state] of byHop) {
    // A lane the request did not need, empty on both ends, is not shown at
    // all -- neither as agreement nor as a gap.
    if (state === 'not_applicable') continue
    const label = key === 'direct' ? `${node(receipt.topology.length - 1)}→${node(0)} direct` : `${node(Number(key))}→${node(Number(key) + 1)}`
    items.push({ label, state, words: HANDOFF_WORDS[state] })
  }
  return items
}

const HANDOFF_WORDS: Record<HandoffItem['state'], string> = {
  agree: 'agree',
  gap: 'gap',
  break: 'break',
  malformed: 'malformed record'
}

/** `you asked · C · split across 3 nodes` */
export function splitRowHeadline(viewer: 'requester' | 'coordinator', receipt: SplitReceipt): string {
  return `${splitRoleWord(viewer)} · ${receipt.coordinator_node_id} · split across ${receipt.topology.length} nodes`
}

/** `you ran a stage · stage 2 of 3 · for C` -- the end requester is never
 *  shown: the stage was never told who it is. */
export function stageRowHeadline(block: StageBlock): string {
  return `${splitRoleWord('stage')} · stage ${block.stage_index + 1} of ${block.stage_count} · for ${block.coordinator_node_id}`
}

export type SplitView = {
  headline: string
  cells: StripCell[]
  handoffs: HandoffItem[]
  runAgrees: boolean
  terminalState: string
}

function blocksOf(record: Obj): Obj | null {
  const ma = record.model_attestation
  const ca = isObj(ma) ? ma.compute_attestation : undefined
  return isObj(ca) ? ca : null
}

/** Everything the strip renders for a requester's or coordinator's row, or
 *  `null` when the main record carries no split. */
export function splitView(split: Extract<SplitRowJson, { viewer: 'requester' | 'coordinator' }>): SplitView | null {
  const ca = blocksOf(split.main)
  const own = ca?.['x-mesh-stage-v1']
  const receipt = ca?.['x-mesh-coordinator-receipt-v1']
  if (stageBlockError(own) !== null || receiptError(receipt) !== null) return null
  const carried = split.stage_records.map((record) => ({
    capsule_id: typeof record.capsule_id === 'string' ? record.capsule_id : '',
    block: blocksOf(record)?.['x-mesh-stage-v1']
  }))
  const verdict = verifySplit(own as StageBlock, receipt as SplitReceipt, carried)
  const r = receipt as SplitReceipt
  return {
    headline: splitRowHeadline(split.viewer, r),
    cells: stripCells(verdict, r),
    handoffs: handoffItems(verdict, r),
    runAgrees: verdict.handoffs_agree,
    terminalState: verdict.terminal_state
  }
}

export type SplitPeer = { nodeId: string; splits: number }

/** "Nodes that served you through a split": every remote stage's node
 *  across the requester's split rows, never merged with direct dealings. */
export function peersThroughSplit(splits: readonly SplitRowJson[]): SplitPeer[] {
  const counts = new Map<string, number>()
  for (const split of splits) {
    if (split.viewer !== 'requester') continue
    const receipt = blocksOf(split.main)?.['x-mesh-coordinator-receipt-v1']
    if (receiptError(receipt) !== null) continue
    for (const hop of (receipt as SplitReceipt).topology.slice(1)) {
      counts.set(hop.assignment.node_id, (counts.get(hop.assignment.node_id) ?? 0) + 1)
    }
  }
  return [...counts].map(([nodeId, splits]) => ({ nodeId, splits })).sort((a, b) => a.nodeId.localeCompare(b.nodeId))
}

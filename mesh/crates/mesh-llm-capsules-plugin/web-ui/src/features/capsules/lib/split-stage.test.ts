// The page's split-stage logic over the fixtures the Rust producer and the
// reference implementation also run (`vectors/split-stage/`). Every
// expected verdict there is hand-written from the design.
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  handoffItems,
  layerRange,
  peersThroughSplit,
  receiptError,
  splitBracketKey,
  splitView,
  stageBlockError,
  stageRowHeadline,
  stripCells,
  verifySplit,
  type SplitReceipt,
  type SplitRowJson,
  type StageBlock
} from '@/features/capsules/lib/split-stage'

const FIXTURES = resolve(__dirname, '../../../../../vectors/split-stage')
const load = (name: string) => JSON.parse(readFileSync(resolve(FIXTURES, name), 'utf8'))

type HopCase = { name: string; own: StageBlock; receipt: SplitReceipt; carried: { capsule_id: string; block: unknown }[]; expect: unknown }
const HOPS: HopCase[] = load('hop-cases.json')
const BLOCKS: { name: string; block: unknown; valid: boolean }[] = load('block-cases.json')
const RECEIPTS: { name: string; receipt: unknown; valid: boolean }[] = load('receipt-cases.json')
const RUST_BUNDLE = load('rust-split-bundle.json')

const hop = (name: string) => HOPS.find((c) => c.name === name) as HopCase

describe('the same rules as the Rust producer and the Python reference', () => {
  it.each(BLOCKS.map((c) => [c.name, c] as const))('block %s', (_, c) => {
    expect(stageBlockError(c.block) === null).toBe(c.valid)
  })

  it.each(RECEIPTS.map((c) => [c.name, c] as const))('receipt %s', (_, c) => {
    expect(receiptError(c.receipt) === null).toBe(c.valid)
  })

  it.each(HOPS.map((c) => [c.name, c] as const))('hop check %s', (_, c) => {
    expect(verifySplit(c.own, c.receipt, c.carried)).toEqual(c.expect)
  })
})

describe('the stage strip', () => {
  it("never marks stage 0: it would only compare the coordinator's claim with its own assignment", () => {
    const c = hop('relayed_all_agree')
    const cells = stripCells(verifySplit(c.own, c.receipt, c.carried), c.receipt)
    expect(cells[0]).toMatchObject({ mark: '', words: "coordinator's own slice", nodeId: 'node-coordinator-c' })
    expect(cells.slice(1).map((cell) => cell.mark)).toEqual(['✓', '✓'])
  })

  it('shows layers inclusively, as a person reads them', () => {
    expect(layerRange(16, 32)).toBe('16–31')
    const c = hop('relayed_all_agree')
    expect(stripCells(verifySplit(c.own, c.receipt, c.carried), c.receipt).map((cell) => cell.layers)).toEqual([
      '0–15',
      '16–31',
      '32–47'
    ])
  })

  it('marks a missing stage not received and its hand-offs a gap, never agreement', () => {
    const c = hop('missing_end_absent')
    const verdict = verifySplit(c.own, c.receipt, c.carried)
    expect(stripCells(verdict, c.receipt)[2]).toMatchObject({ mark: '◌', words: 'not received' })
    expect(handoffItems(verdict, c.receipt)).toEqual([
      { label: 'node-coordinator-c→node-b', state: 'agree', words: 'agree' },
      { label: 'node-b→node-d', state: 'gap', words: 'gap' }
    ])
    expect(verdict.handoffs_agree).toBe(false)
  })

  it('folds a hop to its worst lane and lists the direct return lane on its own', () => {
    const c = hop('direct_all_agree')
    const items = handoffItems(verifySplit(c.own, c.receipt, c.carried), c.receipt)
    expect(items.map((i) => i.label)).toEqual([
      'node-coordinator-c→node-b',
      'node-b→node-d',
      'node-d→node-coordinator-c direct'
    ])
    const broken = hop('one_byte_changed')
    expect(handoffItems(verifySplit(broken.own, broken.receipt, broken.carried), broken.receipt)[1].state).toBe('break')
  })

  it('leaves out a direct lane the request never needed, rather than calling it a gap', () => {
    const c = structuredClone(hop('direct_all_agree'))
    c.own.terminal_state = 'timed_out'
    c.own.direct_return = { received: { ...(c.own.direct_return?.received as { digest: string }), frames: 0 } }
    const last = c.carried[c.carried.length - 1].block as StageBlock
    last.direct_return = { sent: { ...(last.direct_return?.sent as { digest: string }), frames: 0 } }
    const verdict = verifySplit(c.own, c.receipt, c.carried)
    expect(verdict.lanes.at(-1)).toMatchObject({ lane: 'direct_return', required: false, state: 'not_applicable' })
    expect(handoffItems(verdict, c.receipt).map((i) => i.label)).toEqual([
      'node-coordinator-c→node-b',
      'node-b→node-d'
    ])
  })

  it('keeps the terminal state beside the line: a truncated stream can agree', () => {
    const c = hop('stopped_early_agrees_on_truncated_stream')
    const verdict = verifySplit(c.own, c.receipt, c.carried)
    expect(verdict.handoffs_agree).toBe(true)
    expect(verdict.terminal_state).toBe('timed_out')
    expect(handoffItems(verdict, c.receipt).every((i) => i.state === 'agree')).toBe(true)
  })
})

describe('split rows', () => {
  it("reads the Rust plugin's own split bundle: every stage matches and the hand-offs agree", () => {
    const view = splitView({ viewer: 'requester', main: RUST_BUNDLE.capsule, stage_records: RUST_BUNDLE.split_stage_records })
    expect(view).not.toBeNull()
    expect(view?.headline).toBe('you asked · node-coordinator-c · split across 3 nodes')
    expect(view?.cells.map((c) => c.mark)).toEqual(['', '✓', '✓'])
    expect(view?.runAgrees).toBe(true)
    expect(view?.terminalState).toBe('completed')
  })

  it('a main record without the split blocks shows no strip', () => {
    expect(splitView({ viewer: 'requester', main: { model_attestation: {} }, stage_records: [] })).toBeNull()
  })

  it("names the stage's own row without the end requester", () => {
    const block = hop('relayed_all_agree').carried[1].block as StageBlock
    expect(stageRowHeadline(block)).toBe('you ran a stage · stage 3 of 3 · for node-coordinator-c')
  })

  it('keys every record of one split request the same way', () => {
    const c = hop('relayed_all_agree')
    expect(splitBracketKey(c.own)).toBe(splitBracketKey(c.carried[0].block as StageBlock))
    expect(splitBracketKey(c.own)).toBe('split:node-coordinator-c/mesh-split-1790000000000000000-g2/18446744073709551615')
  })
})

describe('who ran each stage is the coordinator’s word', () => {
  it('says so in every place the page names a stage node', async () => {
    const { SPLIT_TOOLTIPS } = await import('@/features/capsules/lib/tooltip-copy')
    expect(SPLIT_TOOLTIPS.stageCell).toMatch(/^The coordinator says which node ran each stage\./)
    expect(SPLIT_TOOLTIPS.throughSplit).toMatch(/^Nodes the coordinator says ran part of/)
    expect(SPLIT_TOOLTIPS.handoffs).toContain('not which node signed them')
    for (const copy of Object.values(SPLIT_TOOLTIPS)) {
      expect(copy).not.toMatch(/Each node signed/)
    }
  })
})

describe('Nodes that served you through a split', () => {
  it("counts each remote stage's node across the requester's splits, and never the coordinator", () => {
    const main = RUST_BUNDLE.capsule
    const splits: SplitRowJson[] = [
      { viewer: 'requester', main, stage_records: [] },
      { viewer: 'requester', main, stage_records: [] },
      { viewer: 'coordinator', main, stage_records: [] }
    ]
    expect(peersThroughSplit(splits)).toEqual([
      { nodeId: 'node-b', splits: 2 },
      { nodeId: 'node-d', splits: 2 }
    ])
  })
})

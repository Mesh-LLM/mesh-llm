import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StageStrip } from '@/features/capsules/components/StageStrip'
import type { SplitRowJson, StageBlock } from '@/features/capsules/lib/split-stage'
import { HANDOFF_TOOLTIPS, STAGE_CELL_TOOLTIPS } from '@/features/capsules/lib/tooltip-copy'

const FIXTURES = resolve(__dirname, '../../../../../vectors/split-stage')
type HopCase = { name: string; own: unknown; receipt: unknown; carried: { capsule_id: string; block: unknown }[] }
const HOPS: HopCase[] = JSON.parse(readFileSync(resolve(FIXTURES, 'hop-cases.json'), 'utf8'))

/** A requester's split row built from one shared hop case. */
function requesterRow(name: string): SplitRowJson {
  const c = HOPS.find((h) => h.name === name) as HopCase
  const blocks = (extra: Record<string, unknown>) => ({ model_attestation: { compute_attestation: extra } })
  return {
    viewer: 'requester',
    main: blocks({ 'x-mesh-stage-v1': c.own, 'x-mesh-coordinator-receipt-v1': c.receipt }),
    stage_records: c.carried.map((r) => ({ capsule_id: r.capsule_id, ...blocks({ 'x-mesh-stage-v1': r.block }) }))
  }
}

describe('StageStrip', () => {
  it('shows every stage, the hand-offs and the terminal state for an agreeing split', () => {
    render(<StageStrip split={requesterRow('relayed_all_agree')} />)
    expect(screen.getByText('you asked · node-coordinator-c · split across 3 nodes')).toBeInTheDocument()
    const cells = document.querySelectorAll('[data-stage-cell]')
    expect([...cells].map((c) => c.getAttribute('data-stage-cell'))).toEqual(['coordinator_slice', 'ok', 'ok'])
    expect(cells[0].textContent).not.toContain('✓')
    expect([...document.querySelectorAll('[data-handoff]')].map((h) => h.getAttribute('data-handoff'))).toEqual([
      'agree',
      'agree'
    ])
    expect(screen.getByText('request: completed')).toBeInTheDocument()
  })

  it("says on its face that each stage's node is the coordinator's word", () => {
    render(<StageStrip split={requesterRow('relayed_all_agree')} />)
    expect(screen.getByText('per coordinator:')).toBeInTheDocument()
  })

  it('says not received and gap for a missing stage, never agree', () => {
    render(<StageStrip split={requesterRow('missing_end_absent')} />)
    expect(screen.getByText(/not received/)).toBeInTheDocument()
    expect(document.querySelector('[data-handoff="gap"]')?.textContent).toContain('node-b→node-d gap')
    expect(document.querySelector('[data-handoff="gap"]')?.textContent).not.toContain('✓')
  })

  it('names a conflict and a break in words', () => {
    render(<StageStrip split={requesterRow('two_records_one_key')} />)
    expect(screen.getByText(/two records for one stage/)).toBeInTheDocument()
  })

  it('gives each cell and each hand-off the hover for its own state', () => {
    render(<StageStrip split={requesterRow('missing_end_absent')} />)
    const missing = document.querySelector('[data-stage-cell="not_received"]') as HTMLElement
    expect(missing).toHaveAccessibleDescription(STAGE_CELL_TOOLTIPS.not_received)
    const ok = document.querySelector('[data-stage-cell="ok"]') as HTMLElement | null
    if (ok) expect(ok).toHaveAccessibleDescription(STAGE_CELL_TOOLTIPS.ok)
    const gap = document.querySelector('[data-handoff="gap"]') as HTMLElement
    expect(gap).toHaveAccessibleDescription(HANDOFF_TOOLTIPS.gap)
  })

  it('shows a stopped request beside hand-offs that agree', () => {
    render(<StageStrip split={requesterRow('stopped_early_agrees_on_truncated_stream')} />)
    expect(screen.getByText('request: timed out')).toBeInTheDocument()
  })

  it("names a stage node's own row by its coordinator only", () => {
    const block = HOPS.find((h) => h.name === 'relayed_all_agree')?.carried[0].block as StageBlock
    render(<StageStrip split={{ viewer: 'stage', stage_block: block }} />)
    expect(screen.getByText('you ran a stage · stage 2 of 3 · for node-coordinator-c')).toBeInTheDocument()
  })

  it('renders nothing when the main record carries no split', () => {
    const { container } = render(
      <StageStrip split={{ viewer: 'requester', main: { model_attestation: {} }, stage_records: [] }} />
    )
    expect(container).toBeEmptyDOMElement()
  })
})

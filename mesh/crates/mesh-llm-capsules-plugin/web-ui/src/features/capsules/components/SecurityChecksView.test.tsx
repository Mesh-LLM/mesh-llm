import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { SecurityChecksView } from '@/features/capsules/components/SecurityChecksView'
import { sharedTheirsDetail } from '@/features/capsules/lib/security-checks-view'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import type { RecomputedIdentity } from '@/features/capsules/lib/recompute-identity'

function paneCRow(overrides: Partial<PaneCRow> = {}): PaneCRow {
  return {
    exchange_key: 'exch-cle',
    role_tag: 'ASKED',
    header_state: 'ok',
    properties: {
      task_binding: { state: 'PASS' },
      local_inclusion: { state: 'NOT_PRESENT' },
      checkpoint_signature: { state: 'NOT_PRESENT' },
      external_registration: { state: 'NOT_PRESENT' },
      continuity: { state: 'NOT_PRESENT' },
      identity_authority: { state: 'NOT_PRESENT' },
      capture_coverage: { state: 'PASS' },
      outcome_corroboration: { state: 'PASS' }
    },
    has_issue: false,
    mine: { state: 'present', capsule_id: 'mine-capsule-1' },
    theirs: { state: 'present', capsule_id: 'theirs-capsule-1' },
    unilateral: false,
    timestamp: '2026-09-08T16:58:05.412Z',
    ...overrides
  }
}

function ledgerRow(raw: PaneCRow): ExchangeLedgerRow {
  return {
    exchangeKey: raw.exchange_key,
    timestamp: raw.timestamp,
    roleTag: raw.role_tag,
    counterparty: 'node:aa11bb22',
    confirmed: true,
    localOnly: false,
    hasIssue: raw.has_issue,
    checksText: '—',
    rightCellState: { kind: 'closed', date: null },
    contentToggleState: { your: { kind: 'populated', date: null }, their: { kind: 'not_asked', date: null } },
    sessionId: null,
    twinBracketId: null,
    raw
  }
}

const RECOMPUTED_MATCH: RecomputedIdentity = { idMatch: true, signatureOk: true }

describe('SecurityChecksView — inline, never a modal', () => {
  it('carries no dialog/alertdialog role anywhere in the subtree', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
  })

  it('renders as an inline region, labelled with the exchange key', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    expect(screen.getByRole('region', { name: /exch-cle/ })).toBeInTheDocument()
  })
})

describe('SecurityChecksView — exact block order: IDENTITY -> HEADER -> WHAT IT COMMITS TO -> CHECKS -> raw', () => {
  it('block headings appear in that order in the DOM', () => {
    const { container } = render(
      <SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />
    )
    const headings = Array.from(container.querySelectorAll('[data-block-heading]')).map((el) =>
      el.getAttribute('data-block-heading')
    )
    expect(headings).toEqual(['identity', 'header', 'what it commits to', 'checks'])
  })

  it('raw bytes are last -- absent from the readable view, shown only behind the [raw] toggle', async () => {
    const user = userEvent.setup()
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    expect(screen.queryByText(/"mine"/)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'raw' }))
    expect(screen.getByText(/"mine"/)).toBeInTheDocument()
  })
})

describe('SecurityChecksView — two labelled groups, no counts', () => {
  it('renders "What this node said it did" and "What actually happened" as the two check-group headings', () => {
    const { container } = render(
      <SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />
    )
    const groupHeadings = Array.from(container.querySelectorAll('[data-check-group]')).map((el) =>
      el.getAttribute('data-check-group')
    )
    expect(groupHeadings).toEqual(['What this node said it did', 'What actually happened'])
  })

  it('outcome_corroboration is the only row under "What actually happened", always visible', () => {
    const { container } = render(
      <SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />
    )
    const actuallyHappenedHeading = Array.from(container.querySelectorAll('[data-check-group]')).find(
      (el) => el.getAttribute('data-check-group') === 'What actually happened'
    ) as HTMLElement
    const group = actuallyHappenedHeading.parentElement as HTMLElement
    const rowsAfterHeading = Array.from(group.children).slice(
      Array.from(group.children).indexOf(actuallyHappenedHeading) + 1
    )
    expect(rowsAfterHeading).toHaveLength(1)
    expect(within(rowsAfterHeading[0] as HTMLElement).getByText('outcome corroboration')).toBeInTheDocument()
  })
})

/** The chip cell whose sentence is `detail` (p2 item 2: the sentence lives
 *  in the chip's tooltip, carried on the cell as `data-detail`). */
function cellWithDetail(detail: string): HTMLElement {
  const cells = Array.from(document.querySelectorAll<HTMLElement>('[data-detail]')).filter(
    (el) => el.getAttribute('data-detail') === detail
  )
  expect(cells.length, `no chip carries "${detail}"`).toBeGreaterThan(0)
  return cells[0]
}

/** The text a chip's tooltip carries (its aria-describedby copy). */
function chipTooltip(cell: HTMLElement): string {
  const id = cell.querySelector('[aria-describedby]')?.getAttribute('aria-describedby')
  return (id ? document.getElementById(id)?.textContent : '') ?? ''
}

describe('SecurityChecksView — p2 item 2: chip-only rows, each sentence moved word for word into its chip’s tooltip', () => {
  it('every sentence is in its chip’s tooltip, never printed beside the chip', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    for (const detail of [
      'recomputed here',
      'No checkpoint covers this record — see Integrity.',
      'no checkpoint covers this record',
      'no receipt covers this record',
      'not linked to an owner'
    ]) {
      const cell = cellWithDetail(detail)
      expect(chipTooltip(cell)).toContain(detail)
      // Chip only on the face: the cell's visible text (screen-reader copy
      // aside) is the state word alone.
      const visible = cell.cloneNode(true) as HTMLElement
      for (const hidden of visible.querySelectorAll('.sr-only')) hidden.remove()
      expect(visible.textContent).not.toContain(detail)
    }
  })

  it('a row reads name -> your chip -> their chip', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    const line = document.querySelector('[data-check-line="content binding"]') as HTMLElement
    const [name, yours, theirs] = Array.from(line.children) as HTMLElement[]
    expect(name.textContent).toBe('content binding')
    expect(yours.querySelector('[data-detail]')).not.toBeNull()
    expect(theirs.querySelector('[data-detail]')).not.toBeNull()
  })

  it('the THEIRS sentence that repeats down the column is said once, as a column note, and left off those chips', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    const note = document.querySelector('[data-theirs-column-note]') as HTMLElement
    expect(note.textContent).toBe('not checked here')
    expect(screen.getAllByText('not checked here')).toHaveLength(1)
    const repeated = Array.from(document.querySelectorAll<HTMLElement>('[data-detail="not checked here"]'))
    expect(repeated.length).toBeGreaterThan(1)
    for (const cell of repeated) expect(chipTooltip(cell)).not.toContain('not checked here')
  })

  it('sharedTheirsDetail: only a sentence that actually repeats becomes the note', () => {
    const cell = (detail: string) => ({ state: 'NOT_CHECKED', label: 'not checked', detail, recomputed: false })
    const row = (key: string, detail: string | null) => ({
      key,
      label: key,
      group: 'g',
      yours: null,
      theirs: detail ? cell(detail) : null
    })
    expect(sharedTheirsDetail([row('a', 'x'), row('b', 'y')])).toBeNull()
    expect(sharedTheirsDetail([row('a', 'x'), row('b', 'x'), row('c', 'y'), row('d', null)])).toBe('x')
  })
})

describe('SecurityChecksView — L-M: recomputed-here vs from-sidecar are visually distinct', () => {
  it('content_binding/producer_signature carry a different class + data-source than a sidecar property', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    const recomputedCell = cellWithDetail('recomputed here')
    const sidecarCell = cellWithDetail('No checkpoint covers this record — see Integrity.')
    expect(recomputedCell).not.toBeNull()
    expect(sidecarCell).not.toBeNull()
    expect(recomputedCell?.getAttribute('data-source')).toBe('recomputed-in-browser')
    expect(sidecarCell?.getAttribute('data-source')).toBe('from-sidecar')
    expect(recomputedCell?.className).not.toBe(sidecarCell?.className)
  })

  it('says the source in words (no coloured bar), with the legend once under the header', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    const recomputedCell = cellWithDetail('recomputed here')
    expect(recomputedCell?.className).not.toMatch(/border-l-2/)
    expect(recomputedCell?.textContent).toContain('checked here')
    expect(screen.getAllByText(/checked here = your browser redid this check just now/)).toHaveLength(1)
  })
})

describe('SecurityChecksView — capture_coverage: sentence or not present, never a bare pass chip', () => {
  it('renders the fixed sentence when present', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    expect(
      screen.getByText('captured by the plugin at this node’s serving boundary (rule: every served exchange)')
    ).toBeInTheDocument()
  })

  it('renders "not present" when the sidecar sends nothing for it', () => {
    render(
      <SecurityChecksView
        identity={RECOMPUTED_MATCH}
        localRecord={null}
        row={ledgerRow(paneCRow({ properties: null }))}
      />
    )
    // Several other checkpoint-dependent rows are ALSO "not present" with
    // no properties at all -- scope to the label that precedes the
    // capture_coverage single line specifically.
    const captureCoverageLine = screen.getByText('capture coverage').closest('p')
    expect(captureCoverageLine?.textContent?.trim()).toBe('capture coverage not present')
  })
})

describe('SecurityChecksView — identity/authority: two facts rendered under one row', () => {
  it('renders both a "binding" and an "authority" fact, each with its own state', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    expect(screen.getByText('identity/authority · binding')).toBeInTheDocument()
    expect(screen.getByText('identity/authority · authority')).toBeInTheDocument()
    // p2 item 4: an empty owner reads "No owner set", never a literal "…".
    const authority = cellWithDetail('No owner set — not bound to a person.')
    expect(chipTooltip(authority)).toContain('No owner set — not bound to a person.')
    expect(document.body.textContent).not.toContain('Owner: …')
  })
})

describe('SecurityChecksView — every chip opens a four-part explanation', () => {
  it('clicking a chip reveals what it means / what this view found / what it does not establish / how to change it', async () => {
    const user = userEvent.setup()
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    await user.click(screen.getAllByText('established')[0])
    expect(screen.getByText(/^What this means:/)).toBeInTheDocument()
    expect(screen.getByText(/^What this view found:/)).toBeInTheDocument()
    expect(screen.getByText(/^What it does not establish:/)).toBeInTheDocument()
    expect(screen.getByText(/^How to change it:/)).toBeInTheDocument()
  })
})

describe('SecurityChecksView — rendered property set', () => {
  it('renders all ten manifesto properties, including task_binding', () => {
    const { container } = render(
      <SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />
    )
    const checksBlock = Array.from(container.querySelectorAll('[data-block-heading]')).find(
      (el) => el.getAttribute('data-block-heading') === 'checks'
    )?.parentElement as HTMLElement
    for (const label of [
      'content binding',
      'producer signature',
      'task binding',
      'local inclusion',
      'checkpoint signature',
      'external registration',
      'continuity',
      'identity/authority · binding',
      'capture coverage',
      'outcome corroboration'
    ]) {
      expect(within(checksBlock).getByText(label)).toBeInTheDocument()
    }
  })
})

describe('SecurityChecksView — WHAT IT COMMITS TO block: ✓ same only after a REAL fetch, never mirrored (finding 2, 2026-09-23 assessment)', () => {
  it('MUTANT-GUARD: a row whose right-cell reads CLOSED but carries no theirsRecompute never renders a mirrored "✓ same" -- the old bug mirrored `yours` the instant the row read CLOSED', () => {
    const { container } = render(
      <SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />
    )
    const commitsBlock = Array.from(container.querySelectorAll('[data-block-heading]')).find(
      (el) => el.getAttribute('data-block-heading') === 'what it commits to'
    )?.parentElement as HTMLElement
    expect(within(commitsBlock).queryByText('✓ same')).not.toBeInTheDocument()
  })

  it('once a real fetch holds the peer\'s own matching record, "✓ same" renders for the fields with a genuine peer-record equivalent (request digest, response digest, model identity) -- never for task binding/served by, which have none', () => {
    const localRecord = {
      effect: { request_digest: 'a'.repeat(64), response_digest: 'b'.repeat(64) },
      model_attestation: { model_id: 'model-x' }
    } as never
    const theirsRecompute = {
      status: 'found' as const,
      idMatch: true,
      signatureOk: true,
      peerRecord: {
        effect: { request_digest: 'a'.repeat(64), response_digest: 'b'.repeat(64) },
        model_attestation: { model_id: 'model-x' }
      },
      fetch: () => {}
    }
    const { container } = render(
      <SecurityChecksView
        identity={RECOMPUTED_MATCH}
        localRecord={localRecord}
        row={ledgerRow(paneCRow({ theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' } }))}
        theirsRecompute={theirsRecompute}
      />
    )
    const commitsBlock = Array.from(container.querySelectorAll('[data-block-heading]')).find(
      (el) => el.getAttribute('data-block-heading') === 'what it commits to'
    )?.parentElement as HTMLElement
    // request digest, response digest, model identity -- NOT task binding, NOT served by.
    expect(within(commitsBlock).getAllByText('✓ same')).toHaveLength(3)
  })

  it('an OPEN row (theirs absent) never renders a theirs value for "what it commits to" -- MUTANT: no ✓/✕ marker with nothing to compare', () => {
    const { container } = render(
      <SecurityChecksView
        identity={RECOMPUTED_MATCH}
        localRecord={null}
        row={ledgerRow(paneCRow({ theirs: { state: 'absent', capsule_id: null } }))}
      />
    )
    const commitsBlock = Array.from(container.querySelectorAll('[data-block-heading]')).find(
      (el) => el.getAttribute('data-block-heading') === 'what it commits to'
    )?.parentElement as HTMLElement
    expect(within(commitsBlock).getByText('request digest')).toBeInTheDocument()
    expect(within(commitsBlock).queryByText('✓ same')).not.toBeInTheDocument()
    expect(within(commitsBlock).queryByText('✕ differs')).not.toBeInTheDocument()
  })
})

describe('SecurityChecksView — per-row "Save evidence file"', () => {
  it('renders the 40-word evidence-file sentence and the save control', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    expect(screen.getByRole('button', { name: 'Save evidence file' })).toBeInTheDocument()
    expect(
      screen.getByText(
        'A file: what you asked, what you got, which machine and model answered, when it was registered, plus your own copy. Anyone can check it, no account needed. Not included: the text, or proof the answer was right.'
      )
    ).toBeInTheDocument()
  })

  it('p2 item 1: no inert "open in Logs" -- it cannot link yet, so it is not shown', () => {
    render(<SecurityChecksView identity={RECOMPUTED_MATCH} localRecord={null} row={ledgerRow(paneCRow())} />)
    expect(screen.queryByRole('button', { name: /open in Logs/i })).not.toBeInTheDocument()
    for (const button of screen.getAllByRole('button')) expect(button).not.toBeDisabled()
  })
})

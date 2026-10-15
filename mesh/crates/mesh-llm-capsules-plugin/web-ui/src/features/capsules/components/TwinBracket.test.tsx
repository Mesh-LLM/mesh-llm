// component-level enforcement of the
// observe-only/disclosure rules on top of the pure-function tests in
// `twin-bracket.test.ts`.
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { TwinBracket } from '@/features/capsules/components/TwinBracket'
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'

function twinRow(exchangeKey: string, overrides: Partial<PaneCRow> = {}): ExchangeLedgerRow {
  const raw: PaneCRow = {
    exchange_key: exchangeKey,
    role_tag: 'ASKED',
    header_state: 'ok',
    properties: null,
    has_issue: false,
    mine: { state: 'present', capsule_id: null },
    theirs: { state: 'present', capsule_id: null },
    unilateral: false,
    timestamp: '2026-09-14T00:00:00Z',
    twin_bracket_id: 'twin-xyz',
    ...overrides
  }
  return {
    exchangeKey: raw.exchange_key,
    timestamp: raw.timestamp,
    roleTag: raw.role_tag,
    counterparty: null,
    confirmed: false,
    localOnly: false,
    hasIssue: false,
    checksText: '—',
    rightCellState: { kind: 'closed', date: null },
    contentToggleState: { your: { kind: 'populated', date: null }, their: { kind: 'not_asked', date: null } },
    sessionId: null,
    twinBracketId: raw.twin_bracket_id ?? null,
    raw
  }
}

describe('TwinBracket — v3 §5, OBSERVE-ONLY', () => {
  it('renders the TWIN header naming the bracket id, and its two children', () => {
    const rows = [twinRow('a'), twinRow('b')]
    render(
      <TwinBracket bracketId="twin-xyz" rows={rows}>
        <p>row a</p>
        <p>row b</p>
      </TwinBracket>
    )
    expect(screen.getByText(/Side-by-side check/)).toBeInTheDocument()
    expect(screen.getByText(/same request, two machines/)).toBeInTheDocument()
    expect(screen.getByText('row a')).toBeInTheDocument()
    expect(screen.getByText('row b')).toBeInTheDocument()
  })

  it('LOAD-BEARING — item 4: never renders a verdict word (PASS/FAIL/identical/differs), only "not adjudicated"', () => {
    const rows = [
      twinRow('a', { mine: { state: 'present', capsule_id: null, text: 'same text' } }),
      twinRow('b', { mine: { state: 'present', capsule_id: null, text: 'same text' } })
    ]
    render(
      <TwinBracket bracketId="twin-xyz" rows={rows}>
        <p>row a</p>
        <p>row b</p>
      </TwinBracket>
    )
    expect(screen.getByText('not adjudicated')).toBeInTheDocument()
    expect(screen.getByText('Not adjudicated yet.')).toBeInTheDocument()
    expect(screen.queryByText(/\bPASS\b/)).not.toBeInTheDocument()
    expect(screen.queryByText(/\bFAIL\b/)).not.toBeInTheDocument()
    expect(screen.queryByText(/identical/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/\bdiffers\b/i)).not.toBeInTheDocument()
  })

  it('says a client marked the pair', () => {
    const rows = [twinRow('a'), twinRow('b')]
    render(
      <TwinBracket bracketId="twin-xyz" rows={rows}>
        <p>row a</p>
        <p>row b</p>
      </TwinBracket>
    )
    expect(
      screen.getByText('A client marked these two requests as one pair (x-mesh-twin-bracket).')
    ).toBeInTheDocument()
  })

  it('Compare is disabled when neither side has response text, and never renders as clickable-but-empty', () => {
    const rows = [twinRow('a'), twinRow('b')]
    render(
      <TwinBracket bracketId="twin-xyz" rows={rows}>
        <p>row a</p>
        <p>row b</p>
      </TwinBracket>
    )
    expect(screen.getByRole('button', { name: /Compare/ })).toBeDisabled()
  })

  it("Compare opens a real side-by-side diff of both peers' response text when both sides have it", async () => {
    const user = userEvent.setup()
    const rows = [
      twinRow('a', { mine: { state: 'present', capsule_id: null, text: 'hello from A' } }),
      twinRow('b', { mine: { state: 'present', capsule_id: null, text: 'hello from B' } })
    ]
    render(
      <TwinBracket bracketId="twin-xyz" rows={rows}>
        <p>row a</p>
        <p>row b</p>
      </TwinBracket>
    )
    const button = screen.getByRole('button', { name: /Compare/ })
    expect(button).toBeEnabled()
    await user.click(button)
    expect(screen.getByText('First answer')).toBeInTheDocument()
    expect(screen.getByText('Second answer')).toBeInTheDocument()
  })

  it('says "same answer" / "different answers" from the pane twin facts, and keeps "not adjudicated" until a referee signs one', () => {
    const facts = (same: boolean | null) => ({ bracket_id: 'twin-xyz', same_answer: same, other_row: null })
    const { unmount } = render(
      <TwinBracket bracketId="twin-xyz" rows={[twinRow('a', { twin: facts(true) }), twinRow('b', { twin: facts(true) })]}>
        <p>row a</p>
      </TwinBracket>
    )
    expect(screen.getByText('same answer')).toBeInTheDocument()
    expect(screen.getByText('not adjudicated')).toBeInTheDocument()
    unmount()
    const second = render(
      <TwinBracket bracketId="twin-xyz" rows={[twinRow('a', { twin: facts(false) }), twinRow('b', { twin: facts(false) })]}>
        <p>row a</p>
      </TwinBracket>
    )
    expect(screen.getByText('different answers')).toBeInTheDocument()
    second.unmount()
    render(
      <TwinBracket bracketId="twin-xyz" rows={[twinRow('a', { twin: facts(null) }), twinRow('b')]}>
        <p>row a</p>
      </TwinBracket>
    )
    expect(screen.getByText('not compared yet')).toBeInTheDocument()
  })

  it("shows the referee's signed verdict, and only then drops \"not adjudicated\"", () => {
    const withVerdict = {
      bracket_id: 'twin-xyz',
      same_answer: false,
      other_row: null,
      verdict: 'contradicted:2e981e80899b34248d9af76c6e28ed39',
      verdict_capsule_id: 'a9a0ca66822384be1f73188ca010620f'
    }
    render(
      <TwinBracket bracketId="twin-xyz" rows={[twinRow('a', { twin: withVerdict }), twinRow('b')]}>
        <p>row a</p>
      </TwinBracket>
    )
    expect(screen.getByText(/referee: contradicted 2e981e8089/)).toBeInTheDocument()
    expect(screen.queryByText('not adjudicated')).not.toBeInTheDocument()
    expect(screen.getByText(/a9a0ca668223/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View the verdict' })).toBeInTheDocument()
    // The signature is checked only when the record is opened, so the face
    // never calls the verdict signed.
    expect(screen.getByText(/Referee's verdict: a9a0ca668223/)).toBeInTheDocument()
    expect(screen.getByTestId('twin-bracket-twin-xyz').textContent).not.toMatch(/signed verdict/)
  })

  it('a not-comparable ruling is a verdict, shown as not comparable, never as not adjudicated or a disagreement', () => {
    render(
      <TwinBracket
        bracketId="twin-xyz"
        rows={[twinRow('a', { twin: { bracket_id: 'twin-xyz', same_answer: false, other_row: null, verdict: 'not_comparable', verdict_capsule_id: 'v'.repeat(32) } }), twinRow('b')]}
       
      >
        <p>row a</p>
      </TwinBracket>
    )
    expect(screen.getByText('not comparable')).toBeInTheDocument()
    expect(screen.queryByText('not adjudicated')).not.toBeInTheDocument()
    expect(screen.queryByText(/contradicted/)).not.toBeInTheDocument()
  })

  it('offers no verdict to open while no referee has signed one', () => {
    render(
      <TwinBracket bracketId="twin-xyz" rows={[twinRow('a'), twinRow('b')]}>
        <p>row a</p>
      </TwinBracket>
    )
    expect(screen.queryByRole('button', { name: 'View the verdict' })).not.toBeInTheDocument()
  })
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import { RowVerdictLine } from '@/features/capsules/components/RowVerdictLine'

vi.mock('@/features/capsules/api/verdictClient', () => ({
  fetchVerdictRecord: vi.fn().mockResolvedValue({
    capsule: {
      model_attestation: {
        compute_attestation: {
          adjudication: { verdict: 'corroborated', half_a_capsule_id: 'a'.repeat(64), half_b_capsule_id: 'b'.repeat(64) }
        }
      }
    },
    signed_by_key_id: 'k'.repeat(64),
    verify_ok: true
  })
}))

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
}

function row(overrides: Partial<PaneCRow>): PaneCRow {
  return {
    exchange_key: 'x',
    role_tag: 'SERVED',
    header_state: 'ok',
    properties: null,
    has_issue: false,
    mine: { state: 'present', capsule_id: 'mine' },
    theirs: { state: 'absent', capsule_id: null },
    unilateral: true,
    timestamp: null,
    ...overrides
  }
}

describe('RowVerdictLine -- a verdict on the exchange it is about', () => {
  it('renders nothing when the row carries no verdict', () => {
    const { container } = render(<RowVerdictLine row={row({})} />, { wrapper })
    expect(container).toBeEmptyDOMElement()
  })

  it('on a judged node: says the referee found your answer contradicted, and opens the signed verdict', async () => {
    const user = userEvent.setup()
    render(
      <RowVerdictLine
        row={row({
          adjudication: {
            verdict: `contradicted:${'e'.repeat(64)}`,
            verdict_capsule_id: 'v'.repeat(64),
            referee_node_id: 'a70d3967bea3b22f'.repeat(4),
            received_at: '2026-09-28T15:00:00Z',
            about_this_node: true
          }
        })}
      />,
      { wrapper }
    )
    expect(screen.getByText('A referee (node a70d3967be…) found your answer contradicted.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'View the verdict' }))
    const dialog = await screen.findByRole('dialog')
    expect(await within(dialog).findByTestId('verdict-record-signature')).toHaveTextContent(
      'The referee’s signature checks on this node, and this node’s log records it.'
    )
    expect(within(dialog).getByText(/aaaaaaaaaa… and bbbbbbbbbb…/)).toBeInTheDocument()
  })

  it('on the referee node: the verdict it issued', () => {
    render(
      <RowVerdictLine
        row={row({ referee_call: true, adjudication_issued: { verdict: 'corroborated', verdict_capsule_id: 'v', halves: ['a', 'b'] } })}
      />,
      { wrapper }
    )
    expect(screen.getByText('You judged the two answers: corroborated.')).toBeInTheDocument()
  })
})

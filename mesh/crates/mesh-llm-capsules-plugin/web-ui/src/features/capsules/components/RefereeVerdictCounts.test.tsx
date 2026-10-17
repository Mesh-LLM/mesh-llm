import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { PaneBRow, RefereeVerdictCounts as Counts } from '@/features/capsules/api/sidecarTypes'
import { fetchVerdictRecord } from '@/features/capsules/api/verdictClient'
import { RefereeVerdictCounts } from '@/features/capsules/components/RefereeVerdictCounts'
import { HARNESS_PANE_B_PAYLOAD } from '@/features/capsules/lib/peer-fixtures'
import { REFEREE_VERDICTS_ABOUT_THEM } from '@/features/capsules/lib/tooltip-copy'
import { GRADING_WORDS } from '@/features/capsules/lib/banned-terms'

vi.mock('@/features/capsules/api/verdictClient', () => ({
  fetchVerdictRecord: vi.fn().mockResolvedValue({
    capsule: {
      model_attestation: {
        compute_attestation: { adjudication: { verdict: `contradicted:${'b'.repeat(64)}` } }
      }
    },
    signed_by_key_id: 'k'.repeat(64),
    verify_ok: true
  })
}))

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
}

const bucket = (...ids: string[]) => ({ count: ids.length, verdict_capsule_ids: ids })

function row(counts: Counts | undefined): PaneBRow {
  return { ...HARNESS_PANE_B_PAYLOAD.rows[0], referee_verdicts: counts }
}

const COUNTS: Counts = {
  corroborated: bucket('c'.repeat(64)),
  contradicted: bucket('1'.repeat(64), '2'.repeat(64)),
  inconclusive: bucket(),
  not_comparable: bucket('n'.repeat(64))
}

describe('RefereeVerdictCounts -- four counts, never one number', () => {
  it('shows all four counts side by side, zero included', () => {
    render(<RefereeVerdictCounts row={row(COUNTS)} />, { wrapper })
    expect(screen.getByTestId('referee-verdicts-corroborated')).toHaveTextContent('corroborated 1')
    expect(screen.getByTestId('referee-verdicts-contradicted')).toHaveTextContent('contradicted 2')
    expect(screen.getByTestId('referee-verdicts-inconclusive')).toHaveTextContent('inconclusive 0')
    expect(screen.getByTestId('referee-verdicts-not_comparable')).toHaveTextContent('not comparable 1')
    expect(screen.getByText(REFEREE_VERDICTS_ABOUT_THEM.explainer)).toBeInTheDocument()
    expect(screen.queryByText(new RegExp(`${GRADING_WORDS.join('|')}|\\d+%`, 'i'))).toBeNull()
  })

  it('a count opens its verdict records, and a record opens the signed verdict', async () => {
    const user = userEvent.setup()
    render(<RefereeVerdictCounts row={row(COUNTS)} />, { wrapper })
    await user.click(screen.getByTestId('referee-verdicts-contradicted'))
    const list = screen.getByRole('list', { name: 'contradicted verdicts' })
    const records = within(list).getAllByRole('button')
    expect(records).toHaveLength(2)
    await user.click(records[1])
    expect(await screen.findByTestId('verdict-record-signature')).toBeInTheDocument()
    expect(fetchVerdictRecord).toHaveBeenCalledWith('2'.repeat(64))
  })

  it('a zero count opens nothing', () => {
    render(<RefereeVerdictCounts row={row(COUNTS)} />, { wrapper })
    expect(screen.getByTestId('referee-verdicts-inconclusive')).toBeDisabled()
  })

  it('an older plugin that sends no counts says so, never four zeros', () => {
    render(<RefereeVerdictCounts row={row(undefined)} />, { wrapper })
    expect(screen.getByText(REFEREE_VERDICTS_ABOUT_THEM.notShown)).toBeInTheDocument()
    expect(screen.queryByTestId('referee-verdicts-contradicted')).toBeNull()
  })
})

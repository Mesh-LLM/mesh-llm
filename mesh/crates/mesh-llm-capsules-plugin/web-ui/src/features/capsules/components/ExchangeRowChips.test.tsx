import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ExchangeRowChips } from '@/features/capsules/components/ExchangeRowChips'
import type { ChecksRow } from '@/features/capsules/lib/security-checks-view'
import { ENTRY_CHIP_RESULT_TOOLTIPS } from '@/features/capsules/lib/tooltip-copy'

function checks(state: Record<string, string>): ChecksRow[] {
  return Object.entries(state).map(([key, wire]) => ({
    key,
    label: key,
    group: 'yours',
    yours: { state: wire, label: wire.toLowerCase(), detail: '' },
    theirs: null
  })) as ChecksRow[]
}

function renderChips(rows: ChecksRow[]) {
  render(
    <TooltipProvider>
      <ExchangeRowChips checks={rows} onChipActivate={vi.fn()} />
    </TooltipProvider>
  )
}

describe('ExchangeRowChips -- one tooltip per result, not one per chip', () => {
  it('a ✓, a ✗ and a – on the same chip each say what that result means', () => {
    renderChips(checks({ content_binding: 'PASS', producer_signature: 'FAIL' }))
    expect(screen.getByLabelText('words match: jump to that check')).toHaveAccessibleDescription(
      ENTRY_CHIP_RESULT_TOOLTIPS.content['✓']
    )
    expect(screen.getByLabelText('signed: jump to that check')).toHaveAccessibleDescription(
      ENTRY_CHIP_RESULT_TOOLTIPS.sig['✗']
    )
    expect(screen.getByLabelText('witnessed: jump to that check')).toHaveAccessibleDescription(
      ENTRY_CHIP_RESULT_TOOLTIPS.registered['–']
    )
  })

  it('every chip has a distinct text for ✓, ✗ and –', () => {
    for (const byMark of Object.values(ENTRY_CHIP_RESULT_TOOLTIPS)) {
      expect(new Set(Object.values(byMark)).size).toBe(3)
    }
  })

  it('never claims the checkpoint coverage or a witness receipt is checked on this page', () => {
    expect(ENTRY_CHIP_RESULT_TOOLTIPS.inclusion['✓']).toContain('doesn’t check that coverage itself')
    expect(ENTRY_CHIP_RESULT_TOOLTIPS.registered['✓']).toContain('isn’t checked on this page')
  })
})

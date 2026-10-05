import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { LogAuditEntry } from '@/features/logs/api/schemas'
import { LogEventInspector } from '@/features/logs/components/LogEventInspector'

const ENDPOINT_ID = '9f0c4cbe8cb7a8d5d577c20e50ef03fd2f63a2e7fd9897c155823bcbb281bb04'

function audit(overrides: Partial<LogAuditEntry> = {}): LogAuditEntry {
  return {
    entryId: 'audit-peer-1',
    occurredAt: '2026-08-08T12:01:00Z',
    source: 'mesh',
    code: 'gossip_policy_rejected',
    severity: 'warning',
    sequence: 7,
    subjectKind: 'mesh_peer',
    subjectId: ENDPOINT_ID,
    remoteAddr: '203.0.113.24:48712',
    pathType: 'direct',
    outcome: 'rejected',
    reasonCode: 'owner_attestation_required',
    ...overrides
  }
}

function renderInspector(entry: LogAuditEntry, auditEntries: readonly LogAuditEntry[] = [entry]) {
  return render(
    <LogEventInspector
      auditEntries={auditEntries}
      inspector={{ type: 'audit', id: entry.entryId }}
      onClose={vi.fn()}
      onRequestTabChange={vi.fn()}
      requestTab="overview"
    />
  )
}

describe('mesh audit inspector', () => {
  it('explains a gossip rejection with peer identity and a loaded-window repeat signal', () => {
    const entry = audit()
    const repeats = [
      entry,
      audit({ entryId: 'audit-peer-2', sequence: 6 }),
      audit({ entryId: 'audit-peer-3', sequence: 5 }),
      audit({ entryId: 'audit-other', sequence: 4, subjectId: 'another-peer' })
    ]

    renderInspector(entry, repeats)

    const dialog = screen.getByRole('dialog', { name: 'Operational event Peer rejected by policy' })
    expect(
      within(dialog).getByRole('heading', { name: 'Operational event Peer rejected by policy' })
    ).toHaveTextContent('Peer rejected by policy')
    expect(within(dialog).getByRole('button', { name: 'Close inspector' })).toBeInTheDocument()
    expect(dialog.querySelector('[data-log-category="gossip"]')).toHaveTextContent('Gossip')
    expect(within(dialog).getByText('warning', { exact: true })).toBeInTheDocument()

    const peer = within(dialog).getByRole('region', { name: 'Peer' })
    expect(peer).toHaveTextContent('9f0c…bb04')
    expect(peer).toHaveTextContent('203.0.113.24:48712')
    expect(peer).toHaveTextContent('Direct')
    expect(peer).toHaveTextContent('3 occurrences in the loaded window')
    expect(within(peer).getByRole('button', { name: 'Copy peer endpoint ID' })).toBeInTheDocument()

    const verdict = within(dialog).getByRole('region', { name: 'Verdict' })
    expect(verdict).toHaveTextContent('owner attestation')
    expect(verdict).toHaveTextContent('was not admitted to the mesh')
    expect(within(dialog).getByRole('heading', { name: 'Event metadata' })).toBeInTheDocument()
    expect(within(dialog).getByText('gossip_policy_rejected', { exact: true })).toBeInTheDocument()
  })

  it('states relay address limits honestly for QUIC failures', () => {
    const entry = audit({
      code: 'mesh_quic_handler_failed',
      pathType: 'relay',
      remoteAddr: undefined,
      outcome: 'failed',
      reasonCode: 'internal',
      durationMs: 2_412
    })

    renderInspector(entry)

    const dialog = screen.getByRole('dialog', { name: 'Operational event QUIC handler failed' })
    expect(within(dialog).getByRole('heading', { name: 'Operational event QUIC handler failed' })).toHaveTextContent(
      'QUIC handler failed'
    )
    expect(dialog.querySelector('[data-log-category="quic"]')).toHaveTextContent('QUIC')
    const peer = within(dialog).getByRole('region', { name: 'Peer' })
    expect(peer).toHaveTextContent('Connected via relay — no direct address observed')
    expect(peer).not.toHaveTextContent('203.0.113.24:48712')
    expect(within(dialog).getByRole('region', { name: 'Signals' })).toHaveTextContent('2.4s')
  })

  it('keeps legacy gossip rows explained without inventing a peer band', () => {
    const entry = audit({
      subjectKind: undefined,
      subjectId: undefined,
      remoteAddr: undefined,
      pathType: undefined,
      outcome: undefined,
      reasonCode: undefined
    })

    renderInspector(entry)

    const dialog = screen.getByRole('dialog', { name: 'Operational event Peer rejected by policy' })
    expect(within(dialog).queryByRole('region', { name: 'Peer' })).not.toBeInTheDocument()
    expect(within(dialog).getByRole('region', { name: 'Verdict' })).toHaveTextContent(
      'This older record does not include peer context.'
    )
  })
})

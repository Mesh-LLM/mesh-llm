import { describe, expect, it } from 'vitest'
import type { RecordsStatus } from '@/features/capsules/api/recordsClient'
import {
  CLEANUP_OPTIONS,
  NEW_LOG_PENDING,
  cleanupBlockedReason,
  cleanupResultMessage,
  exchangeTextNotice,
  heroStatusLine,
  lastCheckpointFact,
  nothingIsShared,
  recordsLocationFact,
  sharingRows,
  type HeroCounts
} from '@/features/capsules/lib/your-records'
import { SAMPLE_DATA_UNAVAILABLE } from '@/features/capsules/lib/tooltip-copy'

function status(overrides: Partial<RecordsStatus> = {}): RecordsStatus {
  return {
    records_path: '/data/ledger',
    record_count: 8,
    head: 'a'.repeat(64),
    log_id: 'capsules',
    stored_text_count: 0,
    new_history_pending: null,
    sharing: {
      record_at_completion: { value: 'counterparty', source: 'default' },
      history_segments: { value: 'counterparties', source: 'default' },
      adjudications: { value: 'deliver_to_subjects', source: 'default' },
      witness: { value: null, source: 'default' }
    },
    ...overrides
  }
}

describe('hero line 3', () => {
  it('says the whole tab in one sentence, with the witness posture', () => {
    expect(heroStatusLine({ records: 8, confirmed: 3, disagreements: 0, witness: 'off' })).toBe(
      '8 records · 3 confirmed by the other side · 0 disagreements · no outside witness (witness off — your choice)'
    )
    expect(heroStatusLine({ records: 1, confirmed: 0, disagreements: 1, witness: 'latest' })).toBe(
      '1 record · 0 confirmed by the other side · 1 disagreement · also held by a witness you don’t run'
    )
  })

  // The live run: a witness was set and held earlier checkpoints, the latest
  // line was a cut not offered yet, and the line said "witness off — your
  // choice". Only a witness that is not set reads as off.
  it('says the witness is off only when none is set', () => {
    const line = (witness: HeroCounts['witness']) =>
      heroStatusLine({ records: 2, confirmed: 1, disagreements: 0, witness }).split(' · ')[3]
    expect(line('earlier')).toBe('held by a witness you don’t run up to an earlier checkpoint')
    expect(line('pending')).toBe('no checkpoint held by a witness yet (witness on)')
    expect(line('unknown')).toBe('no outside witness')
    for (const state of ['latest', 'earlier', 'pending', 'unknown'] as const) {
      expect(line(state)).not.toMatch(/off|your choice/)
    }
  })
})

// Ported from the mesh-llm console fork (your-records.test.ts).
describe('What you share', () => {
  it('four switches in order, each with its setting, where it was set, and one sentence', () => {
    const rows = sharingRows(status())
    expect(rows.map((r) => r.key)).toEqual(['record_at_completion', 'history_segments', 'adjudications', 'witness'])
    expect(rows.map((r) => r.state)).toEqual([
      'to the other side',
      'to nodes you’ve dealt with',
      'to the node it’s about',
      'off'
    ])
    expect(rows.every((r) => r.source === 'default')).toBe(true)
    expect(rows[3].whatLeaves).toBe('Nothing is sent: witness off — your choice.')
  })

  it('reads a saved prospective as counterparties, which it always behaved as', () => {
    const row = (value: string) =>
      sharingRows(status({ sharing: { ...status().sharing, history_segments: { value, source: 'set' } } }))[1]
    expect(row('prospective').state).toBe(row('counterparties').state)
    expect(row('prospective').whatLeaves).toBe(row('counterparties').whatLeaves)
  })

  it('a set witness names where checkpoints go, and never records or text', () => {
    const rows = sharingRows(
      status({ sharing: { ...status().sharing, witness: { value: 'https://w.example', source: 'set' } } })
    )
    expect(rows[3]).toMatchObject({ state: 'on', source: 'set' })
    expect(rows[3].whatLeaves).toBe('Your checkpoints, never records or text, go to https://w.example.')
  })

  it('without a status, no setting is claimed', () => {
    const rows = sharingRows(null)
    expect(rows.every((r) => r.state === null && r.source === null)).toBe(true)
  })
})

describe('Clean up records', () => {
  it('offers exactly three choices, and none of them removes a record', () => {
    expect(CLEANUP_OPTIONS.map((o) => o.action)).toEqual(['delete_stored_text', 'rebuild_index', 'start_new_log'])
    for (const option of CLEANUP_OPTIONS) expect(option.title).not.toMatch(/delete (a|one|this) record/i)
  })

  it('never runs on sample data or without a node that answered; a new log needs the box ticked and none waiting', () => {
    expect(cleanupBlockedReason('rebuild_index', { sample: true, status: status(), confirmed: true })).toBe(
      SAMPLE_DATA_UNAVAILABLE
    )
    expect(cleanupBlockedReason('rebuild_index', { sample: false, status: null, confirmed: true })).not.toBeNull()
    expect(cleanupBlockedReason('rebuild_index', { sample: false, status: status(), confirmed: false })).toBeNull()
    expect(cleanupBlockedReason('start_new_log', { sample: false, status: status(), confirmed: false })).not.toBeNull()
    expect(cleanupBlockedReason('start_new_log', { sample: false, status: status(), confirmed: true })).toBeNull()
    const waiting = status({ new_history_pending: { requested_at: 't', closing_record_id: 'c' } })
    expect(cleanupBlockedReason('start_new_log', { sample: false, status: waiting, confirmed: true })).toBe(
      NEW_LOG_PENDING
    )
  })

  it('names the record each cleanup sealed, and says so when nothing was', () => {
    const sealed = { capsule_id: 'c'.repeat(64), record_number: 9, kind: 'x' }
    expect(cleanupResultMessage({ action: 'delete_stored_text', deleted_count: 3, sealed })).toBe(
      'Deleted the stored text of 3 records. Sealed as record 9.'
    )
    expect(cleanupResultMessage({ action: 'delete_stored_text', deleted_count: 0, sealed: null })).toBe(
      'There was no stored text to delete, so nothing was sealed.'
    )
    expect(cleanupResultMessage({ action: 'rebuild_index', records_checked: 8, sealed })).toMatch(/Checked 8 records/)
    expect(cleanupResultMessage({ action: 'start_new_log', sealed })).toMatch(/next time this node starts/)
  })
})

describe('Your records facts', () => {
  it('location and checkpoint in words', () => {
    expect(recordsLocationFact(status(), false)).toBe('/data/ledger')
    expect(recordsLocationFact(null, true)).toBe('Not shown on sample data.')
    expect(lastCheckpointFact(8, '2026-09-26T17:37:20.014Z')).toBe(
      'Covers 8 records, made no later than 2026-09-26T17:37:20.014Z.'
    )
    expect(lastCheckpointFact(0, null)).toBe('No checkpoint yet.')
  })
})

describe('Local only', () => {
  const off = { value: 'off', source: 'set' as const }
  const allOff = status({
    sharing: {
      record_at_completion: off,
      history_segments: off,
      adjudications: off,
      witness: { value: null, source: 'default' }
    }
  })

  it('holds only when every switch is off', () => {
    expect(nothingIsShared(allOff)).toBe(true)
  })

  it('does not hold with the referee on: it sends a prompt to a peer', () => {
    const refereeOn = {
      ...allOff,
      sharing: { ...allOff.sharing, adjudicate_differing_twins: { value: 'on', source: 'set' } }
    } as typeof allOff
    expect(nothingIsShared(refereeOn)).toBe(false)
  })

  it('does not hold with the defaults, with any one switch on, or without a status', () => {
    expect(nothingIsShared(status())).toBe(false)
    expect(
      nothingIsShared(
        status({ sharing: { ...allOff.sharing, record_at_completion: { value: 'counterparty', source: 'set' } } })
      )
    ).toBe(false)
    expect(
      nothingIsShared(status({ sharing: { ...allOff.sharing, history_segments: { value: 'peers', source: 'set' } } }))
    ).toBe(false)
    expect(
      nothingIsShared(
        status({ sharing: { ...allOff.sharing, adjudications: { value: 'deliver_to_subjects', source: 'set' } } })
      )
    ).toBe(false)
    expect(
      nothingIsShared(
        status({ sharing: { ...allOff.sharing, witness: { value: 'https://w.example', source: 'set' } } })
      )
    ).toBe(false)
    expect(nothingIsShared(null)).toBe(false)
  })
})

describe('witness URL on the page', () => {
  it('shows scheme://host/path only: never credentials, a query or a fragment', () => {
    const s = status()
    const witness = { value: 'https://user:secret@witness.example:8443/log/v1?token=abc#frag', source: 'set' as const }
    const row = sharingRows({ ...s, sharing: { ...s.sharing, witness } }).find((r) => r.key === 'witness')
    expect(row?.whatLeaves).toBe('Your checkpoints, never records or text, go to https://witness.example:8443/log/v1.')
    expect(row?.whatLeaves).not.toMatch(/user|secret|token|frag/)
  })
})

describe('exchangeTextNotice', () => {
  const base = {
    records_path: '/d',
    record_count: 1,
    head: null,
    log_id: 'l',
    stored_text_count: 0,
    new_history_pending: null,
    sharing: {} as never
  }

  it('says plainly that this node keeps exchange text, where, for how long, and that nothing is shared', () => {
    expect(exchangeTextNotice({ ...base, exchange_text: { kept: true, retention_days: 30 } })).toEqual({
      text: 'This node keeps the text of exchanges on its own disk (deleted after 30 days). Nothing is shared.',
      offerDelete: false
    })
    expect(exchangeTextNotice({ ...base, exchange_text: { kept: true, retention_days: 1 } })?.text).toContain(
      'after 1 day)'
    )
  })

  it('once keeping is off, still says so while kept text is on disk, and offers to delete it now', () => {
    expect(
      exchangeTextNotice({ ...base, stored_text_count: 3, exchange_text: { kept: false, retention_days: 30 } })
    ).toEqual({
      text: 'This node still holds 3 kept texts on its own disk (deleted after 30 days, or delete them now). Nothing is shared.',
      offerDelete: true
    })
    expect(
      exchangeTextNotice({ ...base, stored_text_count: 1, exchange_text: { kept: false, retention_days: 30 } })?.text
    ).toContain('holds 1 kept text on')
  })

  it('says nothing when no text is kept or held, or the plugin cannot say', () => {
    expect(exchangeTextNotice({ ...base, exchange_text: { kept: false, retention_days: 30 } })).toBeNull()
    expect(exchangeTextNotice({ ...base, stored_text_count: 2 })).toBeNull()
    expect(exchangeTextNotice(base)).toBeNull()
    expect(exchangeTextNotice(null)).toBeNull()
  })
})

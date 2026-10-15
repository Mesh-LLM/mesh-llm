import { describe, expect, it } from 'vitest'
import {
  buildRegistrationCopy,
  buildSetupSteps,
  CAPTURE_BOUNDARY_FACT,
  CHAIN_BAR_INFO,
  chainStripCaption,
  CHECKPOINTED_NOT_REGISTERED_STATUS,
  checkpointRegistration,
  CONTINUITY_NOT_ESTABLISHED,
  continuityFact,
  coveredRecordCount,
  sealedBreakdownText,
  identityFact,
  INTEGRITY_TILE_INFO,
  RETENTION_FACT,
  witnessKeyProblems,
  witnessRestartNote,
  witnessRows,
  witnessTileNote
} from '@/features/capsules/lib/integrity-view'
import { WITNESS_OFF } from '@/features/capsules/lib/tooltip-copy'
import { ownerLinked } from '@/features/capsules/lib/integrity-view'
import { formatExchangeTimestamp } from '@/features/capsules/lib/local-time'
import type { RecomputedIdentity } from '@/features/capsules/lib/recompute-identity'
import { buildChecksRows } from '@/features/capsules/lib/security-checks-view'
import { STANDING_WORD } from '@/features/capsules/lib/banned-terms'

describe('buildSetupSteps — ledger-ux-from-the-user §6, three steps in value order', () => {
  it('renders all three steps "not set up" / "none received yet" on a bare node (checkpoint_count REPORTED 0)', () => {
    // A bare node's host reports `checkpoint_count: 0` (capsule_panes_native's
    // build_pane_a always supplies the card) -- genuinely none yet, "not set up".
    const steps = buildSetupSteps({ checkpoint_count: 0 }, null)
    expect(steps.map((step) => step.title)).toEqual([
      'Have a witness hold your checkpoints',
      'Bind an owner identity',
      'Get the other side’s record'
    ])
    expect(steps.every((step) => !step.done)).toBe(true)
    expect(steps[0].status).toBe('not set up')
    expect(steps[1].status).toBe('not linked to an owner')
    expect(steps[2].status).toBe('none received yet')
    // Each explains what it buys and what it does not.
    expect(steps[0].body).toMatch(/does not make your records true/)
    expect(steps[1].body).toMatch(/only your own claim about who you are/)
    expect(steps[1].body).not.toMatch(/prove|`/)
    expect(steps[2].body).toMatch(/usually arrives on its own/)
    expect(steps[2].body).not.toMatch(/corroboration|half/i)
    // The retired "never asked" framing is gone -- a half can arrive by push.
    expect(steps[2].status).not.toMatch(/never asked/)
  })

  it('step 3 reflects the push-confirmed reality: a nonzero confirmed count marks it done, without repeating the tile’s count (finding 7)', () => {
    // The rung must not read "never asked" while
    // the tile reads CLOSED-BY-OTHER-SIDE N -- halves arrived by push.
    const steps = buildSetupSteps({ checkpoint_count: 0 }, null, 3)
    expect(steps[2].done).toBe(true)
    expect(steps[2].status).toBe('received')
    expect(steps[2].status).not.toMatch(/\d/)
    expect(steps[2].body).toBeNull()
  })

  it('checkpoints step reads a NULL card as "status not reported", NOT a false "not set up"', () => {
    // The costliest false absence: a null card means the host did not REPORT a
    // checkpoint count -- it must never render as "none exists". Four states,
    // never two: null -> not reported; 0 -> not set up; >0 unwitnessed ->
    // checkpointed locally; witnessed -> registered.
    const notReported = buildSetupSteps(null, null)
    expect(notReported[0].status).toBe('status not reported')
    expect(notReported[0].done).toBe(false)
    expect(notReported[0].body).toMatch(/did not report its checkpoint status/)
    expect(notReported[0].body).not.toMatch(/does not make your records true/)

    const genuinelyNone = buildSetupSteps({ checkpoint_count: 0 }, null)
    expect(genuinelyNone[0].status).toBe('not set up')

    const checkpointedOnly = buildSetupSteps({ checkpoint_count: 2 }, null)
    expect(checkpointedOnly[0].status).toBe(CHECKPOINTED_NOT_REGISTERED_STATUS)
  })

  it('D1: a local checkpoint with ZERO witnesses NEVER reads "witnessed" — the exact §7 overclaim', () => {
    // The live shot: rung 1 said "registered" while the same pane said
    // "Registered with 0 witnesses" and Exchanges said "not registered".
    // Local checkpointing is NOT registration.
    const steps = buildSetupSteps({ checkpoint_count: 3, witnesses: [] }, null)
    expect(steps[0].done).toBe(false)
    expect(steps[0].status).toBe('checkpointed locally · no witness')
    expect(steps[0].status).not.toBe('witnessed')
    // The step still explains what registration would buy -- the reader is
    // exactly the person deciding whether to do it.
    expect(steps[0].body).toMatch(/does not make your records true/)
  })

  it('step 1 flips to "witnessed" ONLY once a witness actually holds a checkpoint, and drops its explanatory body', () => {
    const steps = buildSetupSteps({ checkpoint_count: 3, witnesses: [{ checked: true }] }, null)
    expect(steps[0].done).toBe(true)
    expect(steps[0].status).toBe('witnessed')
    expect(steps[0].body).toBeNull()
  })

  it('rung 1, the witness line, and the Exchanges headline fact all derive from the ONE checkpointRegistration fact', () => {
    // Three surfaces, one derivation -- they can never disagree again.
    const unwitnessed = { checkpoint_count: 5, witnesses: [] }
    const registration = checkpointRegistration(unwitnessed)
    expect(registration.registered).toBe(false)
    expect(registration.checkpointedLocally).toBe(true)
    expect(buildSetupSteps(unwitnessed, null)[0].status).toContain('no witness')
    expect(buildRegistrationCopy(unwitnessed)?.witnessSummary).toContain('no witness')

    const witnessed = { checkpoint_count: 5, witnesses: [{ checked: true }] }
    expect(checkpointRegistration(witnessed).registered).toBe(true)
    expect(buildSetupSteps(witnessed, null)[0].status).toBe('witnessed')
    expect(buildRegistrationCopy(witnessed)?.witnessSummary).toMatch(/^Held by 1 witness/)
  })

  it('step 2 says what binding established once the live owner is verified -- never "bound" alone (UX §4)', () => {
    const steps = buildSetupSteps(null, { status: 'verified', verified: true })
    expect(steps[1].done).toBe(true)
    expect(steps[1].status).toBe('linked (self-asserted)')
    expect(steps[1].body).toBe(
      'Your records are signed by this node’s key, linked to your owner account (self-asserted).'
    )
  })

  it('p2 item 3: an owner the host did NOT verify is not linked -- every non-verified status the host sends', () => {
    // The look: status.owner was { status: "unsigned", verified: false } and
    // Integrity said "linked", while the checks panel said not bound.
    for (const status of ['unsigned', 'expired', 'invalid_signature', 'revoked_owner', 'untrusted_owner']) {
      expect(ownerLinked({ status, verified: false })).toBe(false)
      expect(buildSetupSteps(null, { status, verified: false })[1].status).toBe('not linked to an owner')
    }
    expect(ownerLinked({ status: 'verified', verified: true })).toBe(true)
  })

  it('p2 item 3: the checks panel binding fact and Integrity step 2 say the same thing from the same owner', () => {
    const binding = (linked: boolean) =>
      buildChecksRows(
        {
          exchange_key: 'e',
          role_tag: 'ASKED',
          header_state: 'ok',
          properties: null,
          has_issue: false,
          mine: { state: 'present', capsule_id: 'm' },
          theirs: { state: 'absent', capsule_id: null },
          unilateral: true,
          timestamp: null
        },
        { idMatch: null, signatureOk: null } as RecomputedIdentity,
        undefined,
        { ownerLinked: linked }
      )
        .find((r) => r.key === 'identity_authority')
        ?.facts?.find((f) => f.factLabel === 'binding')?.cell
    const unsigned = { status: 'unsigned', verified: false }
    expect(binding(ownerLinked(unsigned))?.detail).toBe(buildSetupSteps(null, unsigned)[1].status)
    const verified = { status: 'verified', verified: true }
    expect(binding(ownerLinked(verified))?.state).toBe('PASS')
    expect(buildSetupSteps(null, verified)[1].body).toContain(binding(ownerLinked(verified))?.detail as string)
  })

  it('step 2 stays "not linked to an owner" when the wire carries no owner at all', () => {
    const steps = buildSetupSteps(null, undefined)
    expect(steps[1].done).toBe(false)
  })

  it('step 3 flips to "asked <date>" once the card carries an ask record', () => {
    const steps = buildSetupSteps({ asked_peer_at: '2026-09-12' }, null)
    expect(steps[2].done).toBe(true)
    expect(steps[2].status).toBe('asked 2026-09-12')
    expect(steps[2].body).toBeNull()
  })
})

describe('checkpointRegistration — the witness state', () => {
  // The live run's card: a witness set, the latest line a cut no witness
  // holds yet, an earlier one held.
  it('a witness holding an earlier checkpoint is witnessed, never "off"', () => {
    const live = checkpointRegistration({
      checkpoint_count: 3,
      witnesses: [],
      witnessed_checkpoint_count: 1,
      witness_configured: true
    })
    expect(live.registered).toBe(false)
    expect(live.witnessState).toBe('earlier')
    expect(live.witnessedCheckpointCount).toBe(1)
    expect(witnessTileNote(live.witnessState)).toBe('the latest checkpoint is not held yet')
    expect(buildSetupSteps({ checkpoint_count: 3, witnesses: [], witnessed_checkpoint_count: 1 }, null)[0]).toMatchObject({
      done: true,
      status: 'witnessed'
    })
  })

  it('reads off, set-but-not-yet and unknown apart', () => {
    const state = (card: Record<string, unknown>) =>
      checkpointRegistration({ checkpoint_count: 1, witnesses: [], ...card }).witnessState
    expect(state({ witness_configured: false })).toBe('off')
    expect(state({ witness_configured: true })).toBe('pending')
    expect(state({})).toBe('unknown')
    expect(state({ witnesses: [{ checked: true }], witness_configured: false })).toBe('latest')
    expect(witnessTileNote('off')).toBe(WITNESS_OFF)
    expect(witnessTileNote('pending')).not.toMatch(/off/)
    expect(witnessTileNote('latest')).toBeUndefined()
    expect(witnessTileNote('unknown')).toBeUndefined()
  })
})

describe('buildRegistrationCopy — only renders once a checkpoint exists', () => {
  it('is null when no checkpoint exists', () => {
    expect(buildRegistrationCopy(null)).toBeNull()
    expect(buildRegistrationCopy({ checkpoint_count: 0 })).toBeNull()
    expect(buildRegistrationCopy({ checkpoint_count: null })).toBeNull()
  })

  it('renders "Held by N witnesses: <names> (M not operated by this node)"', () => {
    const copy = buildRegistrationCopy({
      checkpoint_count: 2,
      witnesses: [
        { ts_url: 'https://a.example/log', operated_by_producer: true, checked: true },
        { ts_url: 'https://b.example', operated_by_producer: false, checked: true },
        { ts_url: 'https://c.example', checked: true }
      ]
    })
    expect(copy).not.toBeNull()
    expect(copy?.witnessSummary).toBe('Held by 3 witnesses: a.example, b.example, c.example (2 not operated by this node)')
  })

  it('a receipt that did not check is not counted, and never says "held"', () => {
    const card = {
      checkpoint_count: 2,
      witness_configured: true,
      witnesses: [{ ts_url: 'https://a.example', checked: false, check: 'the receipt does not verify' }]
    }
    expect(checkpointRegistration(card).registered).toBe(false)
    expect(checkpointRegistration(card).witnessState).toBe('pending')
    expect(buildRegistrationCopy(card)?.witnessSummary).not.toMatch(/Held by/)
  })

  it('lists every witness by name with what it holds', () => {
    const rows = witnessRows({
      witness_status: [
        { name: 'a.example', url: 'https://a.example', state: 'latest', held_count: 3, checked_count: 3, key_source: 'configured' },
        { name: 'b.example', url: 'https://b.example', state: 'pending', held_count: 0, checked_count: 0, problem: 'refused the connection' },
        { name: 'c.example', url: 'https://c.example', state: 'unchecked', held_count: 1, checked_count: 0, problem: 'its key is not fetched yet' },
        { name: 'd.example', url: 'https://d.example', state: 'earlier', held_count: 2, checked_count: 2, key_source: 'pinned_on_first_contact' }
      ]
    })
    expect(rows.map((row) => [row.name, row.status, row.tone])).toEqual([
      ['a.example', 'holds the latest checkpoint', 'good'],
      ['b.example', 'none held yet', 'pending'],
      ['c.example', '1 receipt not checked', 'bad'],
      ['d.example', 'holds an earlier checkpoint', 'good']
    ])
    expect(rows[1].detail).toBe('refused the connection')
    expect(rows[2].detail).toBe('its key is not fetched yet')
    expect(rows.map((row) => row.key)).toEqual([
      'key configured',
      null,
      null,
      'key pinned on first contact, not configured'
    ])
    expect(witnessRows({})).toEqual([])
  })

  it('a witness with no operated_by_producer field counts toward M (errs independent-claim-is-wrong)', () => {
    const copy = buildRegistrationCopy({ checkpoint_count: 1, witnesses: [{ ts_url: 'https://w.example', checked: true }] })
    expect(copy?.witnessSummary).toBe('Held by 1 witness: w.example (1 not operated by this node)')
  })

  it('D1: an unwitnessed checkpoint reads "checkpointed locally", NEVER "Held by 0 witnesses"', () => {
    const copy = buildRegistrationCopy({ checkpoint_count: 2, witnesses: [], witness_configured: false })
    expect(copy?.witnessSummary).toBe('Checkpointed locally · no witness (witness: off)')
    expect(copy?.witnessSummary).not.toMatch(/Held by 0/)
  })

  it('says "witness: off" only when no witness is set', () => {
    const summary = (card: Record<string, unknown>) =>
      buildRegistrationCopy({ checkpoint_count: 3, witnesses: [], ...card })?.witnessSummary
    expect(summary({ witnessed_checkpoint_count: 1, witness_configured: true })).toBe(
      'Latest checkpoint not held by a witness yet · an earlier one is'
    )
    expect(summary({ witnessed_checkpoint_count: 0, witness_configured: true })).toBe(
      'Checkpointed locally · no witness holds it yet (witness: on)'
    )
    expect(summary({})).toBe('Checkpointed locally · no witness')
  })

  it('adds "witnessed no later than T" (local time) only when a witness holds the checkpoint; unwitnessed says "checkpointed no later than"', () => {
    const registered = buildRegistrationCopy({
      checkpoint_count: 1,
      witnesses: [{ checked: true }],
      registered_no_later_than: '2026-09-10T00:00:00Z'
    })
    const local = formatExchangeTimestamp('2026-09-10T00:00:00Z')
    expect(registered?.registeredNoLaterThan).toBe(`witnessed no later than ${local}`)
    expect(registered?.registeredNoLaterThan).not.toMatch(/T00:00|Z\b/)

    // The card field name is the wire's; the COPY must not overclaim -- an
    // unwitnessed checkpoint's timestamp is a local fact, not a registration.
    const unwitnessed = buildRegistrationCopy({
      checkpoint_count: 1,
      witnesses: [],
      registered_no_later_than: '2026-09-10T00:00:00Z'
    })
    expect(unwitnessed?.registeredNoLaterThan).toBe(`checkpointed no later than ${local}`)

    const withoutDate = buildRegistrationCopy({ checkpoint_count: 1, witnesses: [] })
    expect(withoutDate?.registeredNoLaterThan).toBeNull()
  })
})

describe('coveredRecordCount — coverage in records, never padding leaves', () => {
  it('reads the records a checkpoint covers, not its leaf count', () => {
    // 8 leaves covered, of which 6 are padding: 2 records are covered.
    expect(coveredRecordCount({ covered_leaf_count: 8, covered_record_count: 2 })).toBe(2)
  })

  it('is unknown, never the leaf count, when the host reports no record count', () => {
    expect(coveredRecordCount({ covered_leaf_count: 8 })).toBeNull()
    expect(coveredRecordCount(null)).toBeNull()
  })

  it('so a padded checkpoint never reads unsealed records as sealed', () => {
    const covered = coveredRecordCount({ covered_leaf_count: 8, covered_record_count: 2 })
    expect(chainStripCaption(3, 1, covered)).not.toMatch(/^All 3/)
  })
})

describe('chainStripCaption — leaf pluralization + the three absence states', () => {
  it('partial coverage says how many of how many, and how many are unshaded (never "leaves")', () => {
    // 3rd arg is the covered-leaf count; 2nd is the checkpoint-LINE count.
    expect(chainStripCaption(5, 1, 1)).toBe(
      '1 of 5 records are sealed into a checkpoint · 4 since the last checkpoint are unshaded'
    )
    expect(chainStripCaption(5, 1, 4)).toBe(
      '4 of 5 records are sealed into a checkpoint · 1 since the last checkpoint is unshaded'
    )
  })

  it('full coverage reads "All N records are sealed into a checkpoint" (UX §4)', () => {
    expect(chainStripCaption(1, 1, 1)).toBe('The 1 record is sealed into a checkpoint')
  })

  it('renders the covered-leaf count, NOT the checkpoint-line count', () => {
    // The bug: a SINGLE checkpoint line covering 8 leaves read "1 leaves"
    // because the caption printed checkpoint_count. It must print the covered
    // leaf count (8), against the live-ledger reshoot: mmr_size 15 -> 8 leaves.
    expect(chainStripCaption(8, 1, 8)).toBe('All 8 records are sealed into a checkpoint')
  })

  it('says so honestly when a checkpoint exists but no covered-leaf count was reported', () => {
    // Never reprint the checkpoint-line count as if it were a leaf count.
    expect(chainStripCaption(5, 2, null)).toBe(
      'records sealed into a checkpoint (count not reported) · records since the last checkpoint are unshaded'
    )
  })

  it('keeps the three-state absence handling: null card is "not reported", never a false "no checkpoint yet"', () => {
    expect(chainStripCaption(2, null, null)).toBe('2 entries, all sealed · checkpoint status not reported')
    expect(chainStripCaption(1, 0, null)).toBe('1 entry, all sealed · no checkpoint yet · no witness holds any of it')
  })
})

describe('Item 4 — Integrity tile + chain-bar (i) copy: evidence, never a score', () => {
  it('names an evidence source in every tile line and refuses the score reading', () => {
    for (const info of Object.values(INTEGRITY_TILE_INFO)) {
      expect(info.length).toBeGreaterThan(0)
      // UX §8 rule 4: a banned word stays off the screen even to deny it.
      expect(info).not.toMatch(new RegExp(`\\b(score|rating|proven|${STANDING_WORD}|judgement)\\b`, 'i'))
    }
    // The witness tile names the witness; the closed tile names their signed
    // record, checked on this machine.
    expect(INTEGRITY_TILE_INFO.sharedWithWitness).toMatch(/witness/i)
    expect(INTEGRITY_TILE_INFO.confirmedByOtherSide).toMatch(/their own signed record, checked on this machine/)
  })

  it('the chain-bar (i) carries the checkpoint-coverage explanation (moved off the caption)', () => {
    expect(CHAIN_BAR_INFO).toBe(
      'Shaded: records sealed into a checkpoint. Unshaded: records sealed since the last checkpoint.'
    )
  })
})

describe('once-per-node facts — never fabricated, never per-row', () => {
  it('retention and capture-boundary are stated facts, not data claims', () => {
    expect(RETENTION_FACT).toMatch(/Retention:/)
    expect(CAPTURE_BOUNDARY_FACT).toMatch(/Capture boundary:/)
  })

  it('identity fact renders bound vs. not-bound, always ending "not bound to a person"', () => {
    expect(identityFact(null)).toBe('Owner: not bound — not bound to a person.')
    expect(identityFact({ status: 'verified', verified: true })).toBe(
      'Owner: linked (self-asserted) — not bound to a person.'
    )
  })

  it('continuity default prose names what would establish it: a PRIOR checkpoint, not registration', () => {
    expect(CONTINUITY_NOT_ESTABLISHED).toBe(
      'Continuity: not established. It needs a prior checkpoint for the next one to bind to.'
    )
    expect(CONTINUITY_NOT_ESTABLISHED).not.toMatch(/regist/i)
  })

  it('continuity is stated from the checkpoint count, separately from registration (UX §4)', () => {
    expect(continuityFact(null)).toBe(CONTINUITY_NOT_ESTABLISHED)
    expect(continuityFact(0)).toBe(CONTINUITY_NOT_ESTABLISHED)
    expect(continuityFact(1)).toBe(
      'Continuity: 1 checkpoint so far. The next one builds on it. A witness is what lets someone else check it too.'
    )
    // Never claims each checkpoint binds to the one before -- not reported.
    expect(continuityFact(3)).toBe(
      'Continuity: 3 checkpoints so far. A witness is what lets someone else check them too.'
    )
  })

  it('the Sealed tile reconciles records with exchanges (finding 3)', () => {
    expect(sealedBreakdownText(5, 3)).toBe('5 yours · 3 received from the other side')
    // A block and its undo are sealed, counted apart from exchanges.
    expect(sealedBreakdownText(5, 3, 2)).toBe('5 yours · 3 received from the other side · 2 routing choices')
    expect(sealedBreakdownText(5, 3, 1)).toBe('5 yours · 3 received from the other side · 1 routing choice')
  })
})

describe('banned vocabulary — never "timestamped", never "registered", and "witnessed" only when a witness holds it', () => {
  const allStrings = [
    RETENTION_FACT,
    CAPTURE_BOUNDARY_FACT,
    CONTINUITY_NOT_ESTABLISHED,
    identityFact(null),
    identityFact({ status: 'verified', verified: true }),
    ...buildSetupSteps(null, null).flatMap((step) => [step.title, step.status, step.body ?? '']),
    ...buildSetupSteps({ checkpoint_count: 1 }, { verified: true }).flatMap((step) => [
      step.title,
      step.status,
      step.body ?? ''
    ]),
    buildRegistrationCopy({ checkpoint_count: 1, witnesses: [{ checked: true }] })?.witnessSummary ?? ''
  ]

  it('never renders "timestamped"', () => {
    for (const text of allStrings) expect(text.toLowerCase()).not.toMatch(/timestamped/)
  })

  it('never renders "registered" (UX: jargon; the plain word is "witnessed")', () => {
    for (const text of allStrings) expect(text.toLowerCase()).not.toMatch(/\bregistered\b/)
  })

  it('says "witnessed" only when a witness holds a checkpoint, never for a local one', () => {
    const unwitnessed = { checkpoint_count: 1, witnesses: [], registered_no_later_than: '2026-09-10T00:00:00Z' }
    const local = [
      ...buildSetupSteps(unwitnessed, null).flatMap((step) => [step.title, step.status, step.body ?? '']),
      buildRegistrationCopy(unwitnessed)?.witnessSummary ?? '',
      buildRegistrationCopy(unwitnessed)?.registeredNoLaterThan ?? ''
    ]
    for (const text of local) expect(text.toLowerCase()).not.toMatch(/\bwitnessed\b/)
  })
})

describe('the witness list says when saved changes are not in force yet', () => {
  it('notes a restart only when the saved list differs from the one in use', () => {
    expect(witnessRestartNote({ witness_restart_needed: true })).toMatch(/takes effect when mesh-llm restarts/)
    expect(witnessRestartNote({ witness_restart_needed: false })).toBeNull()
    expect(witnessRestartNote({})).toBeNull()
  })
})

describe('configured witness key problems are shown', () => {
  it('lists each problem string, and nothing without any', () => {
    expect(witnessKeyProblems({ witness_key_problems: ['bad key'] })).toEqual(['bad key'])
    expect(witnessKeyProblems({})).toEqual([])
  })
})

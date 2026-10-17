// v3 §4: every chip opens a four-part
// explanation -- what it means / what this view found (the input already
// rendered inline as `detail`) / what it does not establish / how to
// change it. The fourth part collapses to a fixed "nothing to do" sentence
// for facts that are immutable (`identity_authority`'s `authority`, or any
// property already `established`) -- an absent state instead names the
// concrete step that would supply it, never a generic "not yet".
import type { ChecksSideCell } from '@/features/capsules/lib/security-checks-view'

export type ChipExplanation = {
  whatItMeans: string
  whatThisFound: string
  whatItDoesNotEstablish: string
  howToChange: string
}

const NOTHING_TO_DO = 'Nothing to do — this is a fact about the record, not a problem.'

type StaticExplanation = {
  meaning: string
  doesNotEstablish: string
  /** Called only when the cell is not already `established` (`PASS`). */
  howToSupply: string
}

const STATIC: Record<string, StaticExplanation> = {
  content_binding: {
    meaning: 'The record’s own fingerprint, computed again on this machine, equals its id: nothing in the record changed after it was sealed.',
    doesNotEstablish: 'Who produced the response, or that the request itself was answered honestly.',
    howToSupply: 'This check runs automatically on this machine once the record is loaded; there is nothing to ask for.'
  },
  producer_signature: {
    meaning: 'The signature over this record verifies against the key this node (or the other side) announces.',
    doesNotEstablish: 'That the signing key belongs to a specific, accountable person or organisation.',
    howToSupply: 'This check runs automatically on this machine once the record is loaded; there is nothing to ask for.'
  },
  task_binding: {
    meaning: 'The record names the specific task/model call it is an account of, not a generic placeholder.',
    doesNotEstablish: 'That the task description itself is accurate.',
    howToSupply: 'This comes from the sealed record as sent; there is no separate step to request it.'
  },
  local_inclusion: {
    meaning: 'A checkpoint this node signed covers this record. This page counts that coverage; it does not check the proof.',
    doesNotEstablish: 'That a witness you don’t run holds the checkpoint.',
    howToSupply: 'The node’s next checkpoint covers it (see Integrity).'
  },
  checkpoint_signature: {
    meaning: 'The checkpoint covering this record carries a valid signature.',
    doesNotEstablish: 'That the checkpoint has been witnessed by anyone other than this node.',
    howToSupply: 'The node’s next checkpoint covers this record (see Integrity).'
  },
  external_registration: {
    meaning: 'A witness you don’t run holds the checkpoint covering this record. The witness’s receipt isn’t checked on this page.',
    doesNotEstablish: 'That the record’s content, not just its checkpoint, was reviewed by the witness.',
    howToSupply: 'Turn on a witness in the plugin’s settings; it then holds copies of your checkpoints.'
  },
  continuity: {
    meaning: 'This checkpoint builds on the one before it, so a rewrite between them would show. Not checked on this page.',
    doesNotEstablish: 'That either checkpoint is true, only that a change between them would be visible.',
    howToSupply: 'Needs two checkpoints, the second building on the first (see Integrity).'
  },
  capture_coverage: {
    meaning: 'States the rule this node uses to decide which exchanges get captured at all.',
    doesNotEstablish: 'That every exchange the rule covers was captured without gaps.',
    howToSupply: 'This is a standing policy statement, not a per-record fact to request.'
  },
  outcome_corroboration: {
    meaning: 'The other side’s own signed record of this exchange agrees with yours.',
    doesNotEstablish: 'That either side’s account of the outcome is itself accurate.',
    howToSupply: 'Ask the other side for their record.'
  },
  identity_authority_binding: {
    meaning: 'A key is bound to this node’s identity for this record.',
    doesNotEstablish: 'That the key belongs to a specific accountable person (see authority, below).',
    howToSupply: 'Bind an owner identity to this node (see Integrity).'
  },
  identity_authority_authority: {
    meaning: 'Whether a real person or organisation stands behind this node’s identity.',
    doesNotEstablish: 'Anything about this specific record -- this is a standing fact about the node.',
    howToSupply: NOTHING_TO_DO
  }
}

/** `factKey` disambiguates `identity_authority`'s two facts; omit for every
 *  other property. */
export function explanationFor(key: string, cell: ChecksSideCell, factKey?: 'binding' | 'authority'): ChipExplanation {
  const lookupKey = factKey ? `identity_authority_${factKey}` : key
  const entry = STATIC[lookupKey]
  if (!entry) {
    throw new Error(`chip-explanation.ts: no static explanation registered for "${lookupKey}"`)
  }
  return {
    whatItMeans: entry.meaning,
    whatThisFound: cell.detail,
    whatItDoesNotEstablish: entry.doesNotEstablish,
    howToChange: cell.state === 'PASS' ? NOTHING_TO_DO : entry.howToSupply
  }
}

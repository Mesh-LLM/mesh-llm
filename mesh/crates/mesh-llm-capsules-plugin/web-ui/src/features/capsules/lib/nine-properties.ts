// The ten-property assurance map's labels (still named NINE_PROPERTY_LABELS
// for historical/import-site continuity -- the security view
// added `task_binding`, the manifesto's tenth-named-but-ninth-record-check,
// which this map had been shipping without: 8+1, not 9+1), shared between
// LedgerCard (the Peers/Exchanges card chip strip) and the Exchanges table's
// Checks column (Part 3) so the two
// never drift into different wording for the same properties -- verbatim
// order from `manifesto-the-tenth-check-v6`'s table.
export const NINE_PROPERTY_LABELS: Record<string, string> = {
  content_binding: 'content binding',
  producer_signature: 'producer signature',
  task_binding: 'task binding',
  local_inclusion: 'local inclusion',
  checkpoint_signature: 'checkpoint signature',
  external_registration: 'external registration',
  continuity: 'continuity',
  identity_authority: 'identity/authority',
  capture_coverage: 'capture coverage',
  outcome_corroboration: 'outcome corroboration'
}

// These two are ALWAYS recomputed in-browser — never trusted from sidecar.
export const RECOMPUTED_PROPERTIES = new Set(['content_binding', 'producer_signature'])

// Two labelled groups, no counts (v3 §4 /
// tab-design v2.1): nine properties are this node's own claim about the
// record; `outcome_corroboration` alone is the one axis that isn't this
// node's claim at all -- it is always visible, `not present` today (no
// counterparty ever asked). Never merge the two into one flat list again.
export const WHAT_NODE_SAID_GROUP = 'What this node said it did'
export const WHAT_ACTUALLY_HAPPENED_GROUP = 'What actually happened'

export const PROPERTY_GROUP: Record<string, string> = {
  content_binding: WHAT_NODE_SAID_GROUP,
  producer_signature: WHAT_NODE_SAID_GROUP,
  task_binding: WHAT_NODE_SAID_GROUP,
  local_inclusion: WHAT_NODE_SAID_GROUP,
  checkpoint_signature: WHAT_NODE_SAID_GROUP,
  external_registration: WHAT_NODE_SAID_GROUP,
  continuity: WHAT_NODE_SAID_GROUP,
  identity_authority: WHAT_NODE_SAID_GROUP,
  capture_coverage: WHAT_NODE_SAID_GROUP,
  outcome_corroboration: WHAT_ACTUALLY_HAPPENED_GROUP
}

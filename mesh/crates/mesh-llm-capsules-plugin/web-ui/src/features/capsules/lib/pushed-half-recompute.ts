// Per-row recompute for pushed counterparty halves. The pane hands the
// browser the held foreign body (`theirs.record`); before a pane query
// resolves, this recomputes each body's `capsule_id` HERE and records the
// result as `theirs.id_match`. Running it inside the query function (not in a
// component effect) means every caller that reads the gate synchronously --
// the Exchanges ledger, the Peers table, the headline and Integrity counts --
// sees the finished verdict on first render, so a closed row never renders
// OPEN first and then jumps to CLOSED.
import type { PaneBJson, PaneCListJson, PaneCRow } from '@/features/capsules/api/sidecarTypes'
import { recomputeIdMatch } from '@/features/capsules/lib/recompute-identity'

type Theirs = PaneCRow['theirs']

async function withIdMatch(theirs: Theirs): Promise<Theirs> {
  if (!theirs.record) return theirs
  const idMatch = await recomputeIdMatch(theirs.record, theirs.capsule_id ?? null)
  return { ...theirs, id_match: idMatch }
}

export async function attachPushedHalfRecomputeToPaneC(payload: PaneCListJson): Promise<PaneCListJson> {
  const rows = await Promise.all(
    (payload.rows ?? []).map(async (row) =>
      row.theirs.record ? { ...row, theirs: await withIdMatch(row.theirs) } : row
    )
  )
  return { ...payload, rows }
}

export async function attachPushedHalfRecomputeToPaneB(payload: PaneBJson): Promise<PaneBJson> {
  const rows = await Promise.all(
    (payload.rows ?? []).map(async (row) => {
      const siblings = row.confirmed_siblings
      if (!siblings?.some((sibling) => sibling.theirs.record)) return row
      const confirmed_siblings = await Promise.all(
        siblings.map(async (sibling) => ({ ...sibling, theirs: await withIdMatch(sibling.theirs) }))
      )
      return { ...row, confirmed_siblings }
    })
  )
  return { ...payload, rows }
}

// Fixture pieces for a correlated pushed half, shaped as the pane sends it
// (`capsule_panes_native.rs::theirs_sibling_cell` + `mine_pair_cell`), with
// the browser's `capsule_id` recompute already recorded (`id_match`). Shared by
// the harness fixtures and the component tests so every surface is pinned to
// the same gate inputs. The recompute itself is exercised against real ids in
// `pushed-half-recompute.test.ts`.
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'

export const FIXTURE_REQUEST_DIGEST = 'a'.repeat(64)
export const FIXTURE_RESPONSE_DIGEST = 'b'.repeat(64)
export const FIXTURE_PROVIDER_NODE = 'c'.repeat(64)
const DIFFERENT_RESPONSE_DIGEST = 'e'.repeat(64)

/** A body carrying our exchange's digests and the node that served it. */
export function fixtureHalfBody(
  opts: { capsuleId?: string; responseDigest?: string; servedBy?: string } = {}
): Record<string, unknown> {
  return {
    capsule_id: opts.capsuleId ?? 'mine-1',
    effect: {
      request_digest: FIXTURE_REQUEST_DIGEST,
      response_digest: opts.responseDigest ?? FIXTURE_RESPONSE_DIGEST
    },
    model_attestation: {
      compute_attestation: {
        'x-mesh-poc-v1': { serving_provenance: { served_by_node_id: opts.servedBy ?? FIXTURE_PROVIDER_NODE } }
      }
    }
  }
}

/** Our half of a correlated pair, with its body. */
export function fixtureMineCell(capsuleId = 'mine-1'): PaneCRow['mine'] {
  return {
    state: 'present-unverified',
    capsule_id: capsuleId,
    role: 'requested',
    record: fixtureHalfBody({ capsuleId })
  }
}

/** The pushed counterparty half. `agrees` -> both digests equal ours (closes);
 *  `disagrees` -> the response digest differs (contradicts). */
export function fixtureTheirsCell(
  outcome: 'agrees' | 'disagrees',
  opts: { capsuleId?: string; receivedFrom?: string } = {}
): PaneCRow['theirs'] {
  const capsuleId = opts.capsuleId ?? 'f'.repeat(64)
  return {
    state: 'present-unverified',
    capsule_id: capsuleId,
    role: 'served',
    received_from: opts.receivedFrom ?? 'endpoint-node-a',
    via: 'push',
    signature_ok: true,
    record: fixtureHalfBody({
      capsuleId,
      responseDigest: outcome === 'agrees' ? FIXTURE_RESPONSE_DIGEST : DIFFERENT_RESPONSE_DIGEST
    }),
    id_match: true
  }
}

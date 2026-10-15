// Loose types over the mesh capsule JSON shape (draft-mih-scitt-agent-action-
// capsule-05 + the x-mesh-poc-v1 extension). Deliberately permissive --
// mirrors capsule_mesh_viewer.py's tolerance of both real capture shapes --
// so an unrecognised/absent field degrades to `undefined`, never a parse
// failure. The full record always travels alongside (`record` on
// CapsuleEntry) for in-browser capsule_id recompute.

export type JsonRecord = Record<string, unknown>

export type CapsuleRecord = JsonRecord & {
  capsule_id?: string
  timestamp?: string
  operator?: string
  model_attestation?: {
    model_id?: string
    compute_attestation?: {
      'x-mesh-poc-v1'?: JsonRecord
      // `twin_adjudicator.seal_adjudication_capsule`'s own dict shape --
      // present only on a sealed adjudication capsule, never on the two
      // exchange halves it cites.
      adjudication?: {
        verdict?: string
        margin?: string
        margin_tau?: string
        half_a_capsule_id?: string
        half_b_capsule_id?: string
        referee_capsule_id?: string
        referee_id?: string
      }
    }
  }
  effect?: {
    request_digest?: string
    response_digest?: string
    effect_attestation?: string
  }
  disposition?: {
    decision?: string
    verdict_class?: string
  }
}

export type ServingProvenance = {
  model: string | null
  quantization: string | null
  architecture: string | null
  parameterSize: string | number | null
  contextLength: string | number | null
  layerCount: string | number | null
  modelIdentityHash: string | null
  modelCanonicalRef: string | null
  gpu: string | null
  vramBytes: number | null
  isSoc: boolean | null
  device: string | null
  hostname: string | null
  servedByNodeId: string | null
  requestingParty: string | null
  promptTokens: number | null
  completionTokens: number | null
  totalTokens: number | null
  generationParameters: JsonRecord
}

export type VerdictLine = { mark: 'ok' | 'warn'; text: string }

export type ReconcileField = { advertised: unknown; served: unknown; verdict: string }
export type ReconcileResult = {
  overall: 'match' | 'mismatch' | 'advertisement_absent' | 'no_served_facts'
  advertisementPresent: boolean
  fields: Record<string, ReconcileField>
  mismatches: string[]
}

export type CapsuleLedgerEntry = {
  record: CapsuleRecord
  capsuleId: string
  timestamp: string | null
  sp: ServingProvenance
  friendlyModel: string
}

export type CapsuleLedger = {
  records: CapsuleRecord[]
  nodePubKeyPem: string | null
}

// The OPTIONAL local disclosure preimage a producer may write to
// `<ledger_dir>/disclosures/<capsule_id>.json`. Never part of the signed
// capsule -- request_body/response_body are the EXACT JSON bodies the
// producer digested, so the UI can recompute request_digest/response_digest from them
// in-browser and prove (or disprove) a match against the sealed digests.
export type DisclosurePreimage = {
  capsule_id?: string
  request_body?: JsonRecord
  response_body?: JsonRecord
  request_text?: string | null
  response_text?: string | null
  tool_calls_note?: string | null
}

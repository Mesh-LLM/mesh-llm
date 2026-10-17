// Port of advertisement.py's reconcile_advertised_vs_served() -- verify-
// after-advertise (TRUST-MODEL.md SS12.3): reconciles a node's self-attested
// advertisement CLAIM against its serving_provenance record of what actually
// ran. Three-state discipline preserved: match / mismatch / advertisement_absent
// / no_served_facts -- a missing advertisement or missing served facts is
// NEVER rendered as a silent pass.
import type { JsonRecord, ReconcileResult } from '@/features/capsules/api/types'

function clean(v: unknown): unknown {
  if (typeof v === 'string' && v.trim().toLowerCase() === 'unknown') return null
  return v ?? null
}

function servedFacts(servingProvenance: JsonRecord | undefined): Record<string, unknown> {
  const sp = servingProvenance ?? {}
  const model = (sp.model as JsonRecord) ?? {}
  const hw = (sp.hardware as JsonRecord) ?? {}
  return {
    model_id: clean(sp.model_id ?? model.model_id),
    model_canonical_ref: clean(sp.model_canonical_ref ?? model.canonical_ref),
    quantization: clean(sp.quantization),
    hardware_gpu: clean(hw.gpu ?? sp.gpu),
    hardware_vram_bytes: clean(hw.vram_bytes !== undefined ? hw.vram_bytes : sp.vram_bytes),
    hardware_is_soc: clean(hw.is_soc !== undefined ? hw.is_soc : sp.is_soc)
  }
}

function valuesEqual(advertised: unknown, served: unknown): boolean {
  if (typeof advertised === 'string' && typeof served === 'string') {
    return advertised.trim().toLowerCase() === served.trim().toLowerCase()
  }
  return advertised === served
}

function reconcileField(advertised: unknown, served: unknown): string {
  if (served === null || served === undefined) return 'absent'
  if (advertised === null || advertised === undefined) return 'not_advertised'
  return valuesEqual(advertised, served) ? 'match' : 'mismatch'
}

export function reconcileAdvertisedVsServed(
  advertisement: JsonRecord | null | undefined,
  servingProvenance: JsonRecord | undefined
): ReconcileResult {
  if (!advertisement) {
    return { overall: 'advertisement_absent', advertisementPresent: false, fields: {}, mismatches: [] }
  }
  const hw = (advertisement.hardware as JsonRecord) ?? {}
  const advertised = {
    node_id: advertisement.node_id ?? null,
    model_id: advertisement.model_id ?? null,
    model_canonical_ref: advertisement.model_canonical_ref ?? null,
    quantization: advertisement.quantization ?? null,
    hardware_gpu: hw.gpu ?? null,
    hardware_vram_bytes: hw.vram_bytes ?? null,
    hardware_is_soc: hw.is_soc ?? null
  }
  const served = servedFacts(servingProvenance)
  const servedNodeId = servingProvenance?.served_by_node_id ?? null

  const fieldSpecs: [string, unknown, unknown][] = [
    ['model_id', advertised.model_id, served.model_id],
    ['model_canonical_ref', advertised.model_canonical_ref, served.model_canonical_ref],
    ['quantization', advertised.quantization, served.quantization],
    ['hardware_gpu', advertised.hardware_gpu, served.hardware_gpu],
    ['hardware_vram_bytes', advertised.hardware_vram_bytes, served.hardware_vram_bytes],
    ['hardware_is_soc', advertised.hardware_is_soc, served.hardware_is_soc]
  ]

  const fields: ReconcileResult['fields'] = {}
  const mismatches: string[] = []
  let anyReconcilable = false
  for (const [name, adv, srv] of fieldSpecs) {
    const verdict = reconcileField(adv, srv)
    fields[name] = { advertised: adv, served: srv, verdict }
    if (verdict === 'mismatch') mismatches.push(name)
    if (verdict === 'match' || verdict === 'mismatch') anyReconcilable = true
  }

  let nodeIdConsistent: boolean | null = null
  if (advertised.node_id && servedNodeId) nodeIdConsistent = advertised.node_id === servedNodeId
  if (nodeIdConsistent === false) mismatches.push('node_id')

  const overall =
    !anyReconcilable && mismatches.length === 0 ? 'no_served_facts' : mismatches.length ? 'mismatch' : 'match'

  return { overall, advertisementPresent: true, fields, mismatches }
}

// Port of capsule_mesh_viewer.py's serving_provenance() / friendly_model_name()
// / plain_model_line() / plain_token_split() / plain_gen_params_line(). Same
// honesty discipline: a field genuinely absent from the record stays
// undefined/null here -- nothing is invented, nothing silently defaults to a
// value the capsule didn't carry.
import type { CapsuleRecord, JsonRecord, ServingProvenance } from '@/features/capsules/api/types'

export function pocBlock(record: CapsuleRecord): JsonRecord {
  const block = record.model_attestation?.compute_attestation?.['x-mesh-poc-v1']
  return (block as JsonRecord | undefined) ?? {}
}

function asRecord(v: unknown): JsonRecord {
  return v && typeof v === 'object' ? (v as JsonRecord) : {}
}

function asString(v: unknown): string | null {
  return typeof v === 'string' ? v : null
}

function asNumber(v: unknown): number | null {
  return typeof v === 'number' ? v : null
}

function asBool(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null
}

export function servingProvenance(record: CapsuleRecord): ServingProvenance {
  const poc = pocBlock(record)
  const sp = asRecord(poc.serving_provenance)
  const model = asRecord(sp.model)
  const hw = asRecord(sp.hardware)
  const usage = asRecord(sp.usage)

  function pick(...names: string[]): unknown {
    for (const src of [sp, model, hw]) {
      for (const n of names) {
        if (src[n] !== undefined && src[n] !== null) return src[n]
      }
    }
    return null
  }

  return {
    model:
      asString(sp.model_canonical_ref) ??
      asString(model.canonical_ref) ??
      asString(record.model_attestation?.model_id) ??
      null,
    quantization: asString(sp.quantization),
    architecture: asString(pick('architecture')),
    parameterSize: (pick('parameter_size') as string | number | null) ?? null,
    contextLength: (pick('context_length') as string | number | null) ?? null,
    layerCount: (pick('layer_count') as string | number | null) ?? null,
    modelIdentityHash: asString(model.identity_hash) ?? asString(sp.model_identity_hash),
    modelCanonicalRef: asString(sp.model_canonical_ref) ?? asString(model.canonical_ref),
    gpu: asString(hw.gpu) ?? asString(sp.gpu),
    vramBytes: asNumber(hw.vram_bytes) ?? asNumber(sp.vram_bytes),
    isSoc: asBool(hw.is_soc) ?? asBool(sp.is_soc),
    device: asString(hw.device),
    hostname: asString(sp.hostname),
    servedByNodeId: asString(sp.served_by_node_id),
    requestingParty: asString(sp.requesting_party),
    promptTokens: asNumber(usage.prompt_tokens),
    completionTokens: asNumber(usage.completion_tokens),
    totalTokens: asNumber(usage.total_tokens),
    generationParameters: asRecord(poc.generation_parameters)
  }
}

function formatVram(vramBytes: number | null): string | null {
  if (vramBytes == null) return null
  const gb = vramBytes / 1024 ** 3
  return `${gb.toFixed(0)} GB VRAM`
}

export function plainTokenSplit(sp: ServingProvenance): string | null {
  const parts: string[] = []
  if (sp.promptTokens != null) parts.push(`${sp.promptTokens} in`)
  if (sp.completionTokens != null) parts.push(`${sp.completionTokens} out`)
  if (sp.totalTokens != null) parts.push(`${sp.totalTokens} total`)
  return parts.length ? parts.join(' / ') : null
}

/** "46 → 2 tokens" from the record's own usage block; `null` when the record
 *  carries neither count (never a placeholder). */
export function tokenFlowText(promptTokens: number | null, completionTokens: number | null): string | null {
  if (promptTokens == null && completionTokens == null) return null
  return `${promptTokens ?? '?'} → ${completionTokens ?? '?'} tokens`
}

/** "1.4 s" / "320 ms" from the record's `latency_ms`; `null` when absent or
 *  zero (a zero is "not measured" on this path, not an instant answer). */
export function durationText(latencyMs: unknown): string | null {
  const ms = typeof latencyMs === 'number' ? latencyMs : typeof latencyMs === 'string' ? Number(latencyMs) : Number.NaN
  if (!Number.isFinite(ms) || ms <= 0) return null
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`
}

export function plainModelLine(sp: ServingProvenance): string {
  const bits: string[] = [sp.model ?? '(model not named in record)']
  if (sp.quantization && sp.quantization !== 'unknown') bits.push(`(${sp.quantization})`)
  const hwBits: string[] = []
  if (sp.gpu) hwBits.push(sp.gpu)
  if (sp.isSoc) hwBits.push('SoC')
  const vram = formatVram(sp.vramBytes)
  if (vram) hwBits.push(vram)
  let line = bits.join(' ')
  if (hwBits.length) line += ' on ' + hwBits.join(', ')
  const split = plainTokenSplit(sp)
  if (split) line += `, ${split}`
  return line
}

const GEN_PARAM_LABELS: Record<string, string> = {
  temperature: 'temperature',
  top_p: 'top-p',
  top_k: 'top-k',
  min_p: 'min-p',
  seed: 'seed',
  max_tokens: 'max_tokens',
  max_completion_tokens: 'max_completion_tokens',
  n: 'n',
  presence_penalty: 'presence_penalty',
  frequency_penalty: 'frequency_penalty',
  repeat_penalty: 'repeat_penalty',
  stop: 'stop'
}
const GEN_PARAM_ORDER = Object.keys(GEN_PARAM_LABELS)

function formatGenValue(v: unknown): string {
  if (Array.isArray(v)) return v.map(String).join(', ')
  const s = String(v)
  const f = Number(s)
  if (Number.isNaN(f)) return s
  if (!s.includes('.') && !/e/i.test(s)) return s
  if (f === Math.trunc(f)) return String(Math.trunc(f))
  return String(Number(f.toPrecision(6)))
}

export function plainGenParamsLine(sp: ServingProvenance): string | null {
  const gp = sp.generationParameters
  if (!gp || Object.keys(gp).length === 0) return null
  const ordered = [
    ...GEN_PARAM_ORDER.filter((k) => k in gp),
    ...Object.keys(gp).filter((k) => !(k in GEN_PARAM_LABELS))
  ]
  const seen = new Set<string>()
  const bits: string[] = []
  for (const k of ordered) {
    if (seen.has(k)) continue
    seen.add(k)
    const v = gp[k]
    if (v == null) continue
    bits.push(`${GEN_PARAM_LABELS[k] ?? k} ${formatGenValue(v)}`)
  }
  return bits.length ? 'generated with: ' + bits.join(', ') : null
}

const ARCH_FAMILY: Record<string, string> = {
  llama: 'Llama',
  mistral: 'Mistral',
  qwen: 'Qwen',
  gemma: 'Gemma',
  phi: 'Phi'
}

/** The entry row's own model identity (design
 *  §3A: "model as `family/short-digest…` (full on hover/
 *  copy)") -- deliberately the RAW `family/digest` ref this record itself
 *  carries, unlike `friendlyModelName` below: this row already truncates
 *  every other id the same way (`short-id.ts`), so the raw ref reads as one
 *  more short/copyable identifier here, not the opaque string
 *  `friendlyModelName` exists to hide from a plain-language summary. `null`
 *  when the record carries no model ref at all -- never a placeholder. */
export function formatModelIdentity(modelRef: string | null): string | null {
  if (!modelRef) return null
  const slashIndex = modelRef.indexOf('/')
  if (slashIndex === -1) return modelRef
  const family = modelRef.slice(0, slashIndex)
  // A `sha256-` algorithm prefix is not part of the digest: shortening
  // `sha256-6c1a…` to six characters would print "sha256…" for every model.
  const digest = modelRef.slice(slashIndex + 1).replace(/^sha256[-:]/, '')
  const shortDigest = digest.length > 8 ? `${digest.slice(0, 6)}…` : digest
  return `${family}/${shortDigest}`
}

/** The model a row names: the friendliest name either side's record gives,
 *  ours first. `null` when neither record names one. */
export function rowModelName(own: ServingProvenance | null, theirs: ServingProvenance | null): string | null {
  const names = [own, theirs].filter((sp): sp is ServingProvenance => sp !== null).map(friendlyModelName)
  return names.find((name) => !name.startsWith('local model')) ?? names[0] ?? null
}

/** A human model name derived from architecture + parameter_size (+ quant) --
 * NEVER the raw local-gguf/sha256 id or model_identity_hash. */
export function friendlyModelName(sp: ServingProvenance): string {
  const ref = sp.modelCanonicalRef ?? sp.model ?? ''
  const quant = sp.quantization && sp.quantization !== 'unknown' ? sp.quantization : null

  if (ref && !ref.startsWith('local-gguf/') && ref.includes('/')) {
    const human = ref.split('/').at(-1)!
    return quant ? `${human} · ${quant}` : human
  }

  const arch = (sp.architecture ?? '').toLowerCase()
  const family = ARCH_FAMILY[arch]
  const size = sp.parameterSize
  if (family && size) {
    const base = `${family}-${size}`
    return quant ? `${base} · ${quant}` : base
  }
  if (family) return quant ? `${family} · ${quant}` : family

  return quant ? `local model · ${quant}` : 'local model'
}

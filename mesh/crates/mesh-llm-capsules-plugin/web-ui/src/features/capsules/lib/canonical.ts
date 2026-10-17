// Hand port of agent_action_capsule/canonical.py's vintage format-2
// construction (mirrors capsule-emit-mesh's mesh_viewer_static/mesh_verify.js
// byte-for-byte): drop {capsule_id, chain, signature, key_id}, absent-field
// normalize (remove null / empty-array / empty-object members, bottom-up),
// then RFC 8785 JCS, then SHA-256. Must stay in step with the Rust plugin's
// jcs.rs and the Python reference -- this is the third independent port of
// the same digest.

const LOCAL_ONLY = ['signature', 'key_id']
const CHAIN_LINKAGE = ['capsule_id', 'chain']
const MAX_SAFE = 9007199254740991 // 2^53 - 1

export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue }

function normalize(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(normalize)
  if (value !== null && typeof value === 'object') {
    const out: { [key: string]: JsonValue } = {}
    for (const key of Object.keys(value)) {
      const nv = normalize(value[key])
      if (nv === null || nv === undefined) continue
      if (Array.isArray(nv) && nv.length === 0) continue
      if (!Array.isArray(nv) && typeof nv === 'object' && Object.keys(nv).length === 0) continue
      out[key] = nv
    }
    return out
  }
  return value === undefined ? null : value
}

function jcsString(s: string): string {
  const out: string[] = ['"']
  for (let i = 0; i < s.length; i++) {
    const ch = s.charAt(i)
    const code = s.charCodeAt(i)
    if (ch === '"') out.push('\\"')
    else if (ch === '\\') out.push('\\\\')
    else if (code === 0x08) out.push('\\b')
    else if (code === 0x09) out.push('\\t')
    else if (code === 0x0a) out.push('\\n')
    else if (code === 0x0c) out.push('\\f')
    else if (code === 0x0d) out.push('\\r')
    else if (code < 0x20) out.push('\\u' + code.toString(16).padStart(4, '0'))
    else out.push(ch)
  }
  out.push('"')
  return out.join('')
}

export function jcsValue(v: JsonValue): string {
  if (v === null || v === undefined) return 'null'
  if (v === true) return 'true'
  if (v === false) return 'false'
  if (typeof v === 'string') return jcsString(v)
  if (typeof v === 'number') {
    if (!Number.isInteger(v)) throw new Error(`non-integer number in digest-bearing field: ${v}`)
    if (v > MAX_SAFE || v < -MAX_SAFE) throw new Error(`integer outside JS-safe range: ${v}`)
    return String(v)
  }
  if (Array.isArray(v)) return '[' + v.map(jcsValue).join(',') + ']'
  if (typeof v === 'object') {
    const keys = Object.keys(v).sort()
    return '{' + keys.map((k) => jcsString(k) + ':' + jcsValue(v[k])).join(',') + '}'
  }
  throw new Error('unserializable value')
}

function utf8Bytes(s: string): Uint8Array {
  return new TextEncoder().encode(s)
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const buf = await crypto.subtle.digest('SHA-256', bytes as BufferSource)
  const arr = new Uint8Array(buf)
  let hex = ''
  for (let i = 0; i < arr.length; i++) hex += arr[i].toString(16).padStart(2, '0')
  return hex
}

/** Every `capsule_id`/digest this codebase mints is `sha256Hex`'s own
 *  output shape: 64 lowercase hex characters, nothing else (see `mesh-
 *  console-evidence-tab-honesty-defects` finding 5 -- a chat-completion id
 *  like `capsule-chatcmpl-1790147257740` is not a capsule id, and a
 *  fallback sentinel like `unknown-request:<model>` is not a digest, no
 *  matter how it's labelled on the wire). Callers use this to refuse to
 *  treat either as the real thing. */
const DIGEST_SHAPE = /^[0-9a-f]{64}$/

export function isDigestShaped(value: string | null | undefined): boolean {
  return typeof value === 'string' && DIGEST_SHAPE.test(value)
}

/** Recompute a mesh capsule's `capsule_id` in-browser (the vintage format-2 construction). */
export async function recomputeCapsuleId(record: Record<string, unknown>): Promise<string> {
  if (Object.prototype.hasOwnProperty.call(record, 'canonicalization_id')) {
    const excl4 = { capsule_id: 1, signature: 1, key_id: 1 } as Record<string, 1>
    const c4: Record<string, JsonValue> = {}
    for (const k of Object.keys(record)) if (!excl4[k]) c4[k] = record[k] as JsonValue
    return sha256Hex(utf8Bytes(jcsValue(c4)))
  }
  const excluded: Record<string, 1> = {}
  for (const k of [...CHAIN_LINKAGE, ...LOCAL_ONLY]) excluded[k] = 1
  const canonical: Record<string, JsonValue> = {}
  for (const k of Object.keys(record)) if (!excluded[k]) canonical[k] = record[k] as JsonValue
  return sha256Hex(utf8Bytes(jcsValue(normalize(canonical))))
}

/** The canonical JSON-DIGEST (JCS + SHA-256) of an object whose digest-bearing
 * values are strings/integers only -- the served-terminal-facts shape. */
export async function digestFacts(obj: JsonValue): Promise<string> {
  return sha256Hex(utf8Bytes(jcsValue(obj)))
}

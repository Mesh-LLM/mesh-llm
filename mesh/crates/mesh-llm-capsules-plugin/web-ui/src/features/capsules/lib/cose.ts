// Minimal COSE_Sign1 (RFC 9052 SS4.4) decode + verify, scoped to exactly the
// shape capsule-emit's `cose` module emits: CBOR tag 18, definite-length array
// of [protected: bstr, unprotected: map, payload: bstr, signature: bstr],
// EdDSA over the "Signature1" Sig_structure with an empty external_aad and an
// ATTACHED payload. Not a general CBOR/COSE library -- just enough to verify
// this one producer's signed statements client-side.
import { ed25519 } from '@noble/curves/ed25519'

class CborReader {
  constructor(
    private readonly bytes: Uint8Array,
    public offset = 0
  ) {}

  private byte(): number {
    if (this.offset >= this.bytes.length) throw new Error('CBOR: unexpected end of input')
    return this.bytes[this.offset++]
  }

  private readLength(additional: number): number {
    if (additional < 24) return additional
    if (additional === 24) return this.byte()
    if (additional === 25) {
      const hi = this.byte()
      const lo = this.byte()
      return (hi << 8) | lo
    }
    if (additional === 26) {
      let n = 0
      for (let i = 0; i < 4; i++) n = (n << 8) | this.byte()
      return n >>> 0
    }
    throw new Error(`CBOR: unsupported length encoding (additional=${additional})`)
  }

  /** Reads one CBOR data item. Returns a tag number for major type 6, an
   * array of items for major type 4, a raw byte slice for major type 2/3,
   * or a number for major type 0. Enough for a COSE_Sign1 envelope. */
  readItem(): { majorType: number; tag?: number; value: unknown } {
    const initial = this.byte()
    const majorType = initial >> 5
    const additional = initial & 0x1f
    switch (majorType) {
      case 0: {
        return { majorType, value: this.readLength(additional) }
      }
      case 2: {
        const len = this.readLength(additional)
        const start = this.offset
        this.offset += len
        return { majorType, value: this.bytes.slice(start, this.offset) }
      }
      case 3: {
        const len = this.readLength(additional)
        const start = this.offset
        this.offset += len
        return { majorType, value: new TextDecoder().decode(this.bytes.slice(start, this.offset)) }
      }
      case 4: {
        const len = this.readLength(additional)
        const items: unknown[] = []
        for (let i = 0; i < len; i++) items.push(this.readItem().value)
        return { majorType, value: items }
      }
      case 5: {
        const len = this.readLength(additional)
        const map = new Map<unknown, unknown>()
        for (let i = 0; i < len; i++) {
          const k = this.readItem().value
          const v = this.readItem().value
          map.set(k, v)
        }
        return { majorType, value: map }
      }
      case 6: {
        const tag = this.readLength(additional)
        const inner = this.readItem()
        return { majorType, tag, value: inner.value }
      }
      case 7: {
        if (additional === 22) return { majorType, value: null }
        if (additional === 20) return { majorType, value: false }
        if (additional === 21) return { majorType, value: true }
        throw new Error(`CBOR: unsupported simple value (additional=${additional})`)
      }
      default:
        throw new Error(`CBOR: unsupported major type ${majorType}`)
    }
  }
}

function cborEncodeHead(majorType: number, length: number): number[] {
  const mt = majorType << 5
  if (length < 24) return [mt | length]
  if (length < 256) return [mt | 24, length]
  if (length < 65536) return [mt | 25, (length >> 8) & 0xff, length & 0xff]
  return [mt | 26, (length >>> 24) & 0xff, (length >>> 16) & 0xff, (length >>> 8) & 0xff, length & 0xff]
}

function cborEncodeByteString(bytes: Uint8Array): Uint8Array {
  const head = cborEncodeHead(2, bytes.length)
  const out = new Uint8Array(head.length + bytes.length)
  out.set(head, 0)
  out.set(bytes, head.length)
  return out
}

function cborEncodeTextString(text: string): Uint8Array {
  const textBytes = new TextEncoder().encode(text)
  const head = cborEncodeHead(3, textBytes.length)
  const out = new Uint8Array(head.length + textBytes.length)
  out.set(head, 0)
  out.set(textBytes, head.length)
  return out
}

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((n, c) => n + c.length, 0)
  const out = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}

export type DecodedCoseSign1 = {
  protectedHeader: Uint8Array
  payload: Uint8Array
  signature: Uint8Array
}

/** Decodes a CBOR tag-18 COSE_Sign1 message: [protected, unprotected, payload, signature]. */
export function decodeCoseSign1(bytes: Uint8Array): DecodedCoseSign1 {
  const reader = new CborReader(bytes)
  const item = reader.readItem()
  if (item.majorType !== 6 || item.tag !== 18) {
    throw new Error('not a CBOR tag-18 COSE_Sign1 message')
  }
  const parts = item.value
  if (!Array.isArray(parts) || parts.length !== 4) {
    throw new Error('COSE_Sign1 must be a 4-element array')
  }
  const [protectedHeader, , payload, signature] = parts
  if (!(protectedHeader instanceof Uint8Array)) throw new Error('COSE_Sign1: protected header must be a byte string')
  if (!(payload instanceof Uint8Array)) throw new Error('COSE_Sign1: only an attached payload is supported')
  if (!(signature instanceof Uint8Array)) throw new Error('COSE_Sign1: signature must be a byte string')
  return { protectedHeader, payload, signature }
}

/** Builds the "Signature1" Sig_structure bytes (RFC 9052 SS4.4) COSE signs over. */
function buildSigStructure(protectedHeader: Uint8Array, payload: Uint8Array): Uint8Array {
  const items = [
    cborEncodeTextString('Signature1'),
    cborEncodeByteString(protectedHeader),
    cborEncodeByteString(new Uint8Array(0)), // external_aad, always empty here
    cborEncodeByteString(payload)
  ]
  const head = cborEncodeHead(4, items.length)
  return concatBytes([new Uint8Array(head), ...items])
}

/** SubjectPublicKeyInfo PEM -> raw 32-byte Ed25519 public key. Ed25519 SPKI DER
 * is always exactly 44 bytes (12-byte fixed AlgorithmIdentifier header + the
 * raw 32-byte key), so the last 32 bytes are the key regardless of the exact
 * header encoding. */
export function ed25519PublicKeyFromSpkiPem(pem: string): Uint8Array {
  const base64 = pem
    .replace(/-----BEGIN [^-]+-----/g, '')
    .replace(/-----END [^-]+-----/g, '')
    .replace(/\s+/g, '')
  const der = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))
  if (der.length < 32) throw new Error('SPKI PEM too short to contain an Ed25519 key')
  return der.slice(der.length - 32)
}

export type CoseVerifyResult = { verified: true; payload: Uint8Array } | { verified: false; reason: string }

/** Verifies a COSE_Sign1 message's EdDSA signature against a raw Ed25519 public key. */
export function verifyCoseSign1(bytes: Uint8Array, publicKey: Uint8Array): CoseVerifyResult {
  let decoded: DecodedCoseSign1
  try {
    decoded = decodeCoseSign1(bytes)
  } catch (err) {
    return { verified: false, reason: err instanceof Error ? err.message : 'decode failed' }
  }
  const sigStructure = buildSigStructure(decoded.protectedHeader, decoded.payload)
  try {
    const ok = ed25519.verify(decoded.signature, sigStructure, publicKey)
    return ok ? { verified: true, payload: decoded.payload } : { verified: false, reason: 'signature does not verify' }
  } catch (err) {
    return { verified: false, reason: err instanceof Error ? err.message : 'verify failed' }
  }
}

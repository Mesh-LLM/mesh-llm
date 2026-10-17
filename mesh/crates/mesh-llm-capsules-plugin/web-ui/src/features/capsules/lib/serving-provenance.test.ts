import { describe, expect, it } from 'vitest'
import { formatModelIdentity } from '@/features/capsules/lib/serving-provenance'

describe('formatModelIdentity', () => {
  it('shortens the digest half of a family/digest ref, design §3A verbatim example', () => {
    expect(formatModelIdentity('local-gguf/7089c7abcdef0123456789')).toBe('local-gguf/7089c7…')
  })

  it('leaves a digest of 8 chars or fewer unshortened', () => {
    expect(formatModelIdentity('local-gguf/abcd1234')).toBe('local-gguf/abcd1234')
  })

  it('a ref with no "/" renders as-is -- nothing to split into family/digest', () => {
    expect(formatModelIdentity('qwen2.5-7b')).toBe('qwen2.5-7b')
  })

  it('null (no model ref on the record) -> null, never a placeholder', () => {
    expect(formatModelIdentity(null)).toBeNull()
  })
})

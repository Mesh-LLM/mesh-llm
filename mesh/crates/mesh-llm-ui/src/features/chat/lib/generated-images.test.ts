import { describe, expect, it } from 'vitest'
import {
  MAX_GENERATED_IMAGE_CHARS,
  isGeneratedImageUrl,
  summarizeGeneratedImages
} from '@/features/chat/lib/generated-images'

describe('generated image URLs', () => {
  it('bounds inline raster image URLs before the browser decodes them', () => {
    const prefix = 'data:image/png;base64,'
    const url = prefix + 'A'.repeat(MAX_GENERATED_IMAGE_CHARS - prefix.length)
    expect(isGeneratedImageUrl(url)).toBe(true)
    expect(isGeneratedImageUrl(`${url}A`)).toBe(false)
  })

  it.each([
    'data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=',
    'data:text/html;base64,aGVsbG8=',
    'data:image/png;base64,',
    'data:image/png;base64,aGVsbG8= trailing',
    'data:image/png;base64,aGVsbG8%3D'
  ])('rejects unsupported or malformed inline data: %s', (url) => {
    expect(isGeneratedImageUrl(url)).toBe(false)
  })

  it('removes pixels even after repeated incomplete Markdown image markers', () => {
    const prefix = '!['.repeat(10_000)
    expect(summarizeGeneratedImages(`${prefix}(data:image/png;base64,YWJj`)).toBe(`${prefix}([Generated image]`)
  })
})

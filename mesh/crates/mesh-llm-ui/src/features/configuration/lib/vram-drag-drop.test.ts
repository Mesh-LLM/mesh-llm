import { describe, expect, it } from 'vitest'
import { getTypedDataId, MODEL_MIME_PREFIX, modelDragMimeType } from '@/features/configuration/lib/vram-drag-drop'

describe('vram drag/drop metadata', () => {
  it('round trips mixed-case model ids through a lowercase-safe MIME type', () => {
    const modelId = 'unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL'
    const mimeType = modelDragMimeType(modelId)

    expect(mimeType).toBe(mimeType.toLowerCase())
    expect(getTypedDataId([mimeType], MODEL_MIME_PREFIX)).toBe(modelId)
  })

  it('keeps legacy raw ids readable for existing assignment drag payloads', () => {
    expect(getTypedDataId([`${MODEL_MIME_PREFIX}local-gguf-sha256`], MODEL_MIME_PREFIX)).toBe('local-gguf-sha256')
  })
})

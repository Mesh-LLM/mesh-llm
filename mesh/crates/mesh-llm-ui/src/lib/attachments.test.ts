import { describe, expect, it } from 'vitest'

import {
  AUDIO_MAX_BYTES,
  FILE_MAX_BYTES,
  IMAGE_MAX_BYTES,
  getAttachmentSendIssue,
  resolveImageFile,
  sniffImageType,
  validateAttachmentFile
} from '@/lib/attachments'

const PNG_HEADER = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
const JPEG_HEADER = [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]
const GIF87A_HEADER = [0x47, 0x49, 0x46, 0x38, 0x37, 0x61, 0x04, 0x00]
const GIF89A_HEADER = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x04, 0x00]
const WEBP_HEADER = [0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50]

describe('validateAttachmentFile', () => {
  it('rejects wrong image mime types', () => {
    expect(validateAttachmentFile({ size: 128, type: 'text/plain' }, 'image')).toBe(
      'Only image files are allowed here.'
    )
  })

  it('rejects oversized audio files', () => {
    expect(validateAttachmentFile({ size: AUDIO_MAX_BYTES + 1, type: 'audio/mpeg' }, 'audio')).toContain(
      'Audio file is too large'
    )
  })

  it('accepts generic files within size limits', () => {
    expect(validateAttachmentFile({ size: FILE_MAX_BYTES, type: 'application/pdf' }, 'file')).toBeNull()
  })

  it('rejects oversized images', () => {
    expect(validateAttachmentFile({ size: IMAGE_MAX_BYTES + 1, type: 'image/png' }, 'image')).toContain(
      'Image is too large'
    )
  })
})

describe('getAttachmentSendIssue', () => {
  it('returns selected-model mismatch errors for audio', () => {
    expect(
      getAttachmentSendIssue({
        pendingKinds: new Set(['audio']),
        selectedModel: 'text-only',
        warmModels: ['text-only'],
        audioModels: new Set<string>(),
        multimodalModels: new Set<string>()
      })
    ).toBe("text-only doesn't support audio. Choose an audio-capable model or remove the attachment.")
  })

  it('returns no warm model errors for auto selection', () => {
    expect(
      getAttachmentSendIssue({
        pendingKinds: new Set(['file']),
        selectedModel: 'auto',
        warmModels: ['text-only'],
        audioModels: new Set<string>(),
        multimodalModels: new Set<string>()
      })
    ).toBe('No warm model supports the attached media. Warm a compatible model or remove the attachment.')
  })

  it('returns null when no media kinds need model support', () => {
    // Images are handled in-browser, so pendingKinds should be empty.
    expect(
      getAttachmentSendIssue({
        pendingKinds: new Set(),
        selectedModel: 'auto',
        warmModels: ['text-only'],
        audioModels: new Set<string>(),
        multimodalModels: new Set<string>()
      })
    ).toBeNull()
  })

  it('allows supported audio attachments', () => {
    expect(
      getAttachmentSendIssue({
        pendingKinds: new Set(['audio']),
        selectedModel: 'auto',
        warmModels: ['audio-model'],
        audioModels: new Set(['audio-model']),
        multimodalModels: new Set<string>()
      })
    ).toBeNull()
  })
})

describe('sniffImageType', () => {
  it('detects png, jpeg, gif, and webp magic bytes', () => {
    expect(sniffImageType(new Uint8Array(PNG_HEADER))).toBe('image/png')
    expect(sniffImageType(new Uint8Array(JPEG_HEADER))).toBe('image/jpeg')
    expect(sniffImageType(new Uint8Array(GIF87A_HEADER))).toBe('image/gif')
    expect(sniffImageType(new Uint8Array(GIF89A_HEADER))).toBe('image/gif')
    expect(sniffImageType(new Uint8Array(WEBP_HEADER))).toBe('image/webp')
  })

  it('returns null for non-image bytes', () => {
    expect(sniffImageType(new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d, 0x31, 0x2e, 0x37]))).toBeNull()
    expect(
      sniffImageType(new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45]))
    ).toBeNull()
  })

  it('returns null for headers too short to match', () => {
    expect(sniffImageType(new Uint8Array([0x89, 0x50, 0x4e]))).toBeNull()
    expect(sniffImageType(new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x00, 0x00]))).toBeNull()
    expect(sniffImageType(new Uint8Array())).toBeNull()
  })
})

describe('resolveImageFile', () => {
  it('re-wraps an empty-type image with the sniffed type', async () => {
    const file = new File([new Uint8Array(PNG_HEADER)], 'clipboard-image', { type: '' })

    const resolved = await resolveImageFile(file)

    expect(resolved).not.toBe(file)
    expect(resolved.type).toBe('image/png')
    expect(resolved.name).toBe('clipboard-image')
    expect(resolved.size).toBe(PNG_HEADER.length)
    expect(resolved.lastModified).toBe(file.lastModified)
  })

  it('returns files that already carry a type unchanged', async () => {
    const typed = new File(['image-bytes'], 'cat.png', { type: 'image/png' })

    await expect(resolveImageFile(typed)).resolves.toBe(typed)
  })

  it('returns empty-type files whose bytes are not an image unchanged', async () => {
    const documentFile = new File(['%PDF-1.7'], 'doc.pdf', { type: '' })

    await expect(resolveImageFile(documentFile)).resolves.toBe(documentFile)
  })

  it('returns the file unchanged when its header cannot be read', async () => {
    const unreadable = {
      name: 'revoked',
      type: '',
      lastModified: 0,
      slice: () => {
        throw new Error('blob already revoked')
      }
    } as unknown as File

    await expect(resolveImageFile(unreadable)).resolves.toBe(unreadable)
  })
})

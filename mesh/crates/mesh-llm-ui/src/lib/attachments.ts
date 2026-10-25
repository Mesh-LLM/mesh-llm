export type AttachmentKind = 'image' | 'audio' | 'file'

export const IMAGE_MAX_BYTES = 12 * 1024 * 1024
export const AUDIO_MAX_BYTES = 24 * 1024 * 1024
export const FILE_MAX_BYTES = 24 * 1024 * 1024

type FileLike = {
  size: number
  type: string
}

type AttachmentSupportOptions = {
  /** The *effective* media kinds that still need model support.
   *  Images and PDFs are always handled in-browser (Florence-2 / pdf.js)
   *  so they should NOT appear here. Only audio and generic files need
   *  model support. */
  pendingKinds: ReadonlySet<AttachmentKind>
  selectedModel: string
  warmModels: readonly string[]
  audioModels: ReadonlySet<string>
  multimodalModels: ReadonlySet<string>
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function validateAttachmentFile(file: FileLike, kind: AttachmentKind): string | null {
  if (kind === 'image') {
    if (!file.type.startsWith('image/')) return 'Only image files are allowed here.'
    if (file.size > IMAGE_MAX_BYTES) {
      return `Image is too large (${formatBytes(file.size)}). Max is ${formatBytes(IMAGE_MAX_BYTES)}.`
    }
    return null
  }
  if (kind === 'audio') {
    if (!file.type.startsWith('audio/')) return 'Only audio files are allowed here.'
    if (file.size > AUDIO_MAX_BYTES) {
      return `Audio file is too large (${formatBytes(file.size)}). Max is ${formatBytes(AUDIO_MAX_BYTES)}.`
    }
    return null
  }
  if (file.size > FILE_MAX_BYTES) {
    return `File is too large (${formatBytes(file.size)}). Max is ${formatBytes(FILE_MAX_BYTES)}.`
  }
  return null
}

export function getAttachmentSendIssue({
  pendingKinds,
  selectedModel,
  warmModels,
  audioModels,
  multimodalModels
}: AttachmentSupportOptions): string | null {
  // Images and PDFs are always handled in the browser (described via
  // Florence-2 / extracted via pdf.js) so they never require model support.
  // Only audio and generic file attachments still need a capable model.
  if (!pendingKinds.size) return null

  const modelSupports = (modelName: string) =>
    (!pendingKinds.has('audio') || audioModels.has(modelName)) &&
    (!pendingKinds.has('file') || multimodalModels.has(modelName))

  if (selectedModel && selectedModel !== 'auto') {
    if (modelSupports(selectedModel)) return null

    if (pendingKinds.has('audio') && !audioModels.has(selectedModel)) {
      return `${selectedModel} doesn't support audio. Choose an audio-capable model or remove the attachment.`
    }
    return 'Selected model does not support the attached media. Choose a compatible model or remove the attachment.'
  }

  if (warmModels.some(modelSupports)) return null

  // Auto mode but no warm model supports it.
  if (pendingKinds.has('audio')) {
    return 'No warm model supports audio input. Warm an audio-capable model to use audio attachments.'
  }
  return 'No warm model supports the attached media. Warm a compatible model or remove the attachment.'
}

const IMAGE_HEADER_MAX_BYTES = 12

function headerStartsWith(header: Uint8Array, signature: readonly number[], offset = 0): boolean {
  return signature.every((byte, index) => header[offset + index] === byte)
}

/** Detects the image MIME type from leading magic bytes (PNG, JPEG, GIF,
 *  WebP); returns null for any other byte sequence. */
export function sniffImageType(header: Uint8Array): string | null {
  if (headerStartsWith(header, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) return 'image/png'
  if (headerStartsWith(header, [0xff, 0xd8, 0xff])) return 'image/jpeg'
  if (headerStartsWith(header, [0x47, 0x49, 0x46, 0x38, 0x37, 0x61])) return 'image/gif'
  if (headerStartsWith(header, [0x47, 0x49, 0x46, 0x38, 0x39, 0x61])) return 'image/gif'
  if (headerStartsWith(header, [0x52, 0x49, 0x46, 0x46]) && headerStartsWith(header, [0x57, 0x45, 0x42, 0x50], 8)) {
    return 'image/webp'
  }
  return null
}

/** Clipboard and picker files can arrive with an empty `type` when the browser
 *  could not determine a MIME type. Reads the leading bytes and re-wraps the
 *  file with the sniffed image type so downstream image routing works; every
 *  other file (including unreadable ones) is returned unchanged. */
export async function resolveImageFile(file: File): Promise<File> {
  if (file.type !== '') return file
  let header: Uint8Array
  try {
    header = new Uint8Array(await file.slice(0, IMAGE_HEADER_MAX_BYTES).arrayBuffer())
  } catch {
    return file
  }
  const imageType = sniffImageType(header)
  if (!imageType) return file
  return new File([file], file.name, { type: imageType, lastModified: file.lastModified })
}

export async function resolveImageFiles(files: readonly File[]): Promise<File[]> {
  return Promise.all(files.map((file) => resolveImageFile(file)))
}

export type PasteRange = { start: number; end: number }

/** One suppressed-paste text insertion: `text` replaced `[start, end)`.
 *  `skipped` marks a queued insertion whose paste became an image
 *  attachment; it maps indices and values as if it never existed. */
export type PasteRestoreEffect = { start: number; end: number; text: string; skipped?: boolean }

/** Maps a caret or range edge across one earlier insertion: indices before
 *  it stay, indices at or after its end shift by the length delta, and an
 *  index inside the replaced span collapses onto the end of the inserted
 *  text (sequential pastes at the same point read back-to-back). */
export function adjustPasteIndex(index: number, effect: PasteRestoreEffect): number {
  if (index < effect.start) return index
  if (index >= effect.end) return index + effect.text.length - (effect.end - effect.start)
  return effect.start + effect.text.length
}

export function adjustPasteRange(range: PasteRange, effects: readonly PasteRestoreEffect[]): PasteRange {
  return effects
    .filter((effect) => !effect.skipped)
    .reduce<PasteRange>(
      (adjusted, effect) => ({
        start: adjustPasteIndex(adjusted.start, effect),
        end: adjustPasteIndex(adjusted.end, effect)
      }),
      range
    )
}

/** Applies one insertion the way the suppressed default paste would have. */
export function applyPasteRestore(value: string, range: PasteRange, text: string): string {
  return `${value.slice(0, range.start)}${text}${value.slice(range.end)}`
}

export function applyPasteEffects(value: string, effects: readonly PasteRestoreEffect[]): string {
  return effects
    .filter((effect) => !effect.skipped)
    .reduce((applied, effect) => applyPasteRestore(applied, effect, effect.text), value)
}

/** How many leading effects a captured prompt value already contains, so a
 *  later capture only adjusts for the rest of the queue; -1 when the value
 *  matches no prefix (the draft changed by other means and the queued
 *  restorations should fall back to appending). */
export function pasteRestoreBaseOffset(value: string, base: string, effects: readonly PasteRestoreEffect[]): number {
  let applied = base
  if (value === applied) return 0
  for (let index = 0; index < effects.length; index += 1) {
    if (effects[index].skipped) continue
    applied = applyPasteRestore(applied, effects[index], effects[index].text)
    if (value === applied) return index + 1
  }
  return -1
}

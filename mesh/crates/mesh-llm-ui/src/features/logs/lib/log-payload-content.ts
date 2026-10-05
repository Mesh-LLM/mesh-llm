import type { LogArtifact, LogArtifactUnavailableReason } from '@/features/logs/api/schemas'
import {
  decodeEventStream,
  type LogEventStreamData,
  type LogEventStreamFrame
} from '@/features/logs/lib/log-event-stream'
import { sortByOccurredAt } from '@/features/logs/lib/log-instant'

export { decodeEventStream }
export type { LogEventStreamData, LogEventStreamFrame }

export type AvailableLogArtifact = Extract<LogArtifact, { contentState: 'available' }>

export type LogArtifactCategory = 'request' | 'response' | 'error' | 'unclassified'

export type LogArtifactGroup = {
  readonly primary: LogArtifact | undefined
  readonly artifacts: readonly LogArtifact[]
}

export type ClassifiedLogArtifacts = {
  readonly request: LogArtifactGroup
  readonly response: LogArtifactGroup
  readonly error: LogArtifactGroup
  readonly unclassified: LogArtifactGroup
}

export type LogPayloadContent =
  | {
      readonly state: 'json'
      readonly text: string
      readonly prettyText: string
    }
  | { readonly state: 'text'; readonly text: string }
  | { readonly state: 'event-stream'; readonly frames: readonly LogEventStreamFrame[] }
  | { readonly state: 'malformed-json'; readonly text: string }
  | { readonly state: 'binary'; readonly bytes: Uint8Array }
  | { readonly state: 'not-loaded' }
  | { readonly state: 'unavailable'; readonly reason?: LogArtifactUnavailableReason }
  | { readonly state: 'missing' }
  | { readonly state: 'corrupt' }
  | { readonly state: 'too-large' }
  | { readonly state: 'decode-error'; readonly reason: 'base64' | 'utf8' }

export const LOG_PAYLOAD_RENDER_LIMIT_BYTES = 16 * 1024 * 1024

const LOG_PAYLOAD_MAX_BASE64_LENGTH = 4 * Math.ceil(LOG_PAYLOAD_RENDER_LIMIT_BYTES / 3)

function artifactCategory(kind: string): LogArtifactCategory {
  const tokens = kind.toLowerCase().split(/[^a-z0-9]+/)
  for (const token of tokens) {
    switch (token) {
      case 'request':
        return 'request'
      case 'response':
        return 'response'
      case 'error':
        return 'error'
    }
  }
  return 'unclassified'
}

function artifactGroup(artifacts: readonly LogArtifact[]): LogArtifactGroup {
  const ordered = sortByOccurredAt(artifacts)
  const primary =
    ordered.find((artifact) => artifact.kind.toLowerCase().endsWith('_body')) ??
    ordered.find((artifact) => artifact.contentState === 'available') ??
    ordered[0]
  return { primary, artifacts: ordered }
}

export function classifyLogArtifacts(artifacts: readonly LogArtifact[]): ClassifiedLogArtifacts {
  const request: LogArtifact[] = []
  const response: LogArtifact[] = []
  const error: LogArtifact[] = []
  const unclassified: LogArtifact[] = []

  for (const artifact of artifacts) {
    switch (artifactCategory(artifact.kind)) {
      case 'request':
        request.push(artifact)
        break
      case 'response':
        response.push(artifact)
        break
      case 'error':
        error.push(artifact)
        break
      case 'unclassified':
        unclassified.push(artifact)
        break
    }
  }

  return {
    request: artifactGroup(request),
    response: artifactGroup(response),
    error: artifactGroup(error),
    unclassified: artifactGroup(unclassified)
  }
}

function mediaType(mediaKind: string | undefined): string | undefined {
  return mediaKind?.split(';')[0]?.trim().toLowerCase()
}

function isJsonMediaType(value: string | undefined): boolean {
  return value === 'application/json' || value === 'text/json' || value?.endsWith('+json') === true
}

function looksLikeJson(value: string): boolean {
  const trimmed = value.trimStart()
  return trimmed.startsWith('{') || trimmed.startsWith('[')
}

function base64PaddingLength(value: string): number {
  if (value.endsWith('==')) return 2
  return value.endsWith('=') ? 1 : 0
}

function decodedBase64Length(value: string): number | undefined {
  const remainder = value.length % 4
  if (remainder === 1) return undefined
  if (remainder !== 0) return Math.floor(value.length / 4) * 3 + remainder - 1
  return (value.length / 4) * 3 - base64PaddingLength(value)
}

export function isLogArtifactContentTooLarge(artifact: Pick<AvailableLogArtifact, 'bytes' | 'contentBase64'>): boolean {
  if (artifact.bytes > LOG_PAYLOAD_RENDER_LIMIT_BYTES) return true
  if (artifact.contentBase64 === undefined) return false
  if (artifact.contentBase64.length > LOG_PAYLOAD_MAX_BASE64_LENGTH) return true
  const decodedLength = decodedBase64Length(artifact.contentBase64)
  return decodedLength !== undefined && decodedLength > LOG_PAYLOAD_RENDER_LIMIT_BYTES
}

function base64CharacterValue(characterCode: number): number | undefined {
  if (characterCode >= 65 && characterCode <= 90) return characterCode - 65
  if (characterCode >= 97 && characterCode <= 122) return characterCode - 71
  if (characterCode >= 48 && characterCode <= 57) return characterCode + 4
  if (characterCode === 43) return 62
  if (characterCode === 47) return 63
  return undefined
}

function base64UnusedBitMask(valueLength: number, paddingLength: number): number {
  if (paddingLength === 2 || (paddingLength === 0 && valueLength % 4 === 2)) return 0b1111
  if (paddingLength === 1 || (paddingLength === 0 && valueLength % 4 === 3)) return 0b11
  return 0
}

function hasCanonicalBase64TrailingBits(value: string, contentLength: number, paddingLength: number): boolean {
  const unusedBitMask = base64UnusedBitMask(value.length, paddingLength)
  if (unusedBitMask === 0) return true
  const trailingValue = base64CharacterValue(value.charCodeAt(contentLength - 1))
  if (trailingValue === undefined) return false
  return (trailingValue & unusedBitMask) === 0
}

function isValidBase64(value: string): boolean {
  const remainder = value.length % 4
  if (remainder === 1) return false
  const paddingLength = base64PaddingLength(value)
  if (paddingLength > 0 && remainder !== 0) return false
  const contentLength = value.length - paddingLength
  if (value.indexOf('=') !== -1 && value.indexOf('=') < contentLength) return false
  for (let index = 0; index < contentLength; index += 1) {
    if (base64CharacterValue(value.charCodeAt(index)) === undefined) return false
  }
  return hasCanonicalBase64TrailingBits(value, contentLength, paddingLength)
}

export function decodeBase64(value: string): string | undefined {
  if (!isValidBase64(value)) return undefined
  try {
    return atob(value)
  } catch (error) {
    if (error instanceof DOMException || error instanceof TypeError) return undefined
    throw error
  }
}

function decodeUtf8(bytes: Uint8Array): string | undefined {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch (error) {
    if (error instanceof TypeError) return undefined
    throw error
  }
}

function decodeJson(text: string): Extract<LogPayloadContent, { state: 'json' | 'malformed-json' }> {
  try {
    const value: unknown = JSON.parse(text)
    const formatted = JSON.stringify(value, null, 2)
    return formatted === undefined ? { state: 'malformed-json', text } : { state: 'json', text, prettyText: formatted }
  } catch (error) {
    if (error instanceof SyntaxError) return { state: 'malformed-json', text }
    throw error
  }
}

function decodeAvailableContent(artifact: AvailableLogArtifact): LogPayloadContent {
  if (artifact.contentBase64 === undefined) return { state: 'not-loaded' }
  if (isLogArtifactContentTooLarge(artifact)) return { state: 'too-large' }

  const binary = decodeBase64(artifact.contentBase64)
  if (binary === undefined) return { state: 'decode-error', reason: 'base64' }
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0))
  const decodedText = decodeUtf8(bytes)
  const normalizedMediaType = mediaType(artifact.mediaKind)

  if (normalizedMediaType === 'text/event-stream') {
    return decodedText === undefined
      ? { state: 'decode-error', reason: 'utf8' }
      : { state: 'event-stream', frames: decodeEventStream(decodedText) }
  }
  if (isJsonMediaType(normalizedMediaType)) {
    return decodedText === undefined ? { state: 'decode-error', reason: 'utf8' } : decodeJson(decodedText)
  }
  if (decodedText !== undefined && looksLikeJson(decodedText)) return decodeJson(decodedText)
  if (normalizedMediaType?.startsWith('text/') === true) {
    return decodedText === undefined ? { state: 'decode-error', reason: 'utf8' } : { state: 'text', text: decodedText }
  }
  return { state: 'binary', bytes }
}

export function decodeLogArtifactContent(artifact: LogArtifact): LogPayloadContent {
  switch (artifact.contentState) {
    case 'available':
      return decodeAvailableContent(artifact)
    case 'unavailable':
      return {
        state: 'unavailable',
        ...(artifact.unavailableReason === undefined ? {} : { reason: artifact.unavailableReason })
      }
    case 'missing':
      return { state: 'missing' }
    case 'corrupt':
      return { state: 'corrupt' }
  }
}

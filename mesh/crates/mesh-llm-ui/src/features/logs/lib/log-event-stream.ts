export type LogEventStreamData =
  | { readonly state: 'json'; readonly text: string; readonly prettyText: string }
  | { readonly state: 'text'; readonly text: string }

export type LogEventStreamFrame = {
  readonly event?: string
  readonly id?: string
  readonly truncated?: true
  readonly data: LogEventStreamData
}

const MAX_EVENT_STREAM_FRAMES = 256
const MAX_EVENT_STREAM_DECODED_CHARS = 1_048_576
const TRUNCATED_PREVIEW_MESSAGE =
  'Additional response frames were omitted because the event stream exceeds the safe preview limit.'

type LogEventStreamJsonData =
  Extract<LogEventStreamData, { state: 'json' }> | { readonly state: 'malformed-json'; readonly text: string }

function decodeEventStreamData(text: string): LogEventStreamData {
  try {
    const value: unknown = JSON.parse(text)
    const prettyText = JSON.stringify(value, null, 2)
    const decoded: LogEventStreamJsonData =
      prettyText === undefined ? { state: 'malformed-json', text } : { state: 'json', text, prettyText }
    switch (decoded.state) {
      case 'json':
        return decoded
      case 'malformed-json':
        return { state: 'text', text: decoded.text }
      default:
        return assertNever(decoded)
    }
  } catch (error) {
    if (error instanceof SyntaxError) return { state: 'text', text }
    throw error
  }
}

export function decodeEventStream(text: string): readonly LogEventStreamFrame[] {
  const frames: LogEventStreamFrame[] = []
  const boundedText = text.slice(0, MAX_EVENT_STREAM_DECODED_CHARS + 1)
  const normalized = boundedText.replace(/\r\n?/g, '\n')
  const sourceTruncated = text.length > MAX_EVENT_STREAM_DECODED_CHARS
  const lastCompleteBlockEnd = sourceTruncated ? normalized.lastIndexOf('\n\n') : normalized.length
  const parseableText = lastCompleteBlockEnd < 0 ? '' : normalized.slice(0, lastCompleteBlockEnd)
  let decodedChars = 0
  let truncated = sourceTruncated

  parseBlocks: for (const block of eventStreamBlocks(parseableText)) {
    if (frames.length >= MAX_EVENT_STREAM_FRAMES - 1) {
      truncated = true
      break
    }
    const dataLines: string[] = []
    let event: string | undefined
    let id: string | undefined

    for (const line of block.split('\n')) {
      if (line.startsWith(':')) continue
      const separatorIndex = line.indexOf(':')
      const field = separatorIndex === -1 ? line : line.slice(0, separatorIndex)
      const rawValue = separatorIndex === -1 ? '' : line.slice(separatorIndex + 1)
      const value = rawValue.startsWith(' ') ? rawValue.slice(1) : rawValue
      const nextDecodedChars = decodedChars + value.length + (field === 'data' && dataLines.length > 0 ? 1 : 0)
      if (nextDecodedChars > MAX_EVENT_STREAM_DECODED_CHARS) {
        truncated = true
        break parseBlocks
      }
      decodedChars = nextDecodedChars
      switch (field) {
        case 'data':
          dataLines.push(value)
          break
        case 'event':
          event = value
          break
        case 'id':
          id = value
          break
      }
    }

    if (dataLines.length === 0) continue
    frames.push({
      ...(event === undefined ? {} : { event }),
      ...(id === undefined ? {} : { id }),
      data: decodeEventStreamData(dataLines.join('\n'))
    })
  }
  if (truncated) {
    frames.push({
      event: 'preview-truncated',
      truncated: true,
      data: { state: 'text', text: TRUNCATED_PREVIEW_MESSAGE }
    })
  }
  return frames
}

function* eventStreamBlocks(text: string): Iterable<string> {
  const separator = /\n\n+/g
  let blockStart = 0
  for (let match = separator.exec(text); match !== null; match = separator.exec(text)) {
    yield text.slice(blockStart, match.index)
    blockStart = separator.lastIndex
  }
  yield text.slice(blockStart)
}

function assertNever(value: never): never {
  throw new Error(`Unhandled event stream data: ${String(value)}`)
}

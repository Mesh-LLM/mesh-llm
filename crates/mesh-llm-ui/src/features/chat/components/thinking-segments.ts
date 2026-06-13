export type AssistantContentSegment =
  | {
      kind: 'thinking'
      text: string
      open: boolean
    }
  | {
      kind: 'response'
      text: string
    }

type SplitAssistantThinkingOptions = {
  streaming?: boolean
}

type TagMatch = {
  index: number
  tag: string
}

const THINK_OPEN_TAG = '<think>'
const THINK_CLOSE_TAG = '</think>'
const GEMMA_THOUGHT_CHANNEL_TAGS = ['<|channel|>thought', '<channel|>thought']
const GEMMA_CHANNEL_BOUNDARY_TAGS = ['<|channel|>', '<channel|>']

function indexOfTag(value: string, tag: string, fromIndex: number) {
  for (let index = fromIndex; index <= value.length - tag.length; index += 1) {
    if (value.slice(index, index + tag.length).toLowerCase() === tag) return index
  }

  return -1
}

function findFirstTag(value: string, tags: string[], fromIndex: number): TagMatch | null {
  let bestMatch: TagMatch | null = null

  for (const tag of tags) {
    const index = indexOfTag(value, tag, fromIndex)
    if (index === -1) continue
    if (bestMatch === null || index < bestMatch.index) {
      bestMatch = { index, tag }
    }
  }

  return bestMatch
}

function splitGemmaChannelThinking(body: string): AssistantContentSegment[] | null {
  if (findFirstTag(body, GEMMA_THOUGHT_CHANNEL_TAGS, 0) === null) return null

  const segments: AssistantContentSegment[] = []
  let cursor = 0

  while (cursor < body.length) {
    const open = findFirstTag(body, GEMMA_THOUGHT_CHANNEL_TAGS, cursor)
    if (open === null) {
      const responseText = body.slice(cursor)
      if (responseText.length > 0) {
        segments.push({ kind: 'response', text: responseText })
      }
      break
    }

    const responseText = body.slice(cursor, open.index)
    if (responseText.length > 0) {
      segments.push({ kind: 'response', text: responseText })
    }

    const thinkingStart = open.index + open.tag.length
    const nextChannel = findFirstTag(body, GEMMA_CHANNEL_BOUNDARY_TAGS, thinkingStart)
    if (nextChannel === null) {
      segments.push({ kind: 'thinking', text: body.slice(thinkingStart), open: true })
      break
    }

    const thinkingText = body.slice(thinkingStart, nextChannel.index)
    if (thinkingText.length > 0) {
      segments.push({ kind: 'thinking', text: thinkingText, open: false })
    }
    cursor = nextChannel.index + nextChannel.tag.length
  }

  return segments
}

export function splitAssistantThinking(
  body: string,
  { streaming = false }: SplitAssistantThinkingOptions = {}
): AssistantContentSegment[] {
  if (body.length === 0) return []

  const gemmaSegments = splitGemmaChannelThinking(body)
  if (gemmaSegments !== null) return gemmaSegments

  const segments: AssistantContentSegment[] = []
  let cursor = 0
  let firstSegment = true

  if (streaming && indexOfTag(body, THINK_OPEN_TAG, 0) === -1 && indexOfTag(body, THINK_CLOSE_TAG, 0) === -1) {
    return [{ kind: 'thinking', text: body, open: true }]
  }

  while (cursor < body.length) {
    const openIndex = indexOfTag(body, THINK_OPEN_TAG, cursor)
    const closeIndex = indexOfTag(body, THINK_CLOSE_TAG, cursor)

    if (closeIndex !== -1 && (openIndex === -1 || closeIndex < openIndex)) {
      const thinkingText = body.slice(cursor, closeIndex)
      if (thinkingText.length > 0) {
        segments.push({ kind: 'thinking', text: thinkingText, open: false })
      }
      cursor = closeIndex + THINK_CLOSE_TAG.length
      firstSegment = false
      continue
    }

    if (openIndex === -1) {
      const responseText = body.slice(cursor)
      if (responseText.length > 0) {
        segments.push({ kind: 'response', text: responseText })
      }
      break
    }

    const responseText = body.slice(cursor, openIndex)
    if (responseText.length > 0 || (firstSegment && openIndex > cursor)) {
      segments.push({ kind: 'response', text: responseText })
    }

    const thinkingStart = openIndex + THINK_OPEN_TAG.length
    const thinkingEnd = indexOfTag(body, THINK_CLOSE_TAG, thinkingStart)
    if (thinkingEnd === -1) {
      const thinkingText = body.slice(thinkingStart)
      segments.push({ kind: 'thinking', text: thinkingText, open: true })
      break
    }

    const thinkingText = body.slice(thinkingStart, thinkingEnd)
    if (thinkingText.length > 0) {
      segments.push({ kind: 'thinking', text: thinkingText, open: false })
    }
    cursor = thinkingEnd + THINK_CLOSE_TAG.length
    firstSegment = false
  }

  return segments
}

type ProtectedRange = {
  start: number
  end: number
}

type Fence = {
  character: '`' | '~'
  length: number
}

function isEscapedAt(value: string, index: number): boolean {
  let backslashes = 0
  for (let cursor = index - 1; cursor >= 0 && value[cursor] === '\\'; cursor -= 1) {
    backslashes += 1
  }
  return backslashes % 2 === 1
}

function findUnescapedSequence(value: string, sequence: string, start: number): number {
  let index = value.indexOf(sequence, start)
  while (index >= 0) {
    if (!isEscapedAt(value, index)) return index
    index = value.indexOf(sequence, index + sequence.length)
  }
  return -1
}

function fenceFromLine(line: string): Fence | undefined {
  const match = /^( {0,3})(`{3,}|~{3,})(.*)$/.exec(line)
  if (!match) return undefined

  const marker = match[2]
  const character = marker[0] as Fence['character']
  if (character === '`' && match[3].includes('`')) return undefined

  return { character, length: marker.length }
}

function isFenceClose(line: string, fence: Fence): boolean {
  const match = /^( {0,3})(`+|~+)\s*$/.exec(line)
  if (!match) return false

  const marker = match[2]
  return marker[0] === fence.character && marker.length >= fence.length
}

function fencedCodeRanges(value: string): ProtectedRange[] {
  const ranges: ProtectedRange[] = []
  let fence: { descriptor: Fence; start: number } | undefined
  let lineStart = 0

  while (lineStart < value.length) {
    const newline = value.indexOf('\n', lineStart)
    const lineEnd = newline === -1 ? value.length : newline
    const line = value.slice(lineStart, lineEnd)

    if (fence) {
      if (isFenceClose(line, fence.descriptor)) {
        ranges.push({ start: fence.start, end: newline === -1 ? value.length : newline + 1 })
        fence = undefined
      }
    } else {
      const descriptor = fenceFromLine(line)
      if (descriptor) fence = { descriptor, start: lineStart }
    }

    if (newline === -1) break
    lineStart = newline + 1
  }

  if (fence) ranges.push({ start: fence.start, end: value.length })
  return ranges
}

function inlineCodeRanges(value: string, fencedRanges: readonly ProtectedRange[]): ProtectedRange[] {
  const ranges: ProtectedRange[] = []
  let index = 0
  let fenceIndex = 0

  while (index < value.length) {
    const fence = fencedRanges[fenceIndex]
    if (fence && index >= fence.start) {
      index = fence.end
      fenceIndex += 1
      continue
    }

    if (value[index] !== '`') {
      index += 1
      continue
    }

    let length = 1
    while (value[index + length] === '`') length += 1

    let close = value.indexOf('`'.repeat(length), index + length)
    while (close >= 0 && isEscapedAt(value, close)) {
      close = value.indexOf('`'.repeat(length), close + length)
    }

    if (close < 0) {
      index += length
      continue
    }

    ranges.push({ start: index, end: close + length })
    index = close + length
  }

  return ranges
}

function protectedRanges(value: string): ProtectedRange[] {
  const fencedRanges = fencedCodeRanges(value)
  return [...fencedRanges, ...inlineCodeRanges(value, fencedRanges)].sort((left, right) => left.start - right.start)
}

function displayMath(value: string): string {
  return `\n\n$$\n${value.trim()}\n$$\n\n`
}

function isMarkdownReferenceDefinition(value: string, start: number, close: number): boolean {
  if (value[close + 1] !== ':') return false

  const lineStart = value.lastIndexOf('\n', start - 1) + 1
  return /^ {0,3}$/.test(value.slice(lineStart, start))
}

function normalizedReferenceLabel(value: string): string {
  return value.trim().replace(/\s+/g, ' ').toLowerCase()
}

function collectReferenceDefinitionLabels(value: string, start: number, end: number, labels: Set<string>): void {
  let index = start
  while (index < end) {
    if (value[index] !== '[' || isEscapedAt(value, index)) {
      index += 1
      continue
    }

    const close = findUnescapedSequence(value, ']', index + 1)
    if (close < 0 || close >= end) {
      index += 1
      continue
    }

    if (isMarkdownReferenceDefinition(value, index, close)) {
      labels.add(normalizedReferenceLabel(value.slice(index + 1, close)))
    }
    index = close + 1
  }
}

function shortcutReferenceLabels(value: string, ranges: readonly ProtectedRange[]): ReadonlySet<string> {
  const labels = new Set<string>()
  let cursor = 0

  for (const range of ranges) {
    if (range.start < cursor) continue
    collectReferenceDefinitionLabels(value, cursor, range.start, labels)
    cursor = range.end
  }
  collectReferenceDefinitionLabels(value, cursor, value.length, labels)
  return labels
}

function normalizePlainText(value: string, shortcutLabels: ReadonlySet<string>): string {
  let normalized = ''
  let index = 0

  while (index < value.length) {
    if (value.startsWith('\\[', index) && !isEscapedAt(value, index)) {
      const close = findUnescapedSequence(value, '\\]', index + 2)
      if (close >= 0) {
        normalized += displayMath(value.slice(index + 2, close))
        index = close + 2
        continue
      }
      normalized += value.slice(index, index + 2)
      index += 2
      continue
    }

    if (value.startsWith('\\(', index) && !isEscapedAt(value, index)) {
      const close = findUnescapedSequence(value, '\\)', index + 2)
      if (close >= 0) {
        normalized += `$${value.slice(index + 2, close)}$`
        index = close + 2
        continue
      }
      normalized += value.slice(index, index + 2)
      index += 2
      continue
    }

    if (value.startsWith('$$', index) && !isEscapedAt(value, index)) {
      const close = findUnescapedSequence(value, '$$', index + 2)
      if (close >= 0) {
        normalized += displayMath(value.slice(index + 2, close))
        index = close + 2
        continue
      }
      normalized += '$$'
      index += 2
      continue
    }

    if (value[index] === '[' && !isEscapedAt(value, index) && value[index - 1] !== '!') {
      const close = findUnescapedSequence(value, ']', index + 1)
      if (close >= 0) {
        const inner = value.slice(index + 1, close)
        const followingCharacter = value[close + 1]
        const isMarkdownLink = followingCharacter === '(' || followingCharacter === '['
        const isMarkdownDefinition = isMarkdownReferenceDefinition(value, index, close)
        const isShortcutReference = shortcutLabels.has(normalizedReferenceLabel(inner))
        if (!isMarkdownLink && !isMarkdownDefinition && !isShortcutReference && /\\[A-Za-z]+/.test(inner)) {
          normalized += `$${inner}$`
          index = close + 1
          continue
        }
      }
    }

    normalized += value[index]
    index += 1
  }

  return normalized
}

function preservePlainTextMathDelimiters(value: string): string {
  let preserved = ''
  let index = 0

  while (index < value.length) {
    if (
      (value.startsWith('\\[', index) ||
        value.startsWith('\\]', index) ||
        value.startsWith('\\(', index) ||
        value.startsWith('\\)', index)) &&
      !isEscapedAt(value, index)
    ) {
      preserved += `\\${value.slice(index, index + 2)}`
      index += 2
      continue
    }

    preserved += value[index]
    index += 1
  }

  return preserved
}

/**
 * Normalize the common TeX delimiters emitted by chat models to remark-math's
 * dollar syntax while leaving Markdown code spans and fenced code untouched.
 */
export function normalizeMathDelimiters(value: string): string {
  const ranges = protectedRanges(value)
  const shortcutLabels = shortcutReferenceLabels(value, ranges)
  if (ranges.length === 0) return normalizePlainText(value, shortcutLabels)

  let normalized = ''
  let cursor = 0
  for (const range of ranges) {
    if (range.start < cursor) continue
    normalized += normalizePlainText(value.slice(cursor, range.start), shortcutLabels)
    normalized += value.slice(range.start, range.end)
    cursor = range.end
  }
  return normalized + normalizePlainText(value.slice(cursor), shortcutLabels)
}

/**
 * Keep TeX-style delimiters visible while a response is still streaming.
 * Markdown remains enabled for emphasis and code, but escaped punctuation must
 * be doubled so CommonMark does not silently remove the model's backslash.
 */
export function preserveMathDelimiters(value: string): string {
  const ranges = protectedRanges(value)
  if (ranges.length === 0) return preservePlainTextMathDelimiters(value)

  let preserved = ''
  let cursor = 0
  for (const range of ranges) {
    if (range.start < cursor) continue
    preserved += preservePlainTextMathDelimiters(value.slice(cursor, range.start))
    preserved += value.slice(range.start, range.end)
    cursor = range.end
  }
  return preserved + preservePlainTextMathDelimiters(value.slice(cursor))
}

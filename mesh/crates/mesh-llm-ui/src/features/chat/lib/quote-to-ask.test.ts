import { describe, expect, it } from 'vitest'

import {
  ASSISTANT_QUOTE_SOURCE,
  QUOTE_SOURCE_ATTR,
  extractQuoteFromSelection,
  mergeQuoteIntoPrompt,
  normalizeQuoteText,
  quoteToBlockquote,
  type RangeLike,
  type SelectionLike
} from './quote-to-ask'

function fakeSelection(range: RangeLike | null): SelectionLike {
  return {
    rangeCount: range ? 1 : 0,
    isCollapsed: range ? range.collapsed : true,
    getRangeAt: () => {
      if (!range) throw new Error('no range')
      return range
    }
  }
}

function rangeIn(container: Node, text: string, collapsed = false): RangeLike {
  return { collapsed, toString: () => text, commonAncestorContainer: container }
}

describe('normalizeQuoteText', () => {
  it('collapses CRLF and trims trailing per-line whitespace', () => {
    expect(normalizeQuoteText('a \r\nb\t\r\nc ')).toBe('a\nb\nc')
  })

  it('squeezes runs of blank lines down to one blank line', () => {
    expect(normalizeQuoteText('a\n\n\n\nb')).toBe('a\n\nb')
  })

  it('trims outer whitespace and returns empty for whitespace-only input', () => {
    expect(normalizeQuoteText('  \n x \n ')).toBe('x')
    expect(normalizeQuoteText(' \n\t ')).toBe('')
  })

  it('caps length at MAX_QUOTE_LENGTH plus an ellipsis marker', () => {
    const long = 'x'.repeat(5000)
    const normalized = normalizeQuoteText(long)
    expect(normalized.length).toBe(4001)
    expect(normalized.endsWith('…')).toBe(true)
  })
})

describe('quoteToBlockquote', () => {
  it('prefixes every non-empty line and keeps empty lines as bare markers', () => {
    expect(quoteToBlockquote('first\n\nsecond')).toBe('> first\n>\n> second')
  })
})

describe('mergeQuoteIntoPrompt', () => {
  it('places the blockquote above the question', () => {
    const merged = mergeQuoteIntoPrompt({ id: 'q', text: 'quoted passage' }, 'why?')
    expect(merged).toBe('> quoted passage\n\nwhy?')
  })

  it('returns the prompt untouched without a quote or with blank quote text', () => {
    expect(mergeQuoteIntoPrompt(null, 'why?')).toBe('why?')
    expect(mergeQuoteIntoPrompt(undefined, 'why?')).toBe('why?')
    expect(mergeQuoteIntoPrompt({ id: 'q', text: '   ' }, 'why?')).toBe('why?')
  })

  it('returns just the blockquote when the prompt is empty', () => {
    expect(mergeQuoteIntoPrompt({ id: 'q', text: 'quoted' }, '   ')).toBe('> quoted')
  })

  it('normalizes multi-line quotes into per-line markers', () => {
    const merged = mergeQuoteIntoPrompt({ id: 'q', text: 'line one\nline two' }, 'and?')
    expect(merged).toBe('> line one\n> line two\n\nand?')
  })
})

describe('extractQuoteFromSelection', () => {
  function assistantBody(): HTMLElement {
    const body = document.createElement('div')
    body.setAttribute(QUOTE_SOURCE_ATTR, ASSISTANT_QUOTE_SOURCE)
    document.body.appendChild(body)
    return body
  }

  function selectionInside(container: HTMLElement, text: string, collapsed = false): SelectionLike {
    const textNode = document.createTextNode(text)
    container.appendChild(textNode)
    return fakeSelection(rangeIn(textNode, text, collapsed))
  }

  it('extracts a quote from a selection inside an assistant body', () => {
    const body = assistantBody()
    const quote = extractQuoteFromSelection(selectionInside(body, '  selected passage  '))
    expect(quote?.text).toBe('selected passage')
    expect(quote?.id).toMatch(/^quote-\d+$/)
  })

  it('finds the owner through nested elements and text nodes', () => {
    const body = assistantBody()
    const paragraph = document.createElement('p')
    body.appendChild(paragraph)
    const quote = extractQuoteFromSelection(selectionInside(paragraph, 'nested text'))
    expect(quote?.text).toBe('nested text')
  })

  it('returns null for a collapsed selection, no ranges, or empty text', () => {
    const body = assistantBody()
    expect(extractQuoteFromSelection(fakeSelection(null))).toBeNull()
    expect(extractQuoteFromSelection(selectionInside(body, 'x', true))).toBeNull()
    expect(extractQuoteFromSelection(selectionInside(body, '   \n  '))).toBeNull()
  })

  it('returns null outside any quote-source element', () => {
    const outside = document.createElement('div')
    document.body.appendChild(outside)
    expect(extractQuoteFromSelection(selectionInside(outside, 'selected'))).toBeNull()
  })

  it('returns null for selections in a user message', () => {
    const userBody = document.createElement('div')
    userBody.setAttribute(QUOTE_SOURCE_ATTR, 'user')
    document.body.appendChild(userBody)
    expect(extractQuoteFromSelection(selectionInside(userBody, 'my own words'))).toBeNull()
  })
})

describe('extractQuoteFromSelection cross-message scope', () => {
  it('returns null when a selection spans two assistant messages', () => {
    const list = document.createElement('div')
    document.body.appendChild(list)
    const first = document.createElement('div')
    first.setAttribute('data-quote-source', 'assistant')
    const firstText = document.createTextNode('first answer passage')
    first.appendChild(firstText)
    const second = document.createElement('div')
    second.setAttribute('data-quote-source', 'assistant')
    const secondText = document.createTextNode('second answer passage')
    second.appendChild(secondText)
    list.append(first, second)

    const range = document.createRange()
    range.setStart(firstText, 0)
    range.setEnd(secondText, secondText.length)
    const selection = document.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)

    const quote = extractQuoteFromSelection({
      rangeCount: 1,
      isCollapsed: false,
      getRangeAt: () => range
    })
    expect(quote).toBeNull()
  })
})

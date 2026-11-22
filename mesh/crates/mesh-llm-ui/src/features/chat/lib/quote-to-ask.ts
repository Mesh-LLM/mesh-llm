/**
 * Quote-to-ask: the user selects a passage of an assistant answer and asks a
 * follow-up scoped to it. The selection becomes a composer quote chip, and at
 * send time it merges into the outgoing prompt as a visible blockquote, so
 * what the user sees is exactly what the model receives.
 */

export const QUOTE_SOURCE_ATTR = 'data-quote-source'
export const ASSISTANT_QUOTE_SOURCE = 'assistant'

/** Selections longer than this are truncated; a quote is context, not a paste. */
export const MAX_QUOTE_LENGTH = 4000

export type ComposerQuote = {
  id: string
  text: string
}

/** The DOM surface `extractQuoteFromSelection` needs; satisfied by `Selection`. */
export type SelectionLike = {
  rangeCount: number
  isCollapsed: boolean
  getRangeAt(index: number): RangeLike
}

export type RangeLike = {
  collapsed: boolean
  toString(): string
  commonAncestorContainer: Node
}

let nextQuoteId = 0

/** Collapses CRLF, strips trailing whitespace per line, squeezes blank runs, caps length. */
export function normalizeQuoteText(raw: string): string {
  const normalized = raw
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map((line) => line.replace(/[ \t]+$/g, ''))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
  if (normalized.length <= MAX_QUOTE_LENGTH) return normalized
  return `${normalized.slice(0, MAX_QUOTE_LENGTH).trimEnd()}…`
}

/** Renders quote text as a markdown blockquote (one `> ` prefix per line). */
export function quoteToBlockquote(text: string): string {
  return text
    .split('\n')
    .map((line) => (line.length === 0 ? '>' : `> ${line}`))
    .join('\n')
}

/**
 * Merges the pending quote into the outgoing prompt: the blockquote leads,
 * the user's question follows. No quote (or quote without text) returns the
 * prompt untouched.
 */
export function mergeQuoteIntoPrompt(quote: ComposerQuote | null | undefined, prompt: string): string {
  const text = quote?.text?.trim()
  if (!text) return prompt
  const blockquote = quoteToBlockquote(normalizeQuoteText(text))
  const question = prompt.trim()
  return question.length === 0 ? blockquote : `${blockquote}\n\n${question}`
}

/**
 * Extracts a quote from a DOM selection when it is anchored inside a single
 * assistant message body (marked with `data-quote-source="assistant"`).
 * Selections in user messages, outside messages, collapsed, or effectively
 * empty return null — the caller then leaves the composer alone.
 */
export function extractQuoteFromSelection(selection: SelectionLike): ComposerQuote | null {
  if (selection.rangeCount === 0 || selection.isCollapsed) return null
  const range = selection.getRangeAt(0)
  if (range.collapsed) return null
  const text = normalizeQuoteText(range.toString())
  if (text.length === 0) return null
  const owner = quoteSourceOwner(range.commonAncestorContainer)
  if (owner !== ASSISTANT_QUOTE_SOURCE) return null
  return { id: `quote-${nextQuoteId++}`, text }
}

function quoteSourceOwner(node: Node): string | null {
  let current: Node | null = node
  while (current) {
    if (current instanceof Element) {
      const source = current.getAttribute(QUOTE_SOURCE_ATTR)
      if (source !== null) return source
    }
    current = current.parentElement ?? current.parentNode
  }
  return null
}

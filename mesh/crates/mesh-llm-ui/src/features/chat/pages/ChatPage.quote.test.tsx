import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { chatMock, renderChatPage } from './ChatPage.test-support'

beforeEach(() => {
  // jsdom Ranges have no layout; the popover positions from a rect.
  ;(Range.prototype as { getBoundingClientRect?: unknown }).getBoundingClientRect = () => new DOMRect(10, 10, 120, 24)
})

afterEach(() => {
  delete (Range.prototype as { getBoundingClientRect?: unknown }).getBoundingClientRect
  document.getSelection()?.removeAllRanges()
})

/** Selects the opening characters of the latest rendered assistant answer
 *  (the quote-to-ask marker) and returns the selected text. */
function selectAssistantAnswerPrefix(length = 7) {
  const marker = document.querySelector('[data-quote-source="assistant"]')
  expect(marker).not.toBeNull()
  const walker = document.createTreeWalker(marker as Node, NodeFilter.SHOW_TEXT)
  const node = walker.nextNode() as Text | null
  expect(node).not.toBeNull()
  const text = node?.textContent ?? ''
  const range = document.createRange()
  range.setStart(node as Text, 0)
  range.setEnd(node as Text, Math.min(text.trimStart().length, length))
  const selection = document.getSelection()
  selection?.removeAllRanges()
  selection?.addRange(range)
  return range.toString().trim()
}

/** The page focuses the composer on mount; selecting before that focus
 *  lands gets collapsed by it. Wait for the textarea to own focus first. */
async function waitForComposerFocus() {
  await waitFor(() => expect(document.activeElement).toBeInstanceOf(HTMLTextAreaElement))
}

async function acceptQuoteThroughPopover() {
  fireEvent.mouseUp(document.body)
  const popoverButton = await screen.findByRole('button', { name: /ask about this/i })
  fireEvent.click(popoverButton)
  await screen.findByTestId('composer-quote')
}

describe('ChatPage quote-to-ask wiring', () => {
  it('merges an accepted quote into the outgoing prompt and clears the chip on send', async () => {
    const user = userEvent.setup()

    renderChatPage()
    await waitForComposerFocus()

    const quoted = selectAssistantAnswerPrefix()
    await acceptQuoteThroughPopover()

    const chip = screen.getByTestId('composer-quote')
    expect(chip).toHaveTextContent(quoted)

    await user.type(screen.getByLabelText('Prompt'), 'what did you mean?')
    await user.click(screen.getByRole('button', { name: /^(Send|Queue)$/ }))

    await waitFor(() => expect(chatMock.sendCalls).toHaveLength(1))
    const sentText = JSON.stringify(chatMock.sendCalls[0].content)
    expect(sentText).toContain(`> ${quoted}`)
    expect(sentText).toContain('what did you mean?')
    await waitFor(() => expect(screen.queryByTestId('composer-quote')).not.toBeInTheDocument())
  })
})

describe('ChatPage quote-to-ask queue-drain restore', () => {
  it('restores the typed prompt and quote chip when a queued send fails after the drain', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    // First send: seeds the conversation and leaves status as 'streaming'
    await user.type(screen.getByLabelText('Prompt'), 'Answer me first')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    await waitFor(() => expect(screen.getByText('Partial assistant reply')).toBeInTheDocument())

    selectAssistantAnswerPrefix()
    fireEvent.mouseUp(document.body)
    const popoverButton = await screen.findByRole('button', { name: /ask about this/i })
    fireEvent.click(popoverButton)
    await screen.findByTestId('composer-quote')

    // Type the follow-up
    await user.type(screen.getByLabelText('Prompt'), 'queued follow-up')

    // The queued send should fail
    chatMock.sendErrorMessage = 'Queue drain send failed'

    // Send while streaming: goes to the queue (composer clears, no send yet)
    await user.click(screen.getByRole('button', { name: /^(Send|Queue)$/ }))
    await waitFor(() => expect(screen.getByLabelText('Prompt')).toHaveValue(''))
    expect(chatMock.sendCalls).toHaveLength(1)

    // Stop streaming: this sets status to 'ready', which drains the queue
    const stopButton = screen.getByRole('button', { name: 'Stop' })
    await user.click(stopButton)

    // The queue drain sends (proving the message was queued, not direct)
    await waitFor(() => expect(chatMock.sendCalls).toHaveLength(2), { timeout: 3000 })

    // The drained send fails; the composer restores the typed prompt + chip
    await waitFor(() => expect(screen.getByLabelText('Prompt')).toHaveValue('queued follow-up'), { timeout: 3000 })
    expect(screen.getByTestId('composer-quote')).toBeInTheDocument()
    expect(screen.getByLabelText('Prompt')).not.toHaveValue(expect.stringContaining('>'))
  })
})

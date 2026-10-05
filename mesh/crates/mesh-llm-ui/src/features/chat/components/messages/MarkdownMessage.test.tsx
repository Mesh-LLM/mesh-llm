import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MessageRow } from '@/features/chat/components/MessageRow'
import { MarkdownMessage } from '@/features/chat/components/messages/MarkdownMessage'

async function waitForMath(container: HTMLElement, count = 1) {
  await waitFor(() => expect(container.querySelectorAll('.katex')).toHaveLength(count))
}

describe('chat math rendering', () => {
  it('renders inline and display math in the active MessageRow pipeline', async () => {
    const { container } = render(
      <MessageRow
        messageRole="assistant"
        body={'Inline $x^2 + y^2 = z^2$.\n\n$$\n\\sum_{i=1}^n i\n$$'}
        timestamp="12:11"
        model="Qwen3-8B"
      />
    )

    await waitForMath(container, 2)

    expect(container.querySelector('.katex-display')).toBeInTheDocument()
    expect(container.querySelector('.katex-display')?.closest('pre')).not.toBeInTheDocument()
    expect(container.querySelectorAll('annotation[encoding="application/x-tex"]')).toHaveLength(2)
  })

  it('treats a same-line double-dollar pair as display math', async () => {
    const { container } = render(<MarkdownMessage content="The result is $$x^2$$." />)

    await waitForMath(container)

    expect(container.querySelector('.katex-display')).toBeInTheDocument()
  })

  it('normalizes the issue reproduction bracket into inline math', async () => {
    const issueExample = String.raw`[ 4 \rightarrow 2 \rightarrow 1 \rightarrow 4 \rightarrow 2 \rightarrow 1 \rightarrow \cdots ]`
    const { container } = render(
      <MessageRow messageRole="assistant" body={issueExample} timestamp="12:11" model="Qwen3-8B" />
    )

    await waitForMath(container)

    expect(container.querySelector('.katex')).toBeInTheDocument()
    expect(container.querySelector('annotation')?.textContent).toBe(
      '4 \\rightarrow 2 \\rightarrow 1 \\rightarrow 4 \\rightarrow 2 \\rightarrow 1 \\rightarrow \\cdots'
    )
  })

  it('supports parenthesized and bracketed TeX delimiters', async () => {
    const { container } = render(<MarkdownMessage content={'Inline \\(x^2\\) and display\\n\\[\\frac{1}{2}\\]'} />)

    await waitForMath(container, 2)

    expect(container.querySelector('.katex-display')).toBeInTheDocument()
    expect(container.querySelectorAll('annotation[encoding="application/x-tex"]')[0]).toHaveTextContent('x^2')
    expect(container.querySelectorAll('annotation[encoding="application/x-tex"]')[1]).toHaveTextContent('\\frac{1}{2}')
  })

  it('keeps ordinary square brackets and escaped dollar text literal', async () => {
    const { container } = render(
      <MarkdownMessage content={'[ordinary square brackets] and escaped \\$5 plus \\$ is literal.'} />
    )

    expect(container).toHaveTextContent('[ordinary square brackets] and escaped $5 plus $ is literal.')
    expect(container.querySelector('.katex')).not.toBeInTheDocument()
  })

  it('preserves full and collapsed reference links that resemble bracketed TeX', () => {
    const content = String.raw`[\alpha][formula] and [\beta][]

[formula]: https://example.com/formula
[\beta]: https://example.com/beta`
    const { container } = render(<MarkdownMessage content={content} />)

    expect(screen.getByRole('link', { name: String.raw`\alpha` })).toHaveAttribute(
      'href',
      'https://example.com/formula'
    )
    expect(screen.getByRole('link', { name: String.raw`\beta` })).toHaveAttribute('href', 'https://example.com/beta')
    expect(container.querySelector('.katex')).not.toBeInTheDocument()
  })

  it('preserves shortcut reference links that resemble bracketed TeX', () => {
    const content = String.raw`[\alpha]

[\alpha]: https://example.test`
    const { container } = render(<MarkdownMessage content={content} />)

    expect(screen.getByRole('link', { name: String.raw`\alpha` })).toHaveAttribute('href', 'https://example.test')
    expect(container.querySelector('.katex')).not.toBeInTheDocument()
  })

  it('does not reinterpret inline or fenced code as math', async () => {
    const fenced = ['```text', '$x$ $$y$$ \\[z\\] \\(w\\)', '```'].join('\n')
    const { container } = render(
      <MarkdownMessage content={'Inline `$a$` `$$b$$` `\\[c\\]` and `\\(d\\)`\n\n' + fenced} />
    )

    expect(container.querySelector('.katex')).not.toBeInTheDocument()
    expect(container).toHaveTextContent('Inline $a$ $$b$$ \\[c\\] and \\(d\\)')
    expect(container.querySelector('pre code')).toHaveTextContent('$x$ $$y$$ \\[z\\] \\(w\\)')
  })

  it('leaves complete and partial math delimiters readable while streaming', () => {
    const { container } = render(
      <MessageRow
        messageRole="assistant"
        body={'Complete $x^2$ and partial \\[y^2'}
        timestamp="12:11"
        model="Qwen3-8B"
        state="streaming"
      />
    )

    expect(container.querySelector('.katex')).not.toBeInTheDocument()
    expect(container).toHaveTextContent('Complete $x^2$ and partial \\[y^2')
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toBeInTheDocument()
  })

  it('keeps user and error rows as plain text fallbacks', () => {
    const { container } = render(
      <>
        <MessageRow messageRole="user" body={'[ 4 \\rightarrow 2 ]'} timestamp="12:11" />
        <MessageRow messageRole="assistant" state="error" body={'[ 4 \\rightarrow 2 ]'} timestamp="12:12" />
      </>
    )

    expect(container.querySelector('.katex')).not.toBeInTheDocument()
    expect(container).toHaveTextContent('[ 4 \\rightarrow 2 ]')
    expect(screen.getByRole('alert')).toHaveTextContent('Message failed to send')
  })

  it('keeps rendered assistant content selectable for native copy', async () => {
    const { container } = render(
      <MessageRow messageRole="assistant" body={'Copy this $x^2$ and `code`.'} timestamp="12:11" model="Qwen3-8B" />
    )
    await waitForMath(container)

    const selectable = container.querySelector('.select-text')
    expect(selectable).toBeInTheDocument()

    const selection = window.getSelection()
    const range = document.createRange()
    range.selectNodeContents(selectable as Node)
    selection?.removeAllRanges()
    selection?.addRange(range)

    const copyEvent = new Event('copy', { bubbles: true, cancelable: true })
    expect(selectable?.dispatchEvent(copyEvent)).toBe(true)
    expect(copyEvent.defaultPrevented).toBe(false)
    expect(selection?.toString()).toContain('Copy this')

    selection?.removeAllRanges()
  })

  it('retains safe external link handling in the shared renderer', () => {
    render(<MarkdownMessage content={'[safe](https://example.com) [unsafe](javascript:alert(1))'} />)

    const safeLink = screen.getByRole('link', { name: 'safe' })
    expect(safeLink).toHaveAttribute('href', 'https://example.com')
    expect(safeLink).toHaveAttribute('target', '_blank')
    expect(safeLink).toHaveAttribute('rel', 'noreferrer noopener')
    expect(screen.queryByRole('link', { name: 'unsafe' })).not.toBeInTheDocument()
  })

  it('does not opt into raw HTML rendering', () => {
    const { container } = render(<MarkdownMessage content={'<img src="x" onerror="alert(1)" /> **safe text**'} />)

    expect(container.querySelector('img')).not.toBeInTheDocument()
    expect(container.querySelector('[onerror]')).not.toBeInTheDocument()
    expect(screen.getByText('safe text')).toBeInTheDocument()
  })
})

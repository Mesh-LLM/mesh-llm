import ReactMarkdown from 'react-markdown'
import type { ExtraProps } from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { cn } from '@/lib/utils'
import { KaTeXBlock } from '@/features/chat/components/messages/KaTeXBlock'
import { MermaidBlock } from '@/features/chat/components/messages/MermaidBlock'
import { normalizeMathDelimiters, preserveMathDelimiters } from '@/features/chat/components/messages/math-delimiters'

export type MarkdownMessageProps = {
  content: string
  streaming?: boolean
  linksEnabled?: boolean
  variant?: 'default' | 'thinking'
  className?: string
}

function isEnhancedCodeBlock(node: ExtraProps['node']) {
  const firstChild = node?.children?.[0]
  if (!firstChild || firstChild.type !== 'element') return false

  const className = firstChild.properties?.className
  const classes = Array.isArray(className) ? className.map(String) : String(className ?? '').split(/\s+/)
  return classes.includes('language-math') || classes.includes('language-mermaid')
}

export function MarkdownMessage({
  content,
  streaming = false,
  linksEnabled = true,
  variant = 'default',
  className
}: MarkdownMessageProps) {
  const markdown = streaming ? preserveMathDelimiters(content) : normalizeMathDelimiters(content)
  const remarkPlugins = streaming ? [remarkGfm] : [remarkGfm, remarkMath]

  return (
    <div
      className={cn(
        'block select-text break-words text-sm leading-6',
        '[&_a]:underline [&_a]:underline-offset-2',
        '[&_blockquote]:my-2 [&_blockquote]:block [&_blockquote]:border-l [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-fg-dim',
        '[&_code]:rounded-[calc(var(--radius)-2px)] [&_code]:bg-panel-strong [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.93em]',
        '[&_em]:text-fg-dim [&_strong]:font-semibold [&_strong]:text-foreground',
        '[&_h1]:mb-2 [&_h1]:mt-3 [&_h1]:block [&_h1]:text-[length:var(--density-type-title)] [&_h1]:font-semibold [&_h1:first-child]:mt-0',
        '[&_h2]:mb-2 [&_h2]:mt-3 [&_h2]:block [&_h2]:text-[length:var(--density-type-control-lg)] [&_h2]:font-semibold [&_h2:first-child]:mt-0',
        '[&_h3]:mb-1.5 [&_h3]:mt-3 [&_h3]:block [&_h3]:text-[length:var(--density-type-body-lg)] [&_h3]:font-semibold [&_h3:first-child]:mt-0',
        '[&_hr]:my-3 [&_hr]:block [&_hr]:border-t [&_hr]:border-border-soft',
        '[&_li]:my-0.5 [&_li]:pl-1 [&_li]:marker:text-fg-faint [&_li>p]:my-0',
        '[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5',
        '[&_p]:my-2 [&_p]:block [&_p:first-child]:mt-0 [&_p:last-child]:mb-0',
        '[&_pre]:my-2 [&_pre]:block [&_pre]:max-w-full [&_pre]:overflow-x-auto [&_pre]:whitespace-pre [&_pre]:rounded-[var(--radius)] [&_pre]:border [&_pre]:border-border-soft [&_pre]:bg-panel [&_pre]:p-3 [&_pre_code]:bg-transparent [&_pre_code]:p-0',
        '[&_table]:my-2 [&_table]:w-full [&_table]:border-collapse [&_table]:text-[length:var(--density-type-caption)]',
        '[&_td]:border [&_td]:border-border-soft [&_td]:px-2 [&_td]:py-1 [&_td]:align-top [&_td]:text-fg-dim',
        '[&_th]:border [&_th]:border-border-soft [&_th]:bg-panel-strong [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_th]:font-semibold [&_th]:text-foreground',
        '[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5',
        variant === 'thinking' && '[&_strong]:text-fg-muted',
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={[rehypeHighlight]}
        components={{
          a(props) {
            const { node, ...anchorProps } = props
            void node

            if (!linksEnabled) {
              return (
                <span className="text-accent underline underline-offset-2" title={anchorProps.href}>
                  {anchorProps.children}
                </span>
              )
            }

            return <a {...anchorProps} rel="noreferrer noopener" target="_blank" />
          },
          blockquote(props) {
            const { node, ...blockquoteProps } = props
            void node
            return <blockquote {...blockquoteProps} className="my-2 block border-l border-border pl-3 text-fg-dim" />
          },
          code({ node, className: codeClassName, children, ...props }) {
            void node
            const text = String(children).replace(/\n$/, '')
            if (!streaming) {
              if (/language-mermaid/.test(codeClassName || '')) return <MermaidBlock code={text} />
              if (/language-math/.test(codeClassName || '')) {
                return <KaTeXBlock math={text} display={/math-display/.test(codeClassName || '')} />
              }
            }
            return (
              <code className={codeClassName} {...props}>
                {children}
              </code>
            )
          },
          h1(props) {
            const { node, ...headingProps } = props
            void node
            return (
              <h1
                {...headingProps}
                className="mb-2 mt-3 block text-[length:var(--density-type-title)] font-semibold first:mt-0"
              />
            )
          },
          h2(props) {
            const { node, ...headingProps } = props
            void node
            return (
              <h2
                {...headingProps}
                className="mb-2 mt-3 block text-[length:var(--density-type-control-lg)] font-semibold first:mt-0"
              />
            )
          },
          h3(props) {
            const { node, ...headingProps } = props
            void node
            return (
              <h3
                {...headingProps}
                className="mb-1.5 mt-3 block text-[length:var(--density-type-body-lg)] font-semibold first:mt-0"
              />
            )
          },
          hr(props) {
            const { node, ...separatorProps } = props
            void node
            return <hr {...separatorProps} className="my-3 block border-t border-border-soft" />
          },
          li(props) {
            const { node, ...itemProps } = props
            void node
            return <li {...itemProps} className="my-0.5 pl-1 marker:text-fg-faint [&>p]:my-0" />
          },
          ol(props) {
            const { node, ...listProps } = props
            void node
            return <ol {...listProps} className="my-2 list-decimal pl-5" />
          },
          p(props) {
            const { node, ...paragraphProps } = props
            void node
            return <p {...paragraphProps} className="my-2 block first:mt-0 last:mb-0" />
          },
          pre(props) {
            const { node, children, ...preProps } = props
            if (isEnhancedCodeBlock(node)) return <>{children}</>
            return <pre {...preProps}>{children}</pre>
          },
          table(props) {
            const { className: tableClassName, node, ...tableProps } = props
            void node
            return (
              <div className="my-2 max-w-full overflow-x-auto">
                <table
                  {...tableProps}
                  className={cn('w-full border-collapse text-[length:var(--density-type-caption)]', tableClassName)}
                />
              </div>
            )
          },
          tbody(props) {
            const { node, ...tbodyProps } = props
            void node
            return <tbody {...tbodyProps} />
          },
          td(props) {
            const { className: cellClassName, node, ...cellProps } = props
            void node
            return (
              <td {...cellProps} className={cn('border border-border-soft px-2 py-1 text-fg-dim', cellClassName)} />
            )
          },
          th(props) {
            const { className: cellClassName, node, ...cellProps } = props
            void node
            return (
              <th
                {...cellProps}
                className={cn(
                  'border border-border-soft bg-panel-strong px-2 py-1 text-left font-semibold text-foreground',
                  cellClassName
                )}
              />
            )
          },
          thead(props) {
            const { node, ...theadProps } = props
            void node
            return <thead {...theadProps} />
          },
          tr(props) {
            const { node, ...rowProps } = props
            void node
            return <tr {...rowProps} />
          },
          ul(props) {
            const { node, ...listProps } = props
            void node
            return <ul {...listProps} className="my-2 list-disc pl-5" />
          }
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  )
}

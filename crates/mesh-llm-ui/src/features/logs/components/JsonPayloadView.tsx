import { Copy } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { SegmentedControl, type SegmentedControlOption } from '@/components/ui/SegmentedControl'
import { copyStateLabel } from '@/lib/copyStateLabel'
import { useClipboardCopy } from '@/lib/useClipboardCopy'
import { cn } from '@/lib/utils'

export type JsonTokenType = 'key' | 'string' | 'number' | 'boolean' | 'null' | 'punctuation'

export type JsonPayloadViewProps = {
  readonly text: string
  readonly prettyText: string
  readonly ariaLabel?: string
  readonly className?: string
}

type JsonFormat = 'pretty' | 'raw'

const JSON_FORMAT_OPTIONS = [
  { value: 'pretty', label: 'Pretty', selectedTone: 'accent' },
  { value: 'raw', label: 'Raw', selectedTone: 'accent' }
] as const satisfies readonly SegmentedControlOption[]

const JSON_TOKEN_PATTERN = /"(?:\\.|[^"\\])*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|[{}[\],:]/g

const tokenClass: Record<JsonTokenType, string> = {
  key: 'font-semibold text-accent',
  string: 'text-foreground',
  number: 'text-warn',
  boolean: 'text-accent',
  null: 'italic text-fg-faint',
  punctuation: 'text-fg-faint'
}

function jsonTokenType(source: string, token: string, index: number): JsonTokenType {
  if (token.startsWith('"')) {
    return /^\s*:/.test(source.slice(index + token.length)) ? 'key' : 'string'
  }
  if (token === 'true' || token === 'false') return 'boolean'
  if (token === 'null') return 'null'
  if (/^-?\d/.test(token)) return 'number'
  return 'punctuation'
}

function jsonTokenNodes(source: string): readonly ReactNode[] {
  const nodes: ReactNode[] = []
  let cursor = 0

  for (const match of source.matchAll(JSON_TOKEN_PATTERN)) {
    const index = match.index
    const token = match[0]
    if (index > cursor) nodes.push(source.slice(cursor, index))
    const type = jsonTokenType(source, token, index)
    nodes.push(
      <span className={tokenClass[type]} data-json-token={type} key={`${index}-${type}`}>
        {token}
      </span>
    )
    cursor = index + token.length
  }

  if (cursor < source.length) nodes.push(source.slice(cursor))
  return nodes
}

function isJsonFormat(value: string): value is JsonFormat {
  return value === 'pretty' || value === 'raw'
}

export function JsonPayloadView({ text, prettyText, ariaLabel = 'JSON payload', className }: JsonPayloadViewProps) {
  const [format, setFormat] = useState<JsonFormat>('pretty')
  const { copyState, copyText } = useClipboardCopy()
  const currentText = format === 'pretty' ? prettyText : text
  const lines = useMemo(() => currentText.split('\n'), [currentText])
  const formatLabel = format === 'pretty' ? 'Pretty' : 'Raw'
  const copyAnnouncement =
    copyState === 'copied' ? ' JSON payload copied.' : copyState === 'failed' ? ' JSON payload copy failed.' : ''

  return (
    <section aria-label={ariaLabel}>
      <div className="flex min-w-max items-center justify-between gap-3 border-b border-border-soft bg-panel-strong px-3 py-2">
        <SegmentedControl
          ariaLabel="JSON format"
          onValueChange={(value) => {
            if (isJsonFormat(value)) setFormat(value)
          }}
          options={JSON_FORMAT_OPTIONS}
          value={format}
          variant="pill"
        />
        <Button
          aria-label="Copy JSON payload"
          className="ui-control h-7 gap-1.5 px-2 text-[length:var(--density-type-caption)]"
          onClick={() => void copyText(currentText)}
          size="sm"
          type="button"
          variant="outline"
        >
          <Copy aria-hidden="true" className="size-3" />
          {copyStateLabel(copyState)}
        </Button>
      </div>
      <p aria-atomic="true" aria-live="polite" className="sr-only" role="status">
        {formatLabel} JSON representation selected.{copyAnnouncement}
      </p>
      <pre
        className={cn('min-w-max py-3 font-mono text-[length:var(--density-type-caption)] leading-relaxed', className)}
      >
        <code>
          {lines.map((line, index) => {
            const lineNumber = index + 1
            return (
              <span className="grid grid-cols-[auto_minmax(0,1fr)]" key={`${lineNumber}-${line}`}>
                <span
                  aria-hidden="true"
                  className="w-10 select-none border-r border-border-soft pr-3 text-right text-fg-faint"
                  data-line-number
                >
                  {lineNumber}
                </span>
                <span className="whitespace-pre pl-3 pr-3" data-json-line>
                  {jsonTokenNodes(line)}
                </span>
              </span>
            )
          })}
        </code>
      </pre>
    </section>
  )
}

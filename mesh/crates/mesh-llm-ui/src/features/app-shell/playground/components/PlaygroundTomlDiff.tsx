import { useCallback } from 'react'
import { Copy } from 'lucide-react'
import ReactDiffViewer from 'react-diff-viewer-continued'
import { env } from '@/lib/env'
import { useClipboardCopy } from '@/lib/useClipboardCopy'
import { copyStateLabel } from '@/lib/copyStateLabel'
import { HighlightedTomlLines } from '@/features/configuration/components/toml-highlight'

const SAMPLE_OLD_TOML = `[defaults]
ctx_size = 4096
max_tokens = 512
temperature = 0.7
system_prompt = "You are a helpful assistant."

[defaults.server]
port = 9337
host = "0.0.0.0"

[models."llama-3.2-1b"]
ctx_size = 2048

[models."gemma-2-2b"]
ctx_size = 4096
`

const SAMPLE_NEW_TOML = `[defaults]
ctx_size = 8192
max_tokens = 1024
temperature = 0.8
top_p = 0.95
system_prompt = "You are a helpful assistant."

[defaults.server]
port = 9337
host = "0.0.0.0"

[models."llama-3.2-1b"]
ctx_size = 2048

[models."gemma-2-2b"]
ctx_size = 8192
`

const diffViewerStyles = {
  diffContainer: {
    background: 'var(--color-background)',
    border: '1px solid var(--color-border-soft)',
    borderRadius: 'var(--radius)',
    minWidth: '100%',
    width: '100%',
    fontSize: 'var(--density-type-caption-lg)',
    color: 'var(--color-foreground)'
  },
  titleBlock: {
    background: 'var(--color-panel-strong)',
    color: 'var(--color-fg-dim)',
    borderBottom: '1px solid var(--color-border-soft)',
    borderLeft: 0,
    fontSize: 'var(--density-type-control-lg)',
    padding: '0.45em 0.6em',
    '&& pre': { color: 'inherit' },
    '&:last-child:not(:only-child)': { borderLeft: 0 }
  },
  contentText: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--density-type-caption-lg)',
    lineHeight: 'var(--toml-source-line-height)',
    color: 'inherit'
  },
  gutter: {
    background: 'var(--color-panel)',
    borderRight: '1px solid var(--color-border-soft)',
    color: 'var(--color-fg-dim)',
    '&:hover': { background: 'var(--color-panel-strong)' }
  },
  lineContent: {
    paddingTop: '0.04em',
    paddingBottom: '0.04em'
  },
  diffRemoved: {
    backgroundColor: 'color-mix(in oklch, var(--color-bad) 12%, transparent)',
    color: 'var(--color-fg-dim)',
    '&:hover': { backgroundColor: 'color-mix(in oklch, var(--color-bad) 22%, transparent)' }
  },
  diffAdded: {
    backgroundColor: 'color-mix(in oklch, var(--color-good) 14%, transparent)',
    color: 'var(--color-fg-dim)',
    pre: { color: 'inherit' },
    '&:hover': { backgroundColor: 'color-mix(in oklch, var(--color-good) 22%, transparent)' }
  },
  diffChanged: {
    backgroundColor: 'color-mix(in oklch, var(--color-accent) 10%, transparent)',
    color: 'var(--color-fg-dim)'
  },
  highlightedLine: {
    backgroundColor: 'color-mix(in oklch, var(--color-accent) 18%, transparent)'
  },
  highlightedGutter: {
    backgroundColor: 'color-mix(in oklch, var(--color-accent) 18%, transparent)',
    color: 'var(--color-foreground)'
  },
  lineNumber: { color: 'var(--color-fg-faint)' },
  emptyGutter: { background: 'var(--color-panel)' },
  emptyLine: { background: 'var(--color-background)' },
  marker: { color: 'var(--color-fg-faint)' },
  codeFold: { background: 'var(--color-panel-strong)', color: 'var(--color-fg-dim)' },
  codeFoldGutter: { background: 'var(--color-panel)' },
  codeFoldExpandButton: { background: 'transparent' },
  codeFoldContent: { color: 'inherit' }
}

export default function PlaygroundTomlDiff() {
  const { copyState, copyText } = useClipboardCopy()

  const handleCopy = useCallback(() => {
    copyText(SAMPLE_NEW_TOML)
  }, [copyText])

  if (!env.isDevelopment) {
    return null
  }

  const lineCount = SAMPLE_NEW_TOML.split('\n').length
  const copyLabel = copyStateLabel(copyState)

  return (
    <section className="toml-panel panel-shell flex h-full min-h-0 flex-col overflow-hidden rounded-lg border border-border bg-panel">
      <header className="toml-panel-header panel-divider flex items-center justify-between border-b border-border-soft">
        <div className="toml-panel-heading flex min-w-0 items-baseline gap-3">
          <h2 className="type-panel-title shrink-0">TOML Diff</h2>
        </div>
        <div className="toml-panel-actions flex shrink-0 items-center gap-2">
          <span className="type-caption tabular-nums text-fg-faint">{lineCount} lines</span>
          <button
            className="toml-copy-button ui-control inline-flex shrink-0 items-center justify-center rounded border"
            onClick={handleCopy}
            type="button"
            aria-label={copyLabel}
            title={copyLabel}
          >
            <Copy className="toml-copy-icon" strokeWidth={1.75} />
          </button>
        </div>
      </header>
      <div className="toml-diff-body relative flex-1 overflow-hidden bg-background">
        <div className="toml-diff-viewer toml-source-content">
          <ReactDiffViewer
            oldValue={SAMPLE_OLD_TOML}
            newValue={SAMPLE_NEW_TOML}
            splitView
            disableWordDiff
            hideSummary
            showDiffOnly={false}
            leftTitle="Old config"
            rightTitle="New config"
            styles={diffViewerStyles}
            renderContent={(line: string) => <HighlightedTomlLines toml={line || ' '} />}
          />
        </div>
      </div>
    </section>
  )
}

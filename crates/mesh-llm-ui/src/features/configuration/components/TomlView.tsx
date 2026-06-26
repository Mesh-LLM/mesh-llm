import { useEffect, useState, type CSSProperties, type ReactNode } from 'react'
import { cn } from '@/lib/cn'
import { Copy } from 'lucide-react'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import { HighlightedTomlLines } from '@/features/configuration/components/toml-highlight'
import { validateRuntimeConfigToml, type RuntimeControlDiagnostic } from '@/features/configuration/api/config-adapter'
import type {
  ConfigAssign,
  ConfigModel,
  ConfigNode,
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsValues,
  ConfigurationModelPlacementPaths,
  TomlValidationWarning
} from '@/features/app-tabs/types'
import { copyStateLabel } from '@/lib/copyStateLabel'
import { useClipboardCopy } from '@/lib/useClipboardCopy'
import ReactDiffViewer from 'react-diff-viewer-continued'

type TomlViewProps = {
  nodes: ConfigNode[]
  assigns: ConfigAssign[]
  models?: ConfigModel[]
  defaults?: ConfigurationDefaultsHarnessData
  defaultsValues?: ConfigurationDefaultsValues
  modelPlacementPaths?: ConfigurationModelPlacementPaths
  modelConfigEntries?: readonly Record<string, unknown>[]
  reviewMode?: boolean
  previousToml?: string
  validationEnabled?: boolean
  configPath?: string
  validationWarnings?: TomlValidationWarning[]
  launchSummaryConfig?: { httpBind?: string; mmap?: string }
}

type TomlPanelProps = {
  toml: string
  lineCount: number
  highlighted: ReactNode
  scrollOffset: { left: number; top: number }
  setScrollOffset: (offset: { left: number; top: number }) => void
  sourceStyle?: TomlSourceStyle
  copyLabel: string
  onCopy: () => void
  reviewMode: boolean
  configPath?: string
}

type TomlSourceStyle = CSSProperties & { '--toml-line-count'?: number }

function TomlEditorPanel({
  toml,
  lineCount,
  highlighted,
  scrollOffset,
  setScrollOffset,
  sourceStyle,
  copyLabel,
  onCopy,
  reviewMode,
  configPath
}: TomlPanelProps) {
  return (
    <section
      className={cn('toml-panel panel-shell overflow-hidden rounded-lg border border-border bg-panel', {
        'flex h-full min-h-0 flex-col': reviewMode
      })}
    >
      <header className="toml-panel-header panel-divider flex items-center justify-between border-b border-border-soft">
        <div className="toml-panel-heading flex min-w-0 items-baseline">
          <h2 className="type-panel-title shrink-0">{reviewMode ? 'Generated TOML' : 'Configuration TOML'}</h2>
          {reviewMode && configPath ? (
            <span className="toml-config-path mono truncate font-normal text-fg-faint">{configPath}</span>
          ) : null}
        </div>
        <div className="toml-panel-actions flex shrink-0 items-center">
          <span className="type-caption tabular-nums text-fg-faint">{lineCount} lines</span>
          {reviewMode ? (
            <span className="toml-review-badge rounded-full border border-border-soft text-fg-faint">
              edits this node only
            </span>
          ) : null}
          <button
            className="toml-copy-button ui-control inline-flex shrink-0 items-center justify-center rounded border"
            onClick={onCopy}
            type="button"
            aria-label={copyLabel}
            title={copyLabel}
          >
            <Copy className="toml-copy-icon" strokeWidth={1.75} />
          </button>
        </div>
      </header>
      <div
        className={cn('toml-source relative overflow-hidden bg-background font-mono', {
          'min-h-0 flex-1': reviewMode,
          'toml-source-standalone': !reviewMode
        })}
        style={sourceStyle}
      >
        <pre
          aria-hidden="true"
          className="toml-source-content pointer-events-none absolute inset-0 m-0 whitespace-pre text-fg-dim"
          style={{ transform: `translate(${-scrollOffset.left}px, ${-scrollOffset.top}px)` }}
        >
          {highlighted}
        </pre>
        <textarea
          aria-label="Configuration TOML source"
          className="toml-source-content toml-source-input absolute inset-0 block h-full w-full resize-none overflow-auto bg-transparent font-mono text-transparent caret-transparent"
          onScroll={(event) =>
            setScrollOffset({ left: event.currentTarget.scrollLeft, top: event.currentTarget.scrollTop })
          }
          readOnly
          value={toml}
          wrap="off"
        />
      </div>
    </section>
  )
}

function TomlDiffPanel({
  oldToml,
  newToml,
  lineCount,
  copyLabel,
  onCopy,
  reviewMode,
  configPath
}: {
  oldToml: string
  newToml: string
  lineCount: number
  copyLabel: string
  onCopy: () => void
  reviewMode: boolean
  configPath?: string
}) {
  const diffViewerStyles = {
    diffContainer: {
      background: 'var(--color-background)',
      border: `1px solid var(--color-border-soft)`,
      borderRadius: 'var(--radius)',
      minWidth: '100%',
      width: '100%',
      fontSize: 'var(--density-type-caption-lg)',
      color: 'var(--color-foreground)'
    },
    summary: { background: 'var(--color-panel-strong)' },
    titleBlock: {
      background: 'var(--color-panel-strong)',
      color: 'var(--color-fg-dim)',
      borderBottom: `1px solid var(--color-border-soft)`,
      borderLeft: 0,
      fontSize: 'var(--density-type-control-lg)',
      padding: '0.45em 0.6em',
      '&& pre': {
        color: 'inherit'
      },
      '&:last-child:not(:only-child)': {
        borderLeft: 0
      }
    },
    contentText: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--density-type-caption-lg)',
      lineHeight: 'var(--toml-source-line-height)',
      color: 'inherit'
    },
    gutter: {
      background: 'var(--color-panel)',
      borderRight: `1px solid var(--color-border-soft)`,
      color: 'var(--color-fg-dim)',
      '&:hover': {
        background: 'var(--color-panel-strong)'
      }
    },
    lineContent: {
      paddingTop: '0.04em',
      paddingBottom: '0.04em'
    },
    diffRemoved: {
      backgroundColor: 'color-mix(in oklch, var(--color-bad) 12%, transparent)',
      color: 'var(--color-fg-dim)',
      '&:hover': {
        backgroundColor: 'color-mix(in oklch, var(--color-bad) 22%, transparent)'
      }
    },
    diffAdded: {
      backgroundColor: 'color-mix(in oklch, var(--color-good) 14%, transparent)',
      color: 'var(--color-fg-dim)',
      pre: { color: 'inherit' },
      '&:hover': {
        backgroundColor: 'color-mix(in oklch, var(--color-good) 22%, transparent)'
      }
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
    emptyGutter: {
      background: 'var(--color-panel)'
    },
    emptyLine: {
      background: 'var(--color-background)'
    },
    marker: {
      color: 'var(--color-fg-faint)'
    },
    codeFold: {
      background: 'var(--color-panel-strong)',
      color: 'var(--color-fg-dim)'
    },
    codeFoldGutter: {
      background: 'var(--color-panel)'
    },
    codeFoldExpandButton: {
      background: 'transparent'
    },
    codeFoldContent: {
      color: 'inherit'
    }
  }

  return (
    <section className="toml-panel panel-shell flex h-full min-h-0 flex-col overflow-hidden rounded-lg border border-border bg-panel">
      <header className="toml-panel-header panel-divider flex items-center justify-between border-b border-border-soft">
        <div className="toml-panel-heading flex min-w-0 items-baseline">
          <h2 className="type-panel-title shrink-0">{reviewMode ? 'Generated TOML' : 'Configuration TOML'}</h2>
          {reviewMode && configPath ? (
            <span className="toml-config-path mono truncate font-normal text-fg-faint">{configPath}</span>
          ) : null}
        </div>
        <div className="toml-panel-actions flex shrink-0 items-center">
          <span className="type-caption tabular-nums text-fg-faint">{lineCount} lines</span>
          <span className="toml-review-badge rounded-full border border-border-soft text-fg-faint">
            edits this node only
          </span>
          <button
            className="toml-copy-button ui-control inline-flex shrink-0 items-center justify-center rounded border"
            onClick={onCopy}
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
            oldValue={oldToml}
            newValue={newToml}
            splitView
            disableWordDiff
            hideSummary
            showDiffOnly={false}
            leftTitle="Saved TOML"
            rightTitle="Current TOML"
            styles={diffViewerStyles}
            renderContent={(line: string) => <HighlightedTomlLines toml={line || ' '}></HighlightedTomlLines>}
          />
        </div>
        <textarea
          aria-label="Configuration TOML source"
          className="toml-source-content toml-source-input sr-only"
          readOnly
          value={newToml}
          wrap="off"
        />
      </div>
    </section>
  )
}

function ReviewPanel({ title, children, className }: { title: string; children: ReactNode; className?: string }) {
  return (
    <section
      className={cn('toml-panel panel-shell overflow-hidden rounded-lg border border-border bg-panel', className)}
    >
      <header className="toml-panel-header border-b border-border-soft">
        <h3 className="type-panel-title">{title}</h3>
      </header>
      {children}
    </section>
  )
}

function warningDotClass(kind: TomlValidationWarning['kind']): string {
  if (kind === 'ok') return 'toml-status-dot toml-status-dot-ok shrink-0 rounded-full'
  if (kind === 'warn') return 'toml-status-dot toml-status-dot-warn shrink-0 rounded-full'
  return 'toml-status-dot shrink-0 rounded-full bg-fg-faint'
}

type ValidationMessageParts = {
  path?: string
  message: string
}

function parseValidationMessage(text: string): ValidationMessageParts {
  const separatorIndex = text.indexOf(': ')
  if (separatorIndex <= 0) return { message: text }

  const path = text.slice(0, separatorIndex).trim()
  const message = text.slice(separatorIndex + 2).trim()
  if (!path || !message) return { message: text }

  return { path, message }
}

function WarningItem({ kind, text }: TomlValidationWarning) {
  const { path, message } = parseValidationMessage(text)

  return (
    <div className="toml-warning-item flex items-start" data-kind={kind}>
      <span aria-hidden="true" className={warningDotClass(kind)} />
      <div className="toml-warning-content min-w-0">
        {path ? <span className="toml-warning-path font-mono">{path}</span> : null}
        <span className="toml-warning-message">{message}</span>
      </div>
    </div>
  )
}

const DEFAULT_VALIDATION_WARNINGS: TomlValidationWarning[] = [
  { kind: 'ok', text: 'Local model entries use compact model aliases accepted by mesh-llm config.' },
  {
    kind: 'warn',
    text: 'carrack · GPU 0 · GLM-4.7-Flash will exceed 80% VRAM at 16K context. Consider 8K or moving to GPU 1.'
  },
  { kind: 'ok', text: 'Remote nodes remain read-only context and are excluded from the saved TOML preview.' },
  { kind: 'info', text: 'Request defaults merge at request time, explicit request payload fields still win.' }
]

const VALIDATING_WARNING: TomlValidationWarning = {
  kind: 'info',
  text: 'Validating generated TOML against mesh-llm config rules.'
}

const VALIDATING_WARNINGS: TomlValidationWarning[] = [VALIDATING_WARNING]

type LiveValidationResult = {
  readonly key: string
  readonly warnings: TomlValidationWarning[]
}

function diagnosticWarningKind(diagnostic: RuntimeControlDiagnostic): TomlValidationWarning['kind'] {
  return diagnostic.severity === 'error' || diagnostic.severity === 'warning' || diagnostic.severity === 'warn'
    ? 'warn'
    : 'info'
}

function diagnosticWarningText(diagnostic: RuntimeControlDiagnostic): string {
  const location = diagnostic.canonical_path ?? diagnostic.path
  return location ? `${location}: ${diagnostic.message}` : diagnostic.message
}

function validationWarningsFromResponse(response: Awaited<ReturnType<typeof validateRuntimeConfigToml>>) {
  const warnings = response.diagnostics.map((diagnostic) => ({
    kind: diagnosticWarningKind(diagnostic),
    text: diagnosticWarningText(diagnostic)
  }))
  if (response.error) warnings.unshift({ kind: 'warn', text: response.error })
  if (warnings.length > 0) return warnings
  return [
    {
      kind: response.ok ? 'ok' : 'warn',
      text: response.ok
        ? 'Generated TOML validates against mesh-llm config rules.'
        : 'Generated TOML did not pass mesh-llm config validation.'
    }
  ] satisfies TomlValidationWarning[]
}

function ValidationPanel({ warnings, className }: { warnings?: TomlValidationWarning[]; className?: string }) {
  const resolvedWarnings = warnings ?? DEFAULT_VALIDATION_WARNINGS

  return (
    <ReviewPanel title="Validation" className={className}>
      <div className="toml-warning-list">
        {resolvedWarnings.map((warning) => (
          <WarningItem key={`${warning.kind}-${warning.text}`} kind={warning.kind} text={warning.text} />
        ))}
      </div>
    </ReviewPanel>
  )
}

function LaunchSummaryPanel({
  nodes,
  assigns,
  defaultsValues,
  launchSummaryConfig,
  className
}: {
  nodes: ConfigNode[]
  assigns: ConfigAssign[]
  defaultsValues?: ConfigurationDefaultsValues
  launchSummaryConfig?: { httpBind?: string; mmap?: string }
  className?: string
}) {
  const localNode = nodes[0]
  const gpuCount = localNode?.gpus.length ?? 0
  const flashAttention = defaultsValues?.['defaults.model_fit.flash_attention'] ?? 'auto'
  const kvCache = defaultsValues?.['defaults.model_fit.kv_cache_policy'] ?? 'auto'
  const httpBind = launchSummaryConfig?.httpBind ?? '0.0.0.0:9337'
  const mmap = launchSummaryConfig?.mmap ?? 'off'
  const rows = [
    ['node:', localNode?.hostname ?? 'local'],
    ['placements:', `${assigns.length} models on ${gpuCount} GPUs`],
    ['http:', httpBind],
    ['flash attn:', flashAttention],
    ['kv cache:', `${kvCache} (q8_0/q4_0 above 5GB)`],
    ['mmap:', mmap]
  ]

  return (
    <ReviewPanel title="Effective launch summary" className={className}>
      <div className="toml-summary-list type-caption text-fg-dim">
        {rows.map(([label, value]) => (
          <div className="toml-summary-row" key={label}>
            <span className="text-fg-faint">{label}</span>{' '}
            <span className={cn('font-mono', label === 'flash attn:' ? 'toml-summary-value-good' : 'text-foreground')}>
              {value}
            </span>
          </div>
        ))}
      </div>
    </ReviewPanel>
  )
}

export function TomlView({
  nodes,
  assigns,
  models,
  defaults,
  defaultsValues,
  modelPlacementPaths,
  modelConfigEntries,
  reviewMode = false,
  previousToml,
  validationEnabled = false,
  configPath,
  validationWarnings,
  launchSummaryConfig
}: TomlViewProps) {
  const toml = buildTOML(nodes, assigns, models, { defaults, defaultsValues, modelPlacementPaths, modelConfigEntries })
  const { copyState, copyText } = useClipboardCopy()
  const [scrollOffset, setScrollOffset] = useState({ left: 0, top: 0 })
  const [liveValidationResult, setLiveValidationResult] = useState<LiveValidationResult | undefined>()
  const copyLabel = copyStateLabel(copyState, 'TOML')
  const lines = toml.split('\n')
  const lineCount = lines.length
  const sourceStyle: TomlSourceStyle | undefined = reviewMode ? undefined : { '--toml-line-count': lineCount }
  const highlighted = <HighlightedTomlLines toml={toml} />
  const hasTomlDiff = previousToml !== undefined && previousToml !== toml
  const validationRequestKey = reviewMode && validationEnabled ? `${configPath ?? ''}\n${toml}` : undefined
  const resolvedValidationWarnings =
    validationRequestKey === undefined
      ? validationWarnings
      : liveValidationResult?.key === validationRequestKey
        ? liveValidationResult.warnings
        : VALIDATING_WARNINGS

  useEffect(() => {
    if (validationRequestKey === undefined) return

    let cancelled = false
    void validateRuntimeConfigToml(toml, configPath)
      .then((response) => {
        if (!cancelled) {
          setLiveValidationResult({ key: validationRequestKey, warnings: validationWarningsFromResponse(response) })
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return
        const message = error instanceof Error && error.message.trim() ? error.message : 'Runtime validation failed.'
        setLiveValidationResult({ key: validationRequestKey, warnings: [{ kind: 'warn', text: message }] })
      })

    return () => {
      cancelled = true
    }
  }, [configPath, toml, validationRequestKey])

  const editor = (
    <TomlEditorPanel
      toml={toml}
      lineCount={lineCount}
      highlighted={highlighted}
      scrollOffset={scrollOffset}
      setScrollOffset={setScrollOffset}
      sourceStyle={sourceStyle}
      copyLabel={copyLabel}
      onCopy={() => {
        void copyText(toml)
      }}
      reviewMode={reviewMode}
      configPath={configPath}
    />
  )

  if (reviewMode && hasTomlDiff) {
    return (
      <div className="toml-review-layout grid xl:items-stretch">
        <TomlDiffPanel
          oldToml={previousToml}
          newToml={toml}
          lineCount={lineCount}
          copyLabel={copyLabel}
          onCopy={() => {
            void copyText(toml)
          }}
          reviewMode={reviewMode}
          configPath={configPath}
        />
        <aside className="toml-review-aside flex flex-col" aria-label="TOML review actions">
          <ValidationPanel warnings={resolvedValidationWarnings} className="flex-1 min-h-0" />
          <LaunchSummaryPanel
            nodes={nodes}
            assigns={assigns}
            defaultsValues={defaultsValues}
            launchSummaryConfig={launchSummaryConfig}
            className="flex-1 min-h-0"
          />
        </aside>
      </div>
    )
  }

  if (!reviewMode) return editor

  return (
    <div className="toml-review-layout grid xl:items-stretch">
      {editor}
      <aside className="toml-review-aside flex flex-col" aria-label="TOML review actions">
        <ValidationPanel warnings={resolvedValidationWarnings} className="flex-1 min-h-0" />
        <LaunchSummaryPanel
          nodes={nodes}
          assigns={assigns}
          defaultsValues={defaultsValues}
          launchSummaryConfig={launchSummaryConfig}
          className="flex-1 min-h-0"
        />
      </aside>
    </div>
  )
}

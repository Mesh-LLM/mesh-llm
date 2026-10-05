import type { ReactNode } from 'react'

const TOML_KEY_PATTERN = String.raw`(?:"(?:\\.|[^"\\])*"|[A-Za-z0-9_.-]+)`
const tomlSectionClassName =
  'font-semibold text-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-foreground))]'
const tomlStringClassName = 'text-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-foreground))]'
const tomlBooleanClassName = 'text-[color:color-mix(in_oklab,var(--color-good)_35%,var(--color-foreground))]'
const tomlNumberClassName = 'text-[color:color-mix(in_oklab,var(--color-warn)_35%,var(--color-foreground))]'

function valueClassName(value: string): string {
  const trimmed = value.trim()
  if (/^".*"$/.test(trimmed) || /^'.*'$/.test(trimmed)) return tomlStringClassName
  if (/^(true|false)$/i.test(trimmed)) return tomlBooleanClassName
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) return tomlNumberClassName
  return 'text-foreground'
}

function highlightTomlLine(line: string): ReactNode {
  if (/^\s*#/.test(line)) return <span className="text-fg-dim">{line}</span>
  if (/^\s*\[\[?.+\]?\]\s*$/.test(line)) {
    return (
      <span className={tomlSectionClassName} data-toml-token="section">
        {line}
      </span>
    )
  }

  const keyValue = line.match(new RegExp(`^(\\s*)(${TOML_KEY_PATTERN})(\\s*=\\s*)(.*)$`))
  if (keyValue) {
    const [, indent, key, operator, value] = keyValue
    return (
      <>
        {indent}
        <span className="text-foreground" data-toml-token="key">
          {key}
        </span>
        <span className="text-fg-dim">{operator}</span>
        <span className={valueClassName(value)} data-toml-token="value">
          {value}
        </span>
      </>
    )
  }
  return <span className="text-fg-dim">{line || ' '}</span>
}

export function HighlightedTomlLines({ toml }: { toml: string }) {
  const lineOccurrences = new Map<string, number>()

  return toml.split('\n').map((line) => {
    const occurrence = lineOccurrences.get(line) ?? 0
    lineOccurrences.set(line, occurrence + 1)

    return (
      <div className="relative text-transparent" key={`${line}-${occurrence}`}>
        {line || ' '}
        <span aria-hidden="true" className="pointer-events-none absolute left-0 top-0 whitespace-pre">
          {highlightTomlLine(line)}
        </span>
      </div>
    )
  })
}

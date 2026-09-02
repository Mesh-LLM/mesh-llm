import type { RuntimeControlApplyResponse, RuntimeControlDiagnostic } from './config-adapter-types'

/**
 * Format a severity level as a readable uppercase badge.
 */
function formatSeverity(severity: string): string {
  return `\`${severity.toUpperCase()}\``
}

/**
 * Format a single diagnostic as a pretty markdown block.
 *
 * Produces output like:
 *
 *   **`path.to.setting`** · `ERROR`
 *
 *   The validation message explaining what is wrong.
 *
 *   > **Help:** guidance on how to fix the issue
 */
function formatDiagnosticBlock(diagnostic: RuntimeControlDiagnostic): string {
  const lines: string[] = []

  // Header: path + severity
  const headerParts: string[] = []
  const diagnosticPath = diagnostic.path ?? diagnostic.canonical_path
  if (diagnosticPath) headerParts.push(`**\`${diagnosticPath}\`**`)
  headerParts.push(formatSeverity(diagnostic.severity))
  lines.push(headerParts.join(' · '))
  lines.push('')

  // Body: message
  lines.push(diagnostic.message)

  // Footer: help text
  if (diagnostic.help) {
    lines.push('')
    lines.push(`> **Help:** ${diagnostic.help}`)
  }

  return lines.join('\n')
}

/**
 * Format an array of runtime-control diagnostics as pretty-printed markdown.
 *
 * Each diagnostic is rendered as a block with path, severity, message, and
 * optional help text. Multiple diagnostics are separated by horizontal rules.
 *
 * Returns `undefined` when the array is empty.
 */
export function formatConfigDiagnostics(diagnostics: readonly RuntimeControlDiagnostic[]): string | undefined {
  if (diagnostics.length === 0) return undefined
  return diagnostics.map(formatDiagnosticBlock).join('\n\n---\n\n')
}

export function runtimeControlApplyErrorMessage(response: RuntimeControlApplyResponse | null | undefined) {
  if (!response) return undefined
  const errorText = runtimeControlErrorMessage(response.error)
  if (errorText) return errorText

  const errorDiagnostics = response.diagnostics?.filter((d) => d.severity === 'error')
  if (errorDiagnostics && errorDiagnostics.length > 0) {
    return formatConfigDiagnostics(errorDiagnostics) ?? errorDiagnostics[0].message
  }

  return response.diagnostics?.find((d) => d.severity === 'warning')?.message ?? undefined
}

function runtimeControlErrorMessage(error: unknown): string | undefined {
  if (typeof error === 'string') return error.trim() || undefined
  if (!error || typeof error !== 'object') return undefined

  const details = error as Record<string, unknown>
  if (typeof details['message'] === 'string') return details['message'].trim() || undefined
  if (typeof details['code'] === 'string') return details['code'].replace(/_/g, ' ')
  return undefined
}

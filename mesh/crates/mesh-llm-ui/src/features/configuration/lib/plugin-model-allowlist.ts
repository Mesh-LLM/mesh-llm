import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'
import { parsePluginCanonicalPath } from '@/features/configuration/lib/plugin-setting-path'

/** How the editor spells an explicit empty list, which shares no models. */
export const EMPTY_MODEL_ALLOWLIST_TEXT = '[]'

/** Whether this is a plugin entry's endpoint model allowlist. Its form value
 * must round-trip exactly: `models = []` shares no endpoint models while an
 * unset one shares them all, and a split or merged model ID changes which
 * models are shared. */
export function isPluginModelAllowlist(
  setting: Pick<ConfigurationDefaultsSetting, 'canonicalPath' | 'tomlSection' | 'tomlKey'>
): boolean {
  const parsed = parsePluginCanonicalPath(setting)
  return parsed?.path.length === 1 && parsed.path[0] === 'models'
}

/** A form value: a JSON string array, or the comma-separated form. */
export function parsePluginModelAllowlist(value: string): string[] {
  if (value.trimStart().startsWith('[')) {
    try {
      const parsed: unknown = JSON.parse(value)
      if (Array.isArray(parsed) && parsed.every((item): item is string => typeof item === 'string')) return parsed
    } catch {
      // Not JSON: a model ID that starts with `[`.
    }
  }
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

/** The comma-separated form when it parses back to the same list, else JSON.
 * An empty list is `[]`, since an empty form value means unset. */
export function serializePluginModelAllowlist(models: readonly unknown[]): string {
  const joined = models.join(',')
  const parsed = parsePluginModelAllowlist(joined)
  const roundTrips =
    models.length > 0 && parsed.length === models.length && parsed.every((item, index) => item === models[index])
  return roundTrips ? joined : JSON.stringify(models)
}

/** Editor text for a form value: one model ID per line, commas kept. An
 * explicit empty list shows as `[]`; an unset allowlist shows nothing. */
export function modelAllowlistEditorText(value: string): string {
  if (value.trim().length === 0) return ''
  const models = parsePluginModelAllowlist(value)
  return models.length === 0 ? EMPTY_MODEL_ALLOWLIST_TEXT : models.join('\n')
}

/** The form value for editor text. Each non-blank line is one model ID, kept
 * exactly, and a lone `[]` line is the empty list. No lines leaves the
 * allowlist unset. */
export function modelAllowlistFromEditorText(text: string): string {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0)
  if (lines.length === 0) return ''
  if (lines.length === 1 && lines[0].trim() === EMPTY_MODEL_ALLOWLIST_TEXT) return serializePluginModelAllowlist([])
  return serializePluginModelAllowlist(lines)
}

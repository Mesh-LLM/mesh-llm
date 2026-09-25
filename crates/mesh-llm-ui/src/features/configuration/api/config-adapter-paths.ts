import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'

export function sectionPathSegments(section: string | undefined): string[] {
  return section?.split('.').filter(Boolean) ?? []
}

export function lastPathSegment(canonicalPath: string): string {
  return canonicalPath.split('.').filter(Boolean).at(-1) ?? canonicalPath
}

function defaultsSectionForPath(canonicalPath: string): string | undefined {
  const segments = canonicalPath.split('.')
  if (segments[0] !== 'defaults') return undefined
  if (segments[1] === 'advanced' && segments[2] === 'server') return 'defaults.advanced.server'
  return segments.length >= 2 ? `defaults.${segments[1]}` : undefined
}

/**
 * The TOML table a canonical setting path lives under. Nested runtime
 * sub-tables (for example `runtime.activity` or `runtime.native_runtime`) must
 * keep their own table so a copied preview does not flatten them into
 * `[runtime]`, where the keys are silently ignored.
 */
export function configSectionForPath(canonicalPath: string): string | undefined {
  if (canonicalPath.startsWith('plugin.')) return undefined
  const segments = canonicalPath.split('.').filter(Boolean)
  if (segments.length <= 1) return undefined
  if (segments[0] === 'defaults') return defaultsSectionForPath(canonicalPath)
  return segments.slice(0, -1).join('.')
}

export function resolveConfigSettingPath(setting: ConfigurationDefaultsSetting): string[] {
  if (setting.canonicalPath?.startsWith('defaults.')) {
    return setting.canonicalPath.slice('defaults.'.length).split('.').filter(Boolean)
  }
  if (setting.canonicalPath && !setting.canonicalPath.startsWith('plugin.')) {
    return setting.canonicalPath.split('.').filter(Boolean)
  }

  const key = 'name' in setting.control ? setting.control.name : setting.id
  return [...sectionPathSegments(setting.tomlSection), key]
}

export function readPath(source: unknown, path: readonly string[]): unknown {
  let current = source
  for (const segment of path) {
    if (!current || typeof current !== 'object' || !(segment in current)) return undefined
    current = (current as Record<string, unknown>)[segment]
  }
  return current
}

export function writePath(target: Record<string, unknown>, path: readonly string[], value: unknown) {
  let current = target
  path.forEach((segment, index) => {
    const isLeaf = index === path.length - 1
    if (isLeaf) {
      current[segment] = value
      return
    }

    const next = current[segment]
    if (!next || typeof next !== 'object' || Array.isArray(next)) {
      current[segment] = {}
    }
    current = current[segment] as Record<string, unknown>
  })
}

export function deletePath(target: Record<string, unknown>, path: readonly string[]): boolean {
  const [segment, ...rest] = path
  if (!segment || !(segment in target)) return Object.keys(target).length === 0

  if (rest.length === 0) {
    delete target[segment]
    return Object.keys(target).length === 0
  }

  const next = target[segment]
  if (!next || typeof next !== 'object' || Array.isArray(next)) return Object.keys(target).length === 0

  if (deletePath(next as Record<string, unknown>, rest)) delete target[segment]
  return Object.keys(target).length === 0
}

export function modelEntryPathSegments(path: string): string[] {
  return path
    .replace(/^models.<model-ref>.?/, '')
    .split('.')
    .filter(Boolean)
}

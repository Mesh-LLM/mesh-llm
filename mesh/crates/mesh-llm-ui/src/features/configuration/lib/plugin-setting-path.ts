import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'

/** The plugin and the path inside its `[[plugin]]` entry that a plugin setting
 * writes, or `undefined` for a setting outside plugin entries. */
export function parsePluginCanonicalPath(
  setting: Pick<ConfigurationDefaultsSetting, 'canonicalPath' | 'tomlSection' | 'tomlKey'>
) {
  const canonicalPath = setting.canonicalPath
  const sectionSegments = setting.tomlSection?.split('.').filter(Boolean) ?? []
  if (canonicalPath?.startsWith('plugin.') && sectionSegments[0] === 'plugin' && setting.tomlKey) {
    const hasSettingsSection = sectionSegments.at(-1) === 'settings'
    const pluginName = sectionSegments.slice(1, hasSettingsSection ? -1 : undefined).join('.')
    if (!pluginName) return undefined

    return {
      pluginName,
      path: hasSettingsSection ? ['settings', setting.tomlKey] : [setting.tomlKey]
    }
  }

  const match = canonicalPath?.match(/^plugin\.([^.]+)\.(.+)$/)
  if (!match) return undefined

  return { pluginName: match[1], path: match[2].split('.').filter(Boolean) }
}

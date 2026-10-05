import type {
  ConfigurationDefaultsCategory,
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues,
  ConfigurationSettingsHarnessData
} from '@/features/app-tabs/types'
import { sortCategories, sortSettings } from './schema-setting-order'
import type { ConfigurationDefaultsSchemaPathEntry } from './config-adapter-types'

export function configurationDefaultsSchemaPathEntries(): ConfigurationDefaultsSchemaPathEntry[] {
  return []
}

function cloneControlValue(
  control: ConfigurationDefaultsSetting['control'],
  value: string
): ConfigurationDefaultsSetting['control'] {
  return { ...control, value }
}

export function overlayDefaultsValues(
  harnessDefaults: ConfigurationDefaultsHarnessData,
  defaultsValues: ConfigurationDefaultsValues
): ConfigurationDefaultsHarnessData {
  return {
    ...harnessDefaults,
    settings: harnessDefaults.settings.map((setting) => ({
      ...setting,
      baselineValue: setting.baselineValue ?? setting.control.value,
      control: cloneControlValue(setting.control, defaultsValues[setting.id] ?? setting.control.value)
    }))
  }
}

export function combineSettingsHarnessData(
  ...groups: readonly (ConfigurationSettingsHarnessData | undefined)[]
): ConfigurationSettingsHarnessData {
  const categoryById = new Map<string, ConfigurationDefaultsCategory>()
  const settingById = new Map<string, ConfigurationDefaultsSetting>()
  const preview = groups.flatMap((group) => group?.preview ?? [])

  for (const group of groups) {
    for (const category of group?.categories ?? []) categoryById.set(String(category.id), category)
    for (const setting of group?.settings ?? []) settingById.set(setting.id, setting)
  }

  return {
    categories: sortCategories(Array.from(categoryById.values())),
    settings: sortSettings(Array.from(settingById.values())),
    preview
  }
}

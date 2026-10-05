import type { ConfigurationDefaultsCategory, ConfigurationDefaultsSetting } from '@/features/app-tabs/types'

export const DEFAULT_CATEGORY_ORDER = 1000
export const DEFAULT_SETTING_ORDER = 1000

export function titleCaseIdentifier(value: string) {
  return value
    .replaceAll('_', ' ')
    .replaceAll('-', ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase())
}

export function sortCategories(categories: readonly ConfigurationDefaultsCategory[]) {
  return [...categories].sort(
    (left, right) =>
      (left.order ?? DEFAULT_CATEGORY_ORDER) - (right.order ?? DEFAULT_CATEGORY_ORDER) ||
      left.label.localeCompare(right.label)
  )
}

export function sortSettings(settings: readonly ConfigurationDefaultsSetting[]) {
  return [...settings].sort(
    (left, right) =>
      (left.categoryOrder ?? DEFAULT_CATEGORY_ORDER) - (right.categoryOrder ?? DEFAULT_CATEGORY_ORDER) ||
      (left.settingOrder ?? DEFAULT_SETTING_ORDER) - (right.settingOrder ?? DEFAULT_SETTING_ORDER) ||
      left.label.localeCompare(right.label) ||
      left.id.localeCompare(right.id)
  )
}

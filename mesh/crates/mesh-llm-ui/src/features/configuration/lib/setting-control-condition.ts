import type {
  ConfigurationControlCondition,
  ConfigurationControlConditionOperator,
  ConfigurationControlConditionValue,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues,
  ConfigurationSettingValueSchema
} from '@/features/app-tabs/types'

function findSettingByPath(
  path: string,
  allSettings: readonly ConfigurationDefaultsSetting[]
): ConfigurationDefaultsSetting | undefined {
  return allSettings.find((setting) => setting.id === path || setting.canonicalPath === path)
}

function pathSegmentKey(segment: unknown): string {
  if (typeof segment === 'string') return segment
  if (typeof segment === 'number') return String(segment)
  if (typeof segment === 'object' && segment !== null) {
    const record = segment as Record<string, unknown>
    if (typeof record.name === 'string') return record.name
  }
  return String(segment)
}

function pluginScopedPathPrefix(setting: Pick<ConfigurationDefaultsSetting, 'canonicalPath' | 'id'> | undefined) {
  const currentPath = setting?.canonicalPath ?? setting?.id
  if (!currentPath?.startsWith('plugin.')) return undefined

  const settingsMarkerIndex = currentPath.indexOf('.settings.')
  if (settingsMarkerIndex >= 0) return `${currentPath.slice(0, settingsMarkerIndex)}.settings`

  const lastSegmentIndex = currentPath.lastIndexOf('.')
  return lastSegmentIndex > 0 ? currentPath.slice(0, lastSegmentIndex) : currentPath
}

function resolveConditionPathKey(
  path: string,
  currentSetting: Pick<ConfigurationDefaultsSetting, 'canonicalPath' | 'id'> | undefined
) {
  if (path.includes('.')) return path

  const prefix = pluginScopedPathPrefix(currentSetting)
  return prefix ? `${prefix}.${path}` : path
}

function parseBooleanLike(value: string | undefined): boolean | undefined {
  if (value == null) return undefined
  switch (value.trim().toLowerCase()) {
    case 'true':
    case 'on':
    case '1':
    case 'yes':
      return true
    case 'false':
    case 'off':
    case '0':
    case 'no':
    case '':
      return false
    default:
      return undefined
  }
}

function parseNumberLike(value: string | undefined): number | undefined {
  if (value == null || value.trim().length === 0) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function hasPresentValue(value: string | undefined) {
  return value != null && value.trim().length > 0
}

function getSettingValue(setting: ConfigurationDefaultsSetting, values: ConfigurationDefaultsValues) {
  return values[setting.id] ?? setting.control.value
}

function valueForComparison(
  rawValue: string | undefined,
  expected: ConfigurationControlConditionValue,
  schema: ConfigurationSettingValueSchema | undefined
) {
  switch (expected.kind) {
    case 'bool':
      return parseBooleanLike(rawValue)
    case 'integer':
    case 'float':
      return parseNumberLike(rawValue)
    case 'string':
      if (schema?.kind === 'boolean') {
        const parsed = parseBooleanLike(rawValue)
        return parsed == null ? rawValue : String(parsed)
      }
      return rawValue
    default:
      return rawValue
  }
}

export function conditionPathKey(
  condition: ConfigurationControlCondition,
  currentSetting?: Pick<ConfigurationDefaultsSetting, 'canonicalPath' | 'id'>
) {
  const rawPath = condition.path.segments.map(pathSegmentKey).join('.')
  return resolveConditionPathKey(rawPath, currentSetting)
}

export function evaluateControlCondition(
  condition: ConfigurationControlCondition,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues,
  currentSetting?: Pick<ConfigurationDefaultsSetting, 'canonicalPath' | 'id'>
) {
  const resolvedPath = conditionPathKey(condition, currentSetting)
  const referencedSetting = findSettingByPath(resolvedPath, allSettings)
  const rawValue = referencedSetting ? getSettingValue(referencedSetting, values) : values[resolvedPath]
  const expectedValues = condition.values ?? []

  switch (condition.operator) {
    case 'equals':
    case 'in':
      return expectedValues.some(
        (expected) => valueForComparison(rawValue, expected, referencedSetting?.valueSchema) === expected.value
      )
    case 'not_equals':
    case 'not_in':
      return expectedValues.every(
        (expected) => valueForComparison(rawValue, expected, referencedSetting?.valueSchema) !== expected.value
      )
    case 'present':
      return hasPresentValue(rawValue)
    case 'absent':
      return !hasPresentValue(rawValue)
    case 'truthy': {
      const parsed = parseBooleanLike(rawValue)
      return parsed ?? hasPresentValue(rawValue)
    }
    case 'falsy': {
      const parsed = parseBooleanLike(rawValue)
      return parsed === false || !hasPresentValue(rawValue)
    }
    case 'range': {
      const numericValue = parseNumberLike(rawValue)
      const [minimum, maximum] = expectedValues
        .map((expected) => (expected.kind === 'integer' || expected.kind === 'float' ? expected.value : undefined))
        .filter((value): value is number => value != null)
      if (numericValue == null) return false
      if (minimum != null && numericValue < minimum) return false
      if (maximum != null && numericValue > maximum) return false
      return true
    }
    default:
      return false
  }
}

function formatConditionValues(values: readonly ConfigurationControlConditionValue[]) {
  return values.map((value) => String(value.value)).join(values.length > 2 ? ', ' : ' or ')
}

export function controlConditionReason(
  operator: ConfigurationControlConditionOperator,
  path: string,
  values: readonly ConfigurationControlConditionValue[]
) {
  switch (operator) {
    case 'equals':
    case 'in':
      return values.length > 0 ? `Requires ${path} = ${formatConditionValues(values)}` : `Requires ${path}`
    case 'present':
      return `Requires ${path} to be set`
    case 'absent':
      return `Requires ${path} to be empty`
    case 'truthy':
      return `Requires ${path} to be enabled`
    case 'falsy':
      return `Requires ${path} to be disabled`
    case 'range':
      return values.length > 0
        ? `Requires ${path} within ${formatConditionValues(values)}`
        : `Requires ${path} within range`
    default:
      return `Requires ${path} to satisfy its dependency`
  }
}

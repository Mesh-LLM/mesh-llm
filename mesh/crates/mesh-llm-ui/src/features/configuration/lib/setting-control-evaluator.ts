import type {
  ConfigurationControlAvailabilitySource,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues,
  ConfigurationDisabledWritePolicy,
  ConfigurationRuntimeControlOption,
  ConfigurationRuntimeControlStateEntry
} from '@/features/app-tabs/types'
import {
  conditionPathKey,
  controlConditionReason,
  evaluateControlCondition
} from '@/features/configuration/lib/setting-control-condition'

export type SettingDependencyStatus = {
  disabled: boolean
  reason?: string
}

type SettingWriteDisposition = 'write' | 'preserve' | 'omit'

const DEFAULT_WRITE_POLICY: ConfigurationDisabledWritePolicy = 'preserve_existing'
const DEPENDENCY_WRITE_POLICY: ConfigurationDisabledWritePolicy = 'omit_when_disabled'

export function getSettingValue(setting: ConfigurationDefaultsSetting, values: ConfigurationDefaultsValues) {
  return values[setting.id] ?? setting.control.value
}

export function getSettingBaselineValue(setting: ConfigurationDefaultsSetting) {
  return setting.baselineValue ?? setting.control.value
}

function currentSettingPath(setting: ConfigurationDefaultsSetting) {
  return setting.canonicalPath ?? setting.id
}

function optionsFromState(
  setting: ConfigurationDefaultsSetting
): readonly ConfigurationRuntimeControlOption[] | undefined {
  return setting.controlState?.options
}

function withOptions(
  state: Omit<ConfigurationRuntimeControlStateEntry, 'options'>,
  options: readonly ConfigurationRuntimeControlOption[] | undefined
): ConfigurationRuntimeControlStateEntry {
  return options ? { ...state, options } : state
}

function dependencyRequirementValue(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
) {
  const dependency = setting.dependsOn
  if (!dependency) return undefined

  const parentSetting = allSettings.find((item) => item.id === dependency.settingId)
  if (!parentSetting) return undefined

  const currentValue = getSettingValue(parentSetting, values)
  if (dependency.condition(currentValue)) return currentValue

  if (parentSetting.control.kind === 'choice') {
    const matchingOptions = parentSetting.control.options.filter((option) => dependency.condition(option.value))
    if (matchingOptions.length > 0) return matchingOptions.map((option) => option.value)
  }

  return undefined
}

function formatLegacyRequirementValue(value: string | readonly string[]) {
  return Array.isArray(value) ? value.join(' or ') : value
}

function enabledFallbackState(setting: ConfigurationDefaultsSetting): ConfigurationRuntimeControlStateEntry {
  const options = optionsFromState(setting)
  const baseState = setting.controlState
  return withOptions(
    {
      enabled: true,
      reason: baseState?.reason,
      note: baseState?.note,
      source: baseState?.source ?? setting.controlBehavior?.availability?.source ?? 'static',
      write_policy: baseState?.write_policy ?? setting.controlBehavior?.write_policy ?? DEFAULT_WRITE_POLICY
    },
    options
  )
}

function dependencyDisabledState(
  setting: ConfigurationDefaultsSetting,
  source: ConfigurationControlAvailabilitySource,
  reason: string,
  note?: string,
  writePolicy: ConfigurationDisabledWritePolicy = DEPENDENCY_WRITE_POLICY
) {
  return withOptions(
    {
      enabled: false,
      reason,
      note,
      source,
      write_policy: writePolicy
    },
    optionsFromState(setting)
  )
}

export function evaluateSettingControlState(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
): ConfigurationRuntimeControlStateEntry {
  const staticAvailability = setting.controlBehavior?.availability
  if (staticAvailability && !staticAvailability.enabled) {
    return dependencyDisabledState(
      setting,
      staticAvailability.source,
      staticAvailability.reason ?? 'This setting is not available.',
      staticAvailability.note,
      setting.controlBehavior?.write_policy ?? DEFAULT_WRITE_POLICY
    )
  }

  if (setting.controlState && !setting.controlState.enabled) {
    return withOptions(setting.controlState, optionsFromState(setting))
  }

  for (const condition of setting.controlBehavior?.enable_when ?? []) {
    if (evaluateControlCondition(condition, allSettings, values, setting)) continue
    return dependencyDisabledState(
      setting,
      'dependency',
      controlConditionReason(condition.operator, conditionPathKey(condition, setting), condition.values ?? []),
      undefined,
      setting.controlBehavior?.write_policy ?? DEPENDENCY_WRITE_POLICY
    )
  }

  for (const disableRule of setting.controlBehavior?.disable_when ?? []) {
    if (!evaluateControlCondition(disableRule.condition, allSettings, values, setting)) continue
    return dependencyDisabledState(
      setting,
      'dependency',
      disableRule.reason,
      disableRule.note,
      disableRule.write_policy
    )
  }

  for (const conflictRule of setting.controlBehavior?.conflicts ?? []) {
    if (!evaluateControlCondition(conflictRule.condition, allSettings, values, setting)) continue
    const preferredPath = conflictRule.preferred_path
      ? conditionPathKey(
          {
            path: conflictRule.preferred_path,
            operator: 'present'
          },
          setting
        )
      : undefined
    if (preferredPath != null && preferredPath === currentSettingPath(setting)) continue
    return dependencyDisabledState(
      setting,
      'conflict',
      conflictRule.reason,
      preferredPath ? `Prefer ${preferredPath}` : undefined,
      setting.controlBehavior?.write_policy ?? DEFAULT_WRITE_POLICY
    )
  }

  if (setting.dependsOn) {
    const parentSetting = allSettings.find((item) => item.id === setting.dependsOn?.settingId)
    if (!parentSetting) {
      return dependencyDisabledState(setting, 'dependency', `Requires ${setting.dependsOn.settingId}`)
    }

    const parentValue = getSettingValue(parentSetting, values)
    if (!setting.dependsOn.condition(parentValue)) {
      const requiredValue = dependencyRequirementValue(setting, allSettings, values)
      if (requiredValue !== undefined) {
        return dependencyDisabledState(
          setting,
          'dependency',
          `Requires ${parentSetting.id} = ${formatLegacyRequirementValue(requiredValue)}`
        )
      }
      return dependencyDisabledState(setting, 'dependency', `Requires ${parentSetting.id} to satisfy its dependency`)
    }
  }

  return enabledFallbackState(setting)
}

export function getSettingDependencyStatus(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
): SettingDependencyStatus {
  const evaluation = evaluateSettingControlState(setting, allSettings, values)
  return evaluation.enabled ? { disabled: false } : { disabled: true, reason: evaluation.reason }
}

export function getSettingWriteDisposition(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
): SettingWriteDisposition {
  const nextValue = values[setting.id]
  if (nextValue == null) return 'omit'

  const evaluation = evaluateSettingControlState(setting, allSettings, values)
  if (!evaluation.enabled) {
    return evaluation.write_policy === 'preserve_existing' ? 'preserve' : 'omit'
  }

  if (nextValue === getSettingBaselineValue(setting)) return 'omit'
  if (setting.control.kind === 'text' && nextValue.trim().length === 0) return 'omit'
  return 'write'
}

export function isSettingDisabled(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
) {
  return !evaluateSettingControlState(setting, allSettings, values).enabled
}

export function getSettingDisabledReason(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
) {
  return evaluateSettingControlState(setting, allSettings, values).reason
}

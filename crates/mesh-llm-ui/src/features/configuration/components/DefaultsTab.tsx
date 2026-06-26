import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  BrainCircuit,
  Cog,
  Cpu,
  Filter,
  Gauge,
  Image,
  MemoryStick,
  Network,
  RotateCcw,
  Server,
  ShieldCheck,
  type LucideIcon
} from 'lucide-react'
import { configurationNavigationIconClassName } from '@/features/configuration/components/configuration-navigation-class-names'
import {
  SettingsCategoryRail,
  SettingsPreviewRail,
  SettingsRow,
  SettingsSection,
  SettingsSummaryBanner
} from '@/features/configuration/components/settings/SettingsScaffold'
import { ConfigurationDefaultsControl } from '@/features/configuration/components/settings/ConfigurationDefaultsControl'
import { SettingInfoTrigger } from '@/features/configuration/components/settings/DisabledControlFrame'
import { SettingResetButton } from '@/features/configuration/components/settings/SettingResetButton'
import { validateConfigurationSettingValue } from '@/features/configuration/components/settings/schema-field-validation'
import { configurationControlDetailBuckets } from '@/features/configuration/components/settings/ConfigurationDefaultsControl'
import { useDefaultsSettingsState } from '@/features/configuration/hooks/useDefaultsSettingsState'
import {
  evaluateSettingControlState,
  getSettingBaselineValue,
  getSettingValue
} from '@/features/configuration/lib/settings-utils'
import {
  defaultSettingTomlPlacement,
  defaultSettingTomlScalar,
  shouldOmitDefaultSettingValue,
  shouldOmitSettingFromGeneratedToml
} from '@/features/configuration/lib/build-toml'
import type {
  ConfigurationDefaultsCategory,
  ConfigurationDefaultsCategoryId,
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues,
  ConfigurationTomlSectionId
} from '@/features/app-tabs/types'
import { env } from '@/lib/env'
import { cn } from '@/lib/cn'

const categoryIcons: Partial<Record<ConfigurationDefaultsCategoryId, LucideIcon>> = {
  meshllm: Cpu,
  telemetry: Gauge,
  'runtime-policy': Cog,
  network: Network,
  attestation: ShieldCheck,
  runtime: Cpu,
  memory: MemoryStick,
  'speculative-decoding': BrainCircuit,
  advanced: Cog,
  'request-defaults': Filter,
  'skippy-transport': Network,
  multimodal: Image,
  'advanced-server': Server
}

const defaultsCategoryOrder: readonly ConfigurationDefaultsCategoryId[] = [
  'meshllm',
  'telemetry',
  'runtime-policy',
  'network',
  'attestation',
  'runtime',
  'memory',
  'speculative-decoding',
  'advanced',
  'request-defaults',
  'skippy-transport',
  'multimodal',
  'advanced-server'
]

const defaultsSectionOrder: readonly ConfigurationTomlSectionId[] = [
  'gpu',
  'telemetry',
  'telemetry.metrics',
  'runtime',
  'owner_control',
  'mesh_requirements',
  'defaults',
  'defaults.model_fit',
  'defaults.hardware',
  'defaults.throughput',
  'defaults.skippy',
  'defaults.speculative',
  'defaults.request_defaults',
  'defaults.multimodal',
  'defaults.advanced.server'
]
const SHOW_ADVANCED_STORAGE_KEY = `${env.storageNamespace}:configuration-defaults:show-advanced:v1`

type DefaultsTabProps = {
  data: ConfigurationDefaultsHarnessData
  values: ConfigurationDefaultsValues
  onSettingValueChange: (settingId: string, value: string) => void
  onResetAll?: () => void
  configFilePath?: string
  readOnlyNotice?: ReactNode
  previewTitle?: string
  previewTip?: ReactNode
  screenLabel?: string
  summaryDescription?: ReactNode
  summaryStatus?: string
  summaryTitle?: string
  summaryTitleId?: string
}

type DefaultsPreviewLine =
  | { kind: 'blank'; id: string }
  | { kind: 'section'; id: string; value: string }
  | { kind: 'pair'; id: string; keyName: string; value: string }

function categoryOrderIndex(category: ConfigurationDefaultsCategory) {
  if (category.order !== undefined) return category.order
  const categoryId = category.id
  const index = defaultsCategoryOrder.indexOf(categoryId)
  return index === -1 ? defaultsCategoryOrder.length : index
}

function buildDefaultsPreviewLines(
  data: ConfigurationDefaultsHarnessData,
  values: ConfigurationDefaultsValues,
  settings: readonly ConfigurationDefaultsSetting[] = data.settings
): DefaultsPreviewLine[] {
  const mainLines: DefaultsPreviewLine[] = []
  const sectionGroups = new Map<string, DefaultsPreviewLine[]>()
  const categoryById = new Map(data.categories.map((category) => [category.id, category] as const))

  for (const setting of settings) {
    if (shouldOmitSettingFromGeneratedToml(setting, data.settings, values)) continue

    const value = getSettingValue(setting, values)
    if (value === getSettingBaselineValue(setting)) continue
    if (shouldOmitDefaultSettingValue(setting, value)) continue

    const placement = defaultSettingTomlPlacement(setting, categoryById)
    const line: DefaultsPreviewLine = {
      kind: 'pair',
      id: setting.id,
      keyName: placement.key,
      value: defaultSettingTomlScalar(setting, value)
    }
    const sectionPath = placement.sectionPath
    if (!sectionPath) {
      mainLines.push(line)
      continue
    }

    const groupLines = sectionGroups.get(sectionPath) ?? [
      {
        kind: 'section',
        id: `defaults-${sectionPath.replaceAll('.', '-')}-section`,
        value: `[${sectionPath}]`
      }
    ]
    groupLines.push(line)
    sectionGroups.set(sectionPath, groupLines)
  }

  const orderedSectionPaths = [
    ...defaultsSectionOrder.filter((sectionPath) => sectionGroups.has(sectionPath)),
    ...Array.from(sectionGroups.keys()).filter(
      (sectionPath) => !defaultsSectionOrder.includes(sectionPath as ConfigurationTomlSectionId)
    )
  ]

  const groupedLines = orderedSectionPaths.flatMap((sectionPath, index) => {
    const lines = sectionGroups.get(sectionPath) ?? []
    const spacer = { kind: 'blank' as const, id: `defaults-preview-${sectionPath.replaceAll('.', '-')}-spacer` }
    return mainLines.length > 0 || index > 0 ? [spacer, ...lines] : lines
  })

  return [...mainLines, ...groupedLines]
}

function renderDefaultsPreview(lines: readonly DefaultsPreviewLine[]) {
  return lines
    .map((line) => {
      if (line.kind === 'blank') return ''
      if (line.kind === 'section') return line.value
      return `${line.keyName} = ${line.value}`
    })
    .join('\n')
}

function settingDescription(setting: ConfigurationDefaultsSetting) {
  return setting.description
}

function settingLabelAccessory(
  setting: ConfigurationDefaultsSetting,
  visibleDetails: readonly string[],
  disabledDetails: readonly string[],
  resetAction?: ReactNode
) {
  if (visibleDetails.length === 0 && disabledDetails.length === 0 && !resetAction) return undefined

  return (
    <div className="flex items-center gap-1.5">
      {visibleDetails.length > 0 ? (
        <SettingInfoTrigger
          details={visibleDetails}
          detailsId={`${setting.id}-schema-details`}
          label={`Setting information: ${setting.label}`}
        />
      ) : null}
      {disabledDetails.length > 0 ? (
        <SettingInfoTrigger
          details={disabledDetails}
          detailsId={`${setting.id}-schema-details-disabled`}
          label={`Why unavailable: ${setting.label}`}
        />
      ) : null}
      {resetAction}
    </div>
  )
}

function readShowAdvancedSettings() {
  if (typeof window === 'undefined') return false

  try {
    return window.localStorage.getItem(SHOW_ADVANCED_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

function writeShowAdvancedSettings(showAdvanced: boolean) {
  if (typeof window === 'undefined') return

  try {
    if (showAdvanced) window.localStorage.setItem(SHOW_ADVANCED_STORAGE_KEY, 'true')
    else window.localStorage.removeItem(SHOW_ADVANCED_STORAGE_KEY)
  } catch {
    return
  }
}

function sectionSubtitle(category: ConfigurationDefaultsCategory) {
  if (category.id === 'meshllm') return 'Local process settings'
  if (category.id === 'telemetry') return 'Opt-in metrics export and queue settings'
  if (category.id === 'runtime-policy') return 'Runtime reconciliation behavior'
  if (category.id === 'network') return 'Owner-control listener settings'
  if (category.id === 'attestation') return 'Certified-build admission requirements'
  if (category.id === 'memory') return 'VRAM accounting and KV cache policy'
  if (category.id === 'speculative-decoding') return 'Speculative draft policy defaults'
  if (category.id === 'request-defaults') return 'Request-time sampling and reasoning defaults'
  return category.help
}

function DefaultsSection({
  category,
  settings,
  allSettings,
  values,
  onSettingValueChange
}: {
  category: ConfigurationDefaultsCategory
  settings: readonly ConfigurationDefaultsSetting[]
  allSettings: readonly ConfigurationDefaultsSetting[]
  values: ConfigurationDefaultsValues
  onSettingValueChange: (settingId: string, value: string) => void
}) {
  const Icon = categoryIcons[category.id] ?? Cog

  return (
    <SettingsSection
      id={`defaults-${category.id}`}
      icon={<Icon aria-hidden="true" className="size-[18px]" strokeWidth={1.9} />}
      title={category.label}
      subtitle={sectionSubtitle(category)}
    >
      {settings.map((setting, settingIndex) => {
        const value = getSettingValue(setting, values)
        const evaluatedAvailability = evaluateSettingControlState(setting, allSettings, values)
        const disabled = !evaluatedAvailability.enabled
        const disabledReason = evaluatedAvailability.reason
        const dirty = value !== setting.control.value
        const availability = {
          disabled,
          note: evaluatedAvailability.note,
          reason: evaluatedAvailability.reason,
          source: evaluatedAvailability.source,
          writePolicy: evaluatedAvailability.write_policy
        }
        const details = configurationControlDetailBuckets(setting, value, availability)
        const validation = disabled ? { valid: true } : validateConfigurationSettingValue(setting, value)
        const descriptionId = `${setting.id}-description`
        const validationId = `${setting.id}-validation`
        const ariaDescribedBy = validation.message ? `${descriptionId} ${validationId}` : descriptionId
        const resetAction =
          dirty && setting.mutability === 'restart-required' ? (
            <SettingResetButton
              label={`Reset ${setting.label} to default`}
              onClick={() => onSettingValueChange(setting.id, setting.control.value)}
            />
          ) : undefined

        return (
          <SettingsRow
            className={cn(settingIndex === 0 && 'border-t-0')}
            disabled={disabled}
            disabledReason={disabledReason}
            dirty={dirty}
            errorMessage={validation.message}
            errorMessageId={validationId}
            hintId={descriptionId}
            key={setting.id}
            label={setting.label}
            labelAccessory={settingLabelAccessory(
              setting,
              details.visibleDetails,
              details.disabledDetails,
              resetAction
            )}
            hint={settingDescription(setting)}
            showDisabledReason={false}
          >
            <ConfigurationDefaultsControl
              ariaDescribedBy={ariaDescribedBy}
              availability={availability}
              disabled={disabled}
              invalid={!validation.valid}
              setting={setting}
              value={value}
              onChange={(nextValue) => onSettingValueChange(setting.id, nextValue)}
            />
          </SettingsRow>
        )
      })}
    </SettingsSection>
  )
}

export function DefaultsTab({
  data,
  values,
  onSettingValueChange,
  onResetAll,
  configFilePath,
  readOnlyNotice,
  previewTitle = '[defaults]',
  previewTip,
  screenLabel = 'Configuration · defaults',
  summaryDescription,
  summaryStatus,
  summaryTitle = 'Inherited defaults',
  summaryTitleId = 'defaults-summary-heading'
}: DefaultsTabProps) {
  const { activeCategoryId, setActiveCategoryId } = useDefaultsSettingsState(data)
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(() => readShowAdvancedSettings())
  const changedCount = data.settings.filter(
    (setting) => getSettingValue(setting, values) !== setting.control.value
  ).length
  useEffect(() => {
    writeShowAdvancedSettings(showAdvancedSettings)
  }, [showAdvancedSettings])

  const visibleSettings = useMemo(
    () => data.settings.filter((setting) => showAdvancedSettings || setting.visibility !== 'advanced'),
    [data.settings, showAdvancedSettings]
  )
  const settingsByCategory = useMemo(() => {
    const grouped = new Map<ConfigurationDefaultsCategoryId, ConfigurationDefaultsSetting[]>()

    for (const setting of visibleSettings) {
      const group = grouped.get(setting.categoryId) ?? []
      group.push(setting)
      grouped.set(setting.categoryId, group)
    }

    return grouped
  }, [visibleSettings])
  const categories = useMemo(
    () =>
      data.categories
        .map((category, originalIndex) => ({
          ...category,
          originalIndex,
          count: settingsByCategory.get(category.id)?.length ?? 0
        }))
        .filter((category) => category.count > 0)
        .sort(
          (left, right) =>
            categoryOrderIndex(left) - categoryOrderIndex(right) || left.originalIndex - right.originalIndex
        )
        .map((category) => {
          const Icon = categoryIcons[category.id]
          const CategoryIcon = Icon ?? Cog
          const { originalIndex: _originalIndex, ...resolvedCategory } = category

          return {
            ...resolvedCategory,
            icon: <CategoryIcon aria-hidden="true" className={configurationNavigationIconClassName} strokeWidth={1.7} />
          }
        }),
    [data.categories, settingsByCategory]
  )
  const previewLines = useMemo(() => buildDefaultsPreviewLines(data, values, data.settings), [data, values])

  useEffect(() => {
    if (categories.length === 0) return
    if (categories.some((category) => category.id === activeCategoryId)) return

    setActiveCategoryId(categories[0].id)
  }, [activeCategoryId, categories, setActiveCategoryId])

  const selectCategory = (categoryId: string) => {
    setActiveCategoryId(categoryId)
    const target = document.getElementById(`defaults-${categoryId}`)
    if (target && 'scrollIntoView' in target) target.scrollIntoView({ block: 'start', behavior: 'smooth' })
  }

  return (
    <section aria-labelledby={summaryTitleId} className="space-y-[14px]" data-screen-label={screenLabel}>
      <SettingsSummaryBanner
        titleId={summaryTitleId}
        action={
          <div className="flex flex-wrap items-center gap-2">
            <button
              aria-pressed={showAdvancedSettings}
              className={cn(
                'ui-control inline-flex h-[30px] items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-control)] font-semibold',
                showAdvancedSettings && 'border-accent bg-accent/10 text-accent'
              )}
              onClick={() => setShowAdvancedSettings((current) => !current)}
              type="button"
            >
              {showAdvancedSettings ? 'Hide advanced' : 'Show advanced'}
            </button>
            <button
              className={cn(
                'ui-control inline-flex h-[30px] items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-control)] font-semibold'
              )}
              disabled={changedCount === 0}
              onClick={onResetAll}
              type="button"
            >
              <RotateCcw aria-hidden="true" className="size-3.5" />
              Reset all
            </button>
          </div>
        }
        description={
          summaryDescription ?? (
            <>
              These values flow into every{' '}
              <span className="rounded border border-border-soft bg-surface px-1 font-mono text-foreground">
                [defaults.*]
              </span>{' '}
              section, and every{' '}
              <span className="rounded border border-border-soft bg-surface px-1 font-mono text-foreground">
                [[models]]
              </span>{' '}
              entry can override them with matching nested model sections. Per-placement overrides surface as{' '}
              <span className="rounded border border-border-soft bg-surface px-1 font-mono text-accent">OVERRIDE</span>{' '}
              badges in Model Deployment.
            </>
          )
        }
        status={summaryStatus ?? (changedCount === 0 ? 'all upstream' : `${changedCount} modified`)}
        title={summaryTitle}
      />

      {readOnlyNotice}

      <div className="grid min-w-0 gap-[14px] xl:grid-cols-[250px_minmax(0,1fr)_minmax(280px,340px)]">
        <SettingsCategoryRail
          activeId={activeCategoryId}
          categories={categories}
          footer={
            <span className="inline-flex flex-col gap-1 whitespace-nowrap leading-none">
              <span className="type-label text-fg-faint">Configuration Path</span>
              <span className="font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">
                {configFilePath ?? '~/.mesh-llm/config.toml'}
              </span>
            </span>
          }
          onSelect={selectCategory}
        />

        <div className="min-w-0 space-y-[14px]">
          {categories.map((category) => (
            <DefaultsSection
              category={category}
              allSettings={data.settings}
              key={category.id}
              onSettingValueChange={onSettingValueChange}
              settings={settingsByCategory.get(category.id) ?? []}
              values={values}
            />
          ))}
        </div>

        <div className="hidden min-w-0 xl:block">
          <SettingsPreviewRail
            title={previewTitle}
            code={renderDefaultsPreview(previewLines)}
            tip={
              previewTip ?? (
                <>
                  Adjust placements in <span className="text-foreground">Model Deployment</span> to override these for a
                  single model.
                </>
              )
            }
          />
        </div>
      </div>
    </section>
  )
}

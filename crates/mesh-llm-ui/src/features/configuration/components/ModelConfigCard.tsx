import * as RadioGroup from '@radix-ui/react-radio-group'
import { animate, createScope } from 'animejs'
import { Logs, Trash2 } from 'lucide-react'
import { cn } from '@/lib/cn'
import { useLayoutEffect, useRef, useState } from 'react'
import { NativeSelect } from '@/components/ui/NativeSelect'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { CtxSlider } from '@/features/configuration/components/CtxSlider'
import { CTX_MAX, CTX_MIN, fmtCtx, snapCtx } from '@/features/configuration/components/ctx-slider-utils'
import { MetaPill } from '@/components/ui/MetaPill'
import {
  contextGB,
  contextGBPerK,
  findModel,
  kvGB,
  modelFamilyColorKey,
  modelWeightsGB
} from '@/features/configuration/lib/config-math'
import { formatGB as formatConfigGB } from '@/features/configuration/lib/config-display'
import type {
  ConfigAssign,
  ConfigAssignModelConfig,
  ConfigModel,
  ConfigNode,
  ConfigurationModelPlacementOptions
} from '@/features/app-tabs/types'

type ModelConfigCardProps = {
  assign: ConfigAssign
  node: ConfigNode
  models?: ConfigModel[]
  modelPlacementOptions?: ConfigurationModelPlacementOptions
  containerFreeGB: number
  selected?: boolean
  controlTabIndex?: number
  onPick?: () => void
  onCtxChange: (ctx: number) => void
  onConfigChange?: (config: ConfigAssignModelConfig) => void
  onRemove: () => void
}

type ChoiceOption<T extends string> = { value: T; label: string }

const batchProfileOptions: ChoiceOption<NonNullable<ConfigAssignModelConfig['batchProfile']>>[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'throughput', label: 'Performance' },
  { value: 'saver', label: 'Saver' }
]

const splitModeOptions: ChoiceOption<NonNullable<ConfigAssignModelConfig['splitMode']>>[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'layer', label: 'Layer' },
  { value: 'row', label: 'Row' }
]

const flashAttentionOptions: ChoiceOption<NonNullable<ConfigAssignModelConfig['flashAttention']>>[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'enabled', label: 'On' },
  { value: 'disabled', label: 'Off' }
]

const defaultKvCacheTypeOptions = ['f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1']

function kvCacheTypeOptions(values: string[] | undefined): ChoiceOption<string>[] {
  const concreteValues = (values?.length ? values : defaultKvCacheTypeOptions).filter((value) => value !== 'auto')
  return [{ value: 'auto', label: 'Auto' }, ...concreteValues.map((value) => ({ value, label: value }))]
}

function formatGBLabel(value: number): string {
  return `${formatConfigGB(value, { fixedFractionDigits: 1 })} GB`
}
function formatShortfallGB(value: number): string {
  if (value < 1) return `${Math.max(1, Math.round(value * 1024)).toLocaleString()} MB`
  return formatGBLabel(value)
}
function formatOptional(value: number | string | undefined): string {
  return value === undefined ? 'auto' : `${value}`
}

function hasOverride(value: string | number | undefined): boolean {
  if (value === undefined || value === 'auto' || value === '') return false
  if (typeof value === 'number') return value !== 1
  return true
}

function ConfigControl({
  label,
  children,
  override
}: {
  label: string
  children: React.ReactNode
  override?: boolean
}) {
  return (
    <div className="grid grid-cols-[120px_auto_96px] items-center gap-3">
      <span className="text-[length:var(--density-type-caption)] text-fg-dim">{label}</span>
      <div className="w-fit max-w-[720px]">{children}</div>
      <div
        className={`shrink-0 text-[length:var(--density-type-label)] font-semibold uppercase leading-none text-warn ${override ? '' : 'invisible'}`}
      >
        OVERRIDE
      </div>
    </div>
  )
}

function updateModelConfig(
  current: ConfigAssignModelConfig | undefined,
  patch: Partial<ConfigAssignModelConfig>
): ConfigAssignModelConfig {
  const next = { ...(current ?? {}), ...patch }
  for (const key of Object.keys(next) as Array<keyof ConfigAssignModelConfig>) {
    const value = next[key]
    if (value === undefined || value === '' || value === 'auto') delete next[key]
  }
  return next
}

function SegmentedChoice<T extends string>({
  label,
  value,
  options,
  tabIndex,
  onChange
}: {
  label: string
  value: T
  options: ChoiceOption<T>[]
  tabIndex?: number
  onChange: (value: T) => void
}) {
  return (
    <div className="w-fit">
      <SegmentedControl
        ariaLabel={label}
        itemTabIndex={tabIndex}
        name={`model-config-${label}`}
        onValueChange={(v) => onChange(v as T)}
        options={options.map((o) => ({ label: o.label, value: o.value }))}
        value={value}
        variant="pill"
      />
    </div>
  )
}

function TextConfigInput({
  label,
  value,
  placeholder,
  tabIndex,
  onChange
}: {
  label: string
  value: string
  placeholder: string
  tabIndex?: number
  onChange: (value: string) => void
}) {
  return (
    <div className="w-fit">
      <label htmlFor={label} className="sr-only">
        {label}
      </label>
      <input
        id={label}
        className="h-8 w-[420px] max-w-[560px] min-w-0 rounded-[var(--radius)] border border-border-soft bg-background/70 px-3 font-mono text-[length:var(--density-type-caption)] text-foreground outline-none transition-[border-color,box-shadow] placeholder:text-fg-faint focus:border-accent focus:shadow-[var(--shadow-control-focus)]"
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        tabIndex={tabIndex}
        value={value}
      />
    </div>
  )
}

function SlotMeter({
  value,
  tabIndex,
  onChange
}: {
  value: number
  tabIndex?: number
  onChange: (value: number) => void
}) {
  const min = 1
  const max = 16
  const slotRange = Math.max(1, Math.floor(max - min + 1))
  const slotOptions = Array.from({ length: slotRange }, (_, index) => min + index)
  const selectedSlots = Math.max(min, Math.min(max, value))

  const selectSlotFromPointer = (event: React.PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    if (bounds.width <= 0) return

    const clampedX = Math.max(0, Math.min(bounds.width, event.clientX - bounds.left))
    const slotCount = Math.max(min, Math.min(max, min + Math.floor((clampedX / bounds.width) * slotRange)))
    onChange(slotCount)
  }

  return (
    <RadioGroup.Root
      aria-label="Slots"
      className="w-[560px] max-w-[720px] min-w-0 rounded-[6px] border border-border-soft bg-panel-strong px-2.5 py-2 text-[length:var(--density-type-caption)] text-fg-dim"
      name="model-config-slots"
      onValueChange={(v) => onChange(Number(v))}
      value={String(selectedSlots)}
    >
      <div className="mb-1.5 flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <span className="text-fg-faint">Parallel slots</span>
        <span className="font-mono text-fg-dim">
          {selectedSlots} slot{selectedSlots === 1 ? '' : 's'}
        </span>
      </div>
      <div
        className="grid w-full min-w-0 touch-none select-none gap-px"
        onPointerDown={(event) => {
          event.preventDefault()
          if (event.currentTarget.setPointerCapture) event.currentTarget.setPointerCapture(event.pointerId)
          selectSlotFromPointer(event)
        }}
        onPointerMove={(event) => {
          if (
            event.currentTarget.hasPointerCapture &&
            !event.currentTarget.hasPointerCapture(event.pointerId) &&
            event.buttons !== 1
          )
            return
          selectSlotFromPointer(event)
        }}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId)
          }
        }}
        style={{ gridTemplateColumns: `repeat(${slotRange}, minmax(0, 1fr))` }}
      >
        {slotOptions.map((slotCount) => (
          <RadioGroup.Item
            aria-label={`${slotCount} slot${slotCount === 1 ? '' : 's'}`}
            className="group flex w-full min-w-0 appearance-none items-center rounded-[2px] border-0 bg-transparent py-1 outline-none transition-[opacity] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
            key={slotCount}
            tabIndex={tabIndex}
            value={String(slotCount)}
          >
            <span
              className={cn(
                'h-1.5 w-full rounded-[1px] transition-colors',
                slotCount <= selectedSlots ? 'bg-accent opacity-100' : 'bg-border-soft opacity-50'
              )}
              data-slot-empty={slotCount > selectedSlots ? 'true' : undefined}
            />
          </RadioGroup.Item>
        ))}
      </div>
    </RadioGroup.Root>
  )
}

export function ModelConfigCard({
  assign,
  node,
  models,
  modelPlacementOptions,
  containerFreeGB,
  selected = false,
  controlTabIndex,
  onPick,
  onCtxChange,
  onConfigChange,
  onRemove
}: ModelConfigCardProps) {
  const cardRef = useRef<HTMLElement>(null)
  const [advanced, setAdvanced] = useState(false)
  const model = findModel(assign.modelId, models)

  useLayoutEffect(() => {
    if (!cardRef.current) return

    const reduceMotion =
      typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const scope = createScope({ root: cardRef }).add(() => {
      animate(cardRef.current!, {
        opacity: [0, 1],
        y: reduceMotion ? 0 : [-6, 0],
        scaleY: reduceMotion ? 1 : [0.98, 1],
        duration: reduceMotion ? 0 : 180,
        ease: 'out(4)'
      })
    })

    return () => scope.revert()
  }, [])

  if (!model) return null

  const kv = contextGB(model, assign.ctx)
  const displayKV = kvGB(model, assign.ctx)
  const weightsGB = modelWeightsGB(model)
  const total = weightsGB + kv
  const ctxGBPerK = contextGBPerK(model)
  const maxAllowedCtx = Math.max(CTX_MIN, ctxGBPerK > 0 ? (Math.max(0, containerFreeGB) / ctxGBPerK) * 1024 : CTX_MIN)
  const safeCtx = Math.min(CTX_MAX, maxAllowedCtx)
  const selectedGpu = node.gpus.find((gpu) => gpu.idx === assign.containerIdx)
  const locationLabel =
    node.placement === 'pooled'
      ? `${node.hostname} pool`
      : `GPU ${assign.containerIdx} · ${selectedGpu?.name ?? 'unknown'}`
  const paramsLabel = model.paramsLabel ?? `${model.paramsB}B`
  const modelShortfallGB = Math.max(0, -containerFreeGB)
  const ctxShortfallGB = Math.max(0, kv - Math.max(0, containerFreeGB))
  const hasError = modelShortfallGB > 0 || ctxShortfallGB > 0
  const errorText =
    modelShortfallGB > 0
      ? `ERROR: model allocation exceeds this container by ${formatGBLabel(modelShortfallGB)}.`
      : `ERROR: context allocation needs ${formatShortfallGB(ctxShortfallGB)} more KV cache.`
  const modelConfig = assign.config ?? {}
  const patchConfig = (patch: Partial<ConfigAssignModelConfig>) =>
    onConfigChange?.(updateModelConfig(modelConfig, patch))
  const cacheTypeKOptions = kvCacheTypeOptions(modelPlacementOptions?.cacheTypeK)
  const cacheTypeVOptions = kvCacheTypeOptions(modelPlacementOptions?.cacheTypeV)

  return (
    <article
      aria-invalid={hasError}
      className={`mt-2 select-none rounded-[var(--radius-lg)] border bg-panel px-5 py-4 transition-[border-color,box-shadow] ${hasError ? 'border-bad shadow-[var(--shadow-surface-error-inset)]' : selected ? 'border-[color:color-mix(in_oklab,var(--color-accent)_44%,var(--color-border))] shadow-[var(--shadow-surface-selected)]' : 'border-border-soft'}`}
      data-model-selection-area="true"
      onPointerDown={onPick}
      ref={cardRef}
      style={{
        transformOrigin: 'top center'
      }}
    >
      <div className="flex flex-wrap items-center gap-2.5">
        <span aria-hidden="true" className="size-2 rounded-full" data-model-family-color={modelFamilyColorKey(model)} />
        <h3 className="min-w-0 flex-1 truncate text-[length:var(--density-type-body)] font-semibold">{model.name}</h3>
        <span className="font-mono text-[length:var(--density-type-label)] text-fg-faint">{model.family}</span>
        <MetaPill size="annotation">{paramsLabel}</MetaPill>
        <span className="ml-auto text-[length:var(--density-type-caption)] text-fg-dim">
          on <span className="font-mono text-fg">{locationLabel}</span>
        </span>
        <button
          className="ui-control inline-flex size-[30px] shrink-0 items-center justify-center rounded-[var(--radius)] border transition-colors hover:bg-background/80"
          onClick={() => setAdvanced((v) => !v)}
          tabIndex={controlTabIndex}
          title="Advanced controls"
          type="button"
          aria-label="Toggle advanced controls"
          aria-pressed={advanced}
        >
          <Logs aria-hidden="true" className="size-[15px]" strokeWidth={1.8} />
        </button>
        <button
          className="ui-control-destructive inline-flex size-[30px] shrink-0 items-center justify-center rounded-[var(--radius)] border"
          onClick={onRemove}
          tabIndex={controlTabIndex}
          title="Remove"
          type="button"
          aria-label={`Remove ${model.name} from ${locationLabel}`}
        >
          <Trash2 aria-hidden="true" className="size-[15px]" strokeWidth={1.8} />
        </button>
      </div>

      {hasError ? (
        <div className="mt-2.5 rounded-[var(--radius)] border border-bad/70 bg-bad/10 px-2.5 py-2 text-[length:var(--density-type-caption)] font-medium text-bad">
          {errorText} Reduce context, remove another allocation, or move the model to a larger{' '}
          {node.placement === 'pooled' ? 'pool' : 'GPU'}.
        </div>
      ) : null}

      <div className="mt-2.5">
        <CtxSlider
          value={assign.ctx}
          onChange={onCtxChange}
          maxCtx={safeCtx}
          invalid={hasError}
          controlTabIndex={controlTabIndex}
        />
        <p className="mt-1.5 text-right text-[length:var(--density-type-label)] text-fg-faint">
          max ≈ {fmtCtx(snapCtx(maxAllowedCtx))} on this {node.placement === 'pooled' ? 'pool' : 'GPU'}
        </p>
      </div>

      <div className="mt-4 grid gap-x-6 border-t border-border-soft pt-4 md:grid-cols-[minmax(0,1fr)_240px]">
        <div className="min-w-0 flex flex-col gap-3">
          <span className="text-[length:var(--density-type-caption)] font-semibold uppercase text-fg-faint">
            Runtime
          </span>
          <ConfigControl label="Slots" override={hasOverride(modelConfig.slots)}>
            <SlotMeter
              value={modelConfig.slots ?? 1}
              tabIndex={controlTabIndex}
              onChange={(slots) => patchConfig({ slots })}
            />
          </ConfigControl>

          {advanced && (
            <>
              <div className="border-t border-border-soft pt-3" />
              <ConfigControl label="Profile" override={hasOverride(modelConfig.profile)}>
                <TextConfigInput
                  label="Profile"
                  value={modelConfig.profile ?? ''}
                  placeholder="default"
                  tabIndex={controlTabIndex}
                  onChange={(profile) => patchConfig({ profile })}
                />
              </ConfigControl>

              <ConfigControl label="Split mode" override={hasOverride(modelConfig.splitMode)}>
                <SegmentedChoice
                  label="Split mode"
                  value={modelConfig.splitMode ?? 'auto'}
                  options={splitModeOptions}
                  tabIndex={controlTabIndex}
                  onChange={(splitMode) => patchConfig({ splitMode })}
                />
              </ConfigControl>

              <ConfigControl label="Tensor split" override={hasOverride(modelConfig.tensorSplit)}>
                <TextConfigInput
                  label="Tensor split"
                  value={modelConfig.tensorSplit ?? ''}
                  placeholder="e.g. 50,50 (auto)"
                  tabIndex={controlTabIndex}
                  onChange={(tensorSplit) => patchConfig({ tensorSplit })}
                />
              </ConfigControl>

              <div className="border-t border-border-soft pt-3" />
              <ConfigControl label="mmproj" override={hasOverride(modelConfig.mmproj)}>
                <TextConfigInput
                  label="mmproj"
                  value={modelConfig.mmproj ?? ''}
                  placeholder="path/to/mmproj.gguf (none)"
                  tabIndex={controlTabIndex}
                  onChange={(mmproj) => patchConfig({ mmproj })}
                />
              </ConfigControl>

              <ConfigControl label="Draft model" override={hasOverride(modelConfig.draftModelPath)}>
                <TextConfigInput
                  label="Draft model"
                  value={modelConfig.draftModelPath ?? ''}
                  placeholder="path/to/draft.gguf (none)"
                  tabIndex={controlTabIndex}
                  onChange={(draftModelPath) => patchConfig({ draftModelPath })}
                />
              </ConfigControl>

              <div className="border-t border-border-soft pt-3" />
              <ConfigControl label="Flash attention" override={hasOverride(modelConfig.flashAttention)}>
                <SegmentedChoice
                  label="Flash attention"
                  value={modelConfig.flashAttention ?? 'auto'}
                  options={flashAttentionOptions}
                  tabIndex={controlTabIndex}
                  onChange={(flashAttention) => patchConfig({ flashAttention })}
                />
              </ConfigControl>

              <ConfigControl label="Cache type K" override={hasOverride(modelConfig.cacheTypeK)}>
                <div className="w-fit max-w-[220px]">
                  <NativeSelect
                    ariaLabel="Cache type K"
                    className="min-w-[160px] max-w-[220px]"
                    disabled={cacheTypeKOptions.length <= 1}
                    name="cache-type-k"
                    onValueChange={(value) => patchConfig({ cacheTypeK: value })}
                    options={cacheTypeKOptions.map((o) => ({ label: o.label, value: o.value }))}
                    value={modelConfig.cacheTypeK ?? 'auto'}
                  />
                </div>
              </ConfigControl>

              <ConfigControl label="Cache type V" override={hasOverride(modelConfig.cacheTypeV)}>
                <div className="w-fit max-w-[220px]">
                  <NativeSelect
                    ariaLabel="Cache type V"
                    className="min-w-[160px] max-w-[220px]"
                    disabled={cacheTypeVOptions.length <= 1}
                    name="cache-type-v"
                    onValueChange={(value) => patchConfig({ cacheTypeV: value })}
                    options={cacheTypeVOptions.map((o) => ({ label: o.label, value: o.value }))}
                    value={modelConfig.cacheTypeV ?? 'auto'}
                  />
                </div>
              </ConfigControl>
            </>
          )}

          {!advanced && (
            <>
              <ConfigControl label="Performance Profile" override={hasOverride(modelConfig.batchProfile)}>
                <SegmentedChoice
                  label="Performance Profile"
                  value={modelConfig.batchProfile ?? 'auto'}
                  options={batchProfileOptions}
                  tabIndex={controlTabIndex}
                  onChange={(batchProfile) => patchConfig({ batchProfile })}
                />
              </ConfigControl>
              <p className="pl-[132px] text-[length:var(--density-type-label)] text-fg-faint">
                Controls quantization, batch/ubatch
              </p>
            </>
          )}
        </div>
        <aside className="w-[240px] shrink-0 self-start rounded-[var(--radius-lg)] border border-border-soft bg-background/45 p-3">
          <div className="text-[length:var(--density-type-caption)] font-semibold uppercase text-fg-faint">Memory</div>
          <dl className="mt-2 space-y-1.5 text-[length:var(--density-type-caption)]">
            <div className="flex justify-between gap-3">
              <dt className="text-fg-dim">Weights</dt>
              <dd className="font-mono text-foreground">{formatGBLabel(weightsGB)}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-dim">KV cache</dt>
              <dd className="font-mono text-foreground">{formatGBLabel(displayKV)}</dd>
            </div>
            <div className="flex justify-between gap-3 border-t border-border-soft pt-1.5">
              <dt className="text-foreground">Total</dt>
              <dd className="font-mono text-foreground">{formatGBLabel(total)}</dd>
            </div>
          </dl>
          <div className="mt-4 text-[length:var(--density-type-caption)] font-semibold uppercase text-fg-faint">
            Model
          </div>
          <dl className="mt-2 space-y-1.5 text-[length:var(--density-type-caption)]">
            <div className="flex justify-between gap-3">
              <dt className="text-fg-dim">Layers</dt>
              <dd className="font-mono text-foreground">{formatOptional(model.layers)}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-dim">Heads</dt>
              <dd className="font-mono text-foreground">{formatOptional(model.heads)}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-dim">Embed</dt>
              <dd className="font-mono text-foreground">{formatOptional(model.embed)}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-dim">Tokenizer</dt>
              <dd className="font-mono text-foreground">{formatOptional(model.tokenizer)}</dd>
            </div>
          </dl>
          <p className="mt-4 border-t border-border-soft pt-2 text-[length:var(--density-type-label)] text-fg-faint">
            max ctx ≈ {fmtCtx(snapCtx(maxAllowedCtx))} on this {node.placement === 'pooled' ? 'pool' : 'GPU'}
          </p>
        </aside>
      </div>
    </article>
  )
}

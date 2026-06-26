import * as RadioGroup from '@radix-ui/react-radio-group'
import type { PointerEvent } from 'react'
import { Slider, type SliderBoundary } from '@/components/ui/Slider'
import { cn } from '@/lib/cn'
import { CtxSlider } from '@/features/configuration/components/CtxSlider'
import { NumberField } from '@/features/configuration/components/settings/NumberField'
import {
  effectiveRendererId,
  numericMetadataForSetting,
  type SchemaSettingControlProps
} from '@/features/configuration/components/settings/schema-control-utils'

function finiteNumber(value: string, fallback = 0) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function formatNumberValue(setting: SchemaSettingControlProps['setting'], value: string) {
  if (setting.canonicalPath?.endsWith('.gpu_layers') && value === '-1') return 'auto/all'
  if (setting.canonicalPath === 'defaults.hardware.safety_margin_gb') return finiteNumber(value).toFixed(1)
  if (
    setting.canonicalPath === 'defaults.request_defaults.temperature' ||
    setting.canonicalPath === 'defaults.request_defaults.top_p' ||
    setting.canonicalPath === 'defaults.request_defaults.repeat_penalty'
  )
    return finiteNumber(value).toFixed(2)
  return value
}

function numberUnit(setting: SchemaSettingControlProps['setting'], value: string) {
  if (setting.canonicalPath?.endsWith('.gpu_layers') && value === '-1') return undefined
  if (effectiveRendererId(setting) === 'slot-meter') return `slot${value === '1' ? '' : 's'}`
  if (setting.canonicalPath === 'defaults.hardware.safety_margin_gb') return 'GB'
  if (
    setting.canonicalPath === 'defaults.request_defaults.temperature' ||
    setting.canonicalPath === 'defaults.request_defaults.top_p' ||
    setting.canonicalPath === 'defaults.request_defaults.repeat_penalty'
  )
    return undefined
  return numericMetadataForSetting(setting).unit
}

type SchemaNumberControlProps = SchemaSettingControlProps & {
  readonly lowerBound?: SliderBoundary
  readonly upperBound?: SliderBoundary
}

function SlotCountControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  onChange,
  setting,
  value
}: SchemaSettingControlProps) {
  const control = numericMetadataForSetting(setting)
  const min = control.min ?? 1
  const max = control.max ?? 16
  const slotRange = Math.max(1, Math.floor(max - min + 1))
  const slotOptions = Array.from({ length: slotRange }, (_, index) => min + index)
  const selectedSlots = Math.max(min, Math.min(max, finiteNumber(value, min)))
  const estimatedGb = (selectedSlots * 0.3).toFixed(1)

  const selectSlotFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    if (disabled) return
    const bounds = event.currentTarget.getBoundingClientRect()
    if (bounds.width <= 0) return

    const clampedX = Math.max(0, Math.min(bounds.width, event.clientX - bounds.left))
    const slotCount = Math.max(min, Math.min(max, min + Math.floor((clampedX / bounds.width) * slotRange)))
    onChange(String(slotCount))
  }

  return (
    <RadioGroup.Root
      aria-describedby={ariaDescribedBy}
      aria-label={setting.label}
      aria-invalid={invalid ? 'true' : undefined}
      aria-disabled={disabled ? 'true' : undefined}
      className={cn(
        'min-w-0 max-w-full rounded-[6px] border border-border-soft bg-panel-strong px-2.5 py-2 text-[length:var(--density-type-caption)] text-fg-dim',
        disabled && 'cursor-not-allowed opacity-60',
        invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
      )}
      disabled={disabled}
      name={'name' in setting.control ? setting.control.name : setting.id}
      onValueChange={onChange}
      value={String(selectedSlots)}
    >
      <div className="mb-1.5 flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <span className="text-fg-faint">est. KV @ 16K ctx</span>
        <span className="font-mono text-fg-dim">
          {estimatedGb} GB · {selectedSlots} × 0.30 GB
        </span>
      </div>
      <div
        className="grid w-full min-w-0 touch-none select-none gap-px"
        data-testid="defaults-slot-meter"
        onPointerDown={(event) => {
          if (disabled) return
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
            className={cn(
              'group flex w-full min-w-0 appearance-none items-center rounded-[2px] border-0 bg-transparent py-1 outline-none transition-[opacity] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
              disabled ? 'cursor-not-allowed' : 'cursor-pointer'
            )}
            key={slotCount}
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

export function SchemaNumberControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  lowerBound,
  onChange,
  setting,
  upperBound,
  value
}: SchemaNumberControlProps) {
  const numeric = numericMetadataForSetting(setting)
  const step = numeric.step ?? (setting.valueSchema?.kind === 'float' ? 0.01 : 1)
  const sliderMin = numeric.min
  const sliderMax = numeric.max
  const showSlider = sliderMin !== undefined && sliderMax !== undefined
  const unit = numberUnit(setting, value)
  const rendererId = effectiveRendererId(setting)

  if (rendererId === 'slot-meter') {
    return (
      <div className="grid min-w-[280px] gap-2">
        <SlotCountControl
          ariaDescribedBy={ariaDescribedBy}
          disabled={disabled}
          invalid={invalid}
          onChange={onChange}
          setting={setting}
          value={value}
        />
      </div>
    )
  }

  if (rendererId === 'context-slider') {
    const minCtx = sliderMin ?? 512
    const selectedCtx = finiteNumber(value, minCtx)
    const maxCtx = sliderMax ?? selectedCtx

    return (
      <div className="min-w-[320px]">
        <CtxSlider
          ariaDescribedBy={ariaDescribedBy}
          ariaLabel={setting.label}
          disabled={disabled}
          exactValuePosition="top-right"
          exactValueVisibility="shown"
          invalid={invalid}
          maxCtx={maxCtx}
          maxSelectableCtx={maxCtx}
          minSelectableCtx={minCtx}
          onChange={(nextValue) => onChange(String(nextValue))}
          showHeader={false}
          value={selectedCtx}
        />
      </div>
    )
  }

  if (showSlider) {
    return (
      <div>
        <Slider
          ariaDescribedBy={ariaDescribedBy}
          ariaLabel={setting.label}
          disabled={disabled}
          formatValue={(nextValue) => formatNumberValue(setting, nextValue)}
          invalid={invalid}
          lowerBound={lowerBound}
          max={sliderMax}
          min={sliderMin}
          name={'name' in setting.control ? setting.control.name : setting.id}
          onValueChange={onChange}
          step={step}
          unit={unit}
          upperBound={upperBound}
          value={value}
          valueLabelAlign="right"
          valueLabelPlacement="bottom"
        />
      </div>
    )
  }

  return (
    <div className="grid min-w-[280px] justify-items-end gap-2">
      <NumberField
        aria-describedby={ariaDescribedBy}
        aria-label={setting.label}
        disabled={disabled}
        invalid={invalid}
        max={numeric.max}
        min={numeric.min}
        name={'name' in setting.control ? setting.control.name : setting.id}
        onChange={(event) => onChange(event.currentTarget.value)}
        step={step}
        type="number"
        unit={unit}
        value={value}
      />
    </div>
  )
}

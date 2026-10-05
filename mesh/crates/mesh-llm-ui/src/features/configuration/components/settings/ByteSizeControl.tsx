import { useState } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Stepper } from '@/components/ui/Stepper'
import { cn } from '@/lib/cn'
import { numericMetadataForSetting, type SchemaSettingControlProps } from './schema-control-utils'

const FALLBACK_UNITS = [
  { value: 'bytes', label: 'B', multiplier: 1 },
  { value: 'kilobytes', label: 'KB', multiplier: 1_024 },
  { value: 'megabytes', label: 'MB', multiplier: 1_048_576 },
  { value: 'gigabytes', label: 'GB', multiplier: 1_073_741_824 }
] as const

function displayNumber(value: number) {
  return Number(value.toFixed(3))
}

function initialUnit(bytes: number, units: readonly { value: string; multiplier: number }[]) {
  return [...units].reverse().find((unit) => bytes >= unit.multiplier)?.value ?? units[0]?.value ?? 'bytes'
}

function compactBytes(bytes: number) {
  if (bytes >= 1_073_741_824) return `${displayNumber(bytes / 1_073_741_824)} GB`
  if (bytes >= 1_048_576) return `${displayNumber(bytes / 1_048_576)} MB`
  if (bytes >= 1_024) return `${displayNumber(bytes / 1_024)} KB`
  return `${bytes} B`
}

export function ByteSizeControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  onChange,
  setting,
  value
}: SchemaSettingControlProps) {
  const units = setting.displayUnits?.length ? setting.displayUnits : FALLBACK_UNITS
  const bytes = Number.isFinite(Number(value)) ? Number(value) : 0
  const [unitValue, setUnitValue] = useState(() => initialUnit(bytes, units))
  const unit = units.find((candidate) => candidate.value === unitValue) ?? units[0] ?? FALLBACK_UNITS[0]
  const numeric = numericMetadataForSetting(setting)
  const min = numeric.min === undefined ? undefined : numeric.min / unit.multiplier
  const max = numeric.max === undefined ? undefined : numeric.max / unit.multiplier
  const shownValue = displayNumber(bytes / unit.multiplier)
  const step = max !== undefined && max <= 10 ? 0.1 : 1

  return (
    <div className="grid min-w-[280px] justify-items-end gap-1.5">
      <div className="flex items-center gap-1.5">
        <Stepper
          aria-describedby={ariaDescribedBy}
          aria-invalid={invalid}
          aria-label={setting.label}
          disabled={disabled}
          inputClassName="w-[78px] font-mono tabular-nums"
          max={max}
          min={min}
          onChange={(next) => onChange(String(Math.round(next * unit.multiplier)))}
          step={step}
          value={shownValue}
        />
        <Select disabled={disabled} onValueChange={setUnitValue} value={unit.value}>
          <SelectTrigger
            aria-label={`${setting.label} unit`}
            className={cn(
              'ui-control h-8 w-[72px] rounded-[var(--radius)] border-border bg-surface px-2 font-mono text-[length:var(--density-type-control)]',
              invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
            )}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="surface-menu-panel min-w-[72px] bg-panel-strong">
            {units.map((candidate) => (
              <SelectItem className="font-mono text-xs" key={candidate.value} value={candidate.value}>
                {candidate.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {numeric.min !== undefined && numeric.max !== undefined ? (
        <span className="font-mono text-[length:var(--density-type-annotation)] text-fg-faint">
          {compactBytes(numeric.min)}–{compactBytes(numeric.max)}
        </span>
      ) : null}
    </div>
  )
}

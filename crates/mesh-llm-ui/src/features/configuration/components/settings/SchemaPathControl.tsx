import { useState } from 'react'
import { FolderOpen, LoaderCircle } from 'lucide-react'
import { cn } from '@/lib/cn'
import { env } from '@/lib/env'
import {
  TEXT_FIELD_BASE_CLASS,
  type SchemaSettingControlProps
} from '@/features/configuration/components/settings/schema-control-utils'

export function SchemaPathControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  onChange,
  setting,
  value
}: SchemaSettingControlProps) {
  const supportsPicker = setting.rendererId === 'host-directory-picker'
  const [pickerPending, setPickerPending] = useState(false)
  const [pickerError, setPickerError] = useState<string>()

  const pickDirectory = async () => {
    setPickerPending(true)
    setPickerError(undefined)
    try {
      const response = await fetch(`${env.managementApiUrl}/api/runtime/pick-directory`, { method: 'POST' })
      const payload = (await response.json().catch(() => ({}))) as {
        path?: string
        error?: string
        cancelled?: boolean
      }
      if (!response.ok) throw new Error(payload.error ?? `Directory picker returned HTTP ${response.status}`)
      if (payload.path) onChange(payload.path)
    } catch (error) {
      setPickerError(
        error instanceof Error ? error.message : 'Directory picker is unavailable; enter the host path manually'
      )
    } finally {
      setPickerPending(false)
    }
  }

  return (
    <div className="grid min-w-[280px] gap-1.5">
      <div className="flex items-center gap-1.5">
        <input
          aria-describedby={ariaDescribedBy}
          aria-invalid={invalid ? 'true' : undefined}
          aria-label={setting.label}
          autoCapitalize="off"
          autoCorrect="off"
          className={cn(
            TEXT_FIELD_BASE_CLASS,
            'w-full min-w-0',
            (disabled || pickerPending) && 'cursor-not-allowed opacity-60',
            invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
          )}
          disabled={disabled || pickerPending}
          name={'name' in setting.control ? setting.control.name : setting.id}
          onChange={(event) => onChange(event.currentTarget.value)}
          placeholder={'placeholder' in setting.control ? setting.control.placeholder : './path/to/file'}
          spellCheck={false}
          type="text"
          value={value}
        />
        {supportsPicker ? (
          <button
            className="ui-control inline-flex h-8 shrink-0 items-center gap-1.5 rounded-[var(--radius)] border bg-surface px-2.5 text-[length:var(--density-type-control)] text-fg-dim transition-colors hover:bg-secondary hover:text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-60"
            disabled={disabled || pickerPending}
            onClick={() => void pickDirectory()}
            type="button"
          >
            {pickerPending ? (
              <LoaderCircle aria-hidden="true" className="size-3.5 animate-spin" />
            ) : (
              <FolderOpen aria-hidden="true" className="size-3.5" />
            )}
            Browse
          </button>
        ) : null}
      </div>
      {pickerError ? <p className="max-w-[420px] text-xs leading-snug text-bad">{pickerError}</p> : null}
    </div>
  )
}

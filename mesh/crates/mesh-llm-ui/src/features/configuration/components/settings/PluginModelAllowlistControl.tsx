import { useState } from 'react'
import { cn } from '@/lib/cn'
import type { SchemaSettingControlProps } from '@/features/configuration/components/settings/schema-control-utils'
import {
  EMPTY_MODEL_ALLOWLIST_TEXT,
  modelAllowlistEditorText,
  modelAllowlistFromEditorText,
  parsePluginModelAllowlist
} from '@/features/configuration/lib/plugin-model-allowlist'

/** Edits a plugin entry's endpoint model allowlist one model ID per line, so
 * an ID containing a comma stays one ID. Every edit is encoded with the same
 * rules the configuration adapter saves with. */
export function PluginModelAllowlistControl(props: SchemaSettingControlProps) {
  const { ariaDescribedBy, disabled = false, invalid = false, onChange, setting, value } = props
  const [draft, setDraft] = useState(() => modelAllowlistEditorText(value))
  // Keep the text being typed (a new empty line, say) while it still encodes
  // to the current value; show the value itself after a reset or reload.
  const text = modelAllowlistFromEditorText(draft) === value ? draft : modelAllowlistEditorText(value)
  const models = value.trim().length === 0 ? [] : parsePluginModelAllowlist(value)

  return (
    <div className="grid min-w-[280px] gap-2">
      <textarea
        aria-describedby={ariaDescribedBy}
        aria-invalid={invalid ? 'true' : undefined}
        aria-label={setting.label}
        className={cn(
          'ui-control min-h-[88px] w-full rounded-[var(--radius)] border bg-surface px-2.5 py-2 font-mono text-[length:var(--density-type-control)] text-foreground outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
          disabled && 'cursor-not-allowed opacity-60',
          invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
        )}
        disabled={disabled}
        name={'name' in setting.control ? setting.control.name : setting.id}
        onChange={(event) => {
          setDraft(event.currentTarget.value)
          onChange(modelAllowlistFromEditorText(event.currentTarget.value))
        }}
        placeholder={
          'placeholder' in setting.control && setting.control.placeholder
            ? setting.control.placeholder
            : `One model ID per line; ${EMPTY_MODEL_ALLOWLIST_TEXT} alone shares none`
        }
        value={text}
      />
      {models.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {models.map((model, index) => (
            <span
              className="inline-flex items-center rounded-full border border-border-soft bg-panel-strong px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] text-fg-dim"
              key={`${index}:${model}`}
            >
              {model}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

import { NativeSelect } from '@/components/ui/NativeSelect'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { cn } from '@/lib/cn'
import {
  effectiveRendererId,
  resolvedChoiceOptions,
  type SchemaSettingControlProps
} from '@/features/configuration/components/settings/schema-control-utils'

function choiceItemClassName(setting: SchemaSettingControlProps['setting']) {
  return cn(
    'min-w-[64px] capitalize',
    setting.control.kind === 'choice' && setting.control.presentation === 'toggle' && 'min-w-[38px]',
    setting.canonicalPath?.endsWith('.flash_attention') && 'min-w-[38px]',
    effectiveRendererId(setting) === 'kv-cache-policy' && 'min-w-[58px]',
    setting.canonicalPath === 'defaults.speculative.mode' && 'min-w-[72px]',
    setting.canonicalPath === 'defaults.speculative.draft_selection_policy' && 'min-w-[86px]',
    setting.canonicalPath === 'defaults.speculative.pairing_fault' && 'min-w-[104px]',
    setting.canonicalPath === 'defaults.request_defaults.reasoning_format' && 'min-w-[118px]'
  )
}

function KvPolicyMatrix({ policy }: { readonly policy: string }) {
  const rows = [
    { label: '<5GB', detail: 'K F16 · V F16', active: policy === 'auto' || policy === 'quality' },
    { label: '5–50GB', detail: 'K q8_0 · V q4_0', active: policy === 'auto' || policy === 'balanced' },
    { label: '≥50GB', detail: 'K q4_0 · V q4_0', active: policy === 'auto' || policy === 'saver' }
  ]

  return (
    <fieldset
      className="mt-2 grid max-w-[420px] grid-cols-3 gap-1.5 font-mono text-[length:var(--density-type-annotation)]"
      aria-label="KV cache memory tiers"
    >
      {rows.map((row) => (
        <span
          className={cn(
            'rounded-[5px] border px-2 py-1.5 text-fg-faint transition-[background-color,border-color,opacity]',
            row.active
              ? 'border-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_6%,var(--color-panel-strong))] opacity-100'
              : 'border-border-soft bg-panel-strong opacity-50'
          )}
          data-kv-tier-active={row.active ? 'true' : undefined}
          key={row.label}
        >
          <span className="block text-[9.5px] uppercase tracking-[0.05em]">{row.label}</span>
          <span className="mt-0.5 block text-fg-dim">{row.detail}</span>
        </span>
      ))}
    </fieldset>
  )
}

export function SchemaChoiceControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  onChange,
  setting,
  value
}: SchemaSettingControlProps) {
  const options = resolvedChoiceOptions(setting)
  const presentation = setting.control.kind === 'choice' ? (setting.control.presentation ?? 'segmented') : 'segmented'

  return (
    <div
      aria-disabled={disabled ? 'true' : undefined}
      className={cn(
        'min-w-0',
        effectiveRendererId(setting) === 'kv-cache-policy' && 'flex flex-col items-stretch md:items-end'
      )}
      data-setting-control-disabled={disabled ? 'true' : undefined}
    >
      {presentation === 'select' ? (
        <NativeSelect
          ariaDescribedBy={ariaDescribedBy}
          ariaLabel={setting.label}
          disabled={disabled}
          invalid={invalid}
          name={'name' in setting.control ? setting.control.name : setting.id}
          onValueChange={onChange}
          options={options}
          value={value}
        />
      ) : (
        <SegmentedControl
          ariaDescribedBy={ariaDescribedBy}
          ariaLabel={setting.label}
          disabled={disabled}
          invalid={invalid}
          itemClassName={choiceItemClassName(setting)}
          name={'name' in setting.control ? setting.control.name : setting.id}
          onValueChange={onChange}
          options={options}
          value={value}
          variant="pill"
        />
      )}
      {effectiveRendererId(setting) === 'kv-cache-policy' ? <KvPolicyMatrix policy={value} /> : null}
    </div>
  )
}

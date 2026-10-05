import { NativeSelect } from '@/components/ui/NativeSelect'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { cn } from '@/lib/cn'
import {
  resolvedChoiceOptions,
  type SchemaSettingControlProps
} from '@/features/configuration/components/settings/schema-control-utils'

function choiceItemClassName(setting: SchemaSettingControlProps['setting']) {
  return cn(
    'min-w-[64px] capitalize',
    setting.control.kind === 'choice' && setting.control.presentation === 'toggle' && 'min-w-[38px]',
    setting.canonicalPath?.endsWith('.flash_attention') && 'min-w-[38px]',
    setting.canonicalPath === 'defaults.speculative.mode' && 'min-w-[72px]',
    setting.canonicalPath === 'defaults.speculative.draft_selection_policy' && 'min-w-[86px]',
    setting.canonicalPath === 'defaults.speculative.pairing_fault' && 'min-w-[104px]',
    setting.canonicalPath === 'defaults.request_defaults.reasoning_format' && 'min-w-[118px]'
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
  const options = resolvedChoiceOptions(setting, value)
  const presentation = setting.control.kind === 'choice' ? (setting.control.presentation ?? 'segmented') : 'segmented'
  const selectedDescription = options.find((option) => option.value === value)?.description

  return (
    <div
      aria-disabled={disabled ? 'true' : undefined}
      className="min-w-0"
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
      {selectedDescription ? (
        <p className="mt-1.5 max-w-[360px] text-[length:var(--density-type-annotation)] leading-snug text-fg-faint">
          {selectedDescription}
        </p>
      ) : null}
    </div>
  )
}

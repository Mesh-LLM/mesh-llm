import { Check, Plus } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import type { ConfigurationSettingObjectSchema } from '@/features/app-tabs/types'
import { SchemaObjectFields } from '@/features/configuration/components/settings/SchemaObjectFields'
import {
  TEXT_FIELD_BASE_CLASS,
  type SchemaSettingControlProps
} from '@/features/configuration/components/settings/schema-control-utils'
import {
  parseSchemaObjectArrayValue,
  schemaValueFromInput,
  setValuePrimaryProperty,
  singularItemLabel,
  updateObjectPath
} from '@/features/configuration/lib/schema-object-array'
import { cn } from '@/lib/cn'

const CONFIRM_MS = 2000

type SchemaSetValueArrayControlProps = SchemaSettingControlProps & {
  readonly itemSchema: ConfigurationSettingObjectSchema
}

type SetRowProps = {
  readonly ariaLabel: string
  readonly committed: string | undefined
  readonly confirmed: boolean
  readonly confirmedText: string
  readonly disabled: boolean
  readonly ghost?: boolean
  readonly onRemove?: () => void
  readonly onSet: (value: string) => void
  readonly placeholder: string | undefined
}

// A value row: greyed until Set, black once Set, a brief checkmark after each Set.
function SetRow({
  ariaLabel,
  committed,
  confirmed,
  confirmedText,
  disabled,
  ghost = false,
  onRemove,
  onSet,
  placeholder
}: SetRowProps) {
  const [draft, setDraft] = useState(committed ?? '')

  const shown = ghost ? (placeholder ?? '') : draft
  const unset = ghost || committed === undefined
  const canSet = !disabled && shown.trim().length > 0 && (ghost || shown !== committed)

  const set = () => {
    onSet(shown.trim())
  }

  return (
    <div className="flex min-w-0 items-center gap-2">
      <input
        aria-label={ariaLabel}
        autoCapitalize="off"
        autoCorrect="off"
        className={cn(
          TEXT_FIELD_BASE_CLASS,
          'w-full min-w-[200px]',
          ghost ? 'text-fg-dim' : 'text-foreground',
          disabled && 'cursor-not-allowed opacity-60'
        )}
        data-set={unset ? 'false' : 'true'}
        disabled={disabled}
        onChange={(event) => setDraft(event.currentTarget.value)}
        placeholder={placeholder}
        readOnly={ghost}
        spellCheck={false}
        value={ghost ? shown : draft}
      />
      <Button disabled={!canSet} onClick={set} size="sm" type="button" variant="outline">
        Set
      </Button>
      <span aria-live="polite" className="inline-flex min-w-4 shrink-0 items-center gap-1 text-good type-caption">
        {confirmed ? (
          <>
            <Check aria-hidden="true" className="size-4" />
            {confirmedText}
          </>
        ) : null}
      </span>
      {onRemove ? (
        <Button disabled={disabled} onClick={onRemove} size="sm" type="button" variant="ghost">
          Clear
        </Button>
      ) : null}
    </div>
  )
}

// Generic renderer for an array of objects whose main field is set explicitly. With nothing
// set, the setting's `placeholder` shows greyed; the value only exists once the user clicks Set.
export function SchemaSetValueArrayControl({
  disabled = false,
  invalid = false,
  itemSchema,
  onChange,
  setting,
  value
}: SchemaSetValueArrayControlProps) {
  const primary = setValuePrimaryProperty(itemSchema)
  const items = parseSchemaObjectArrayValue(value) ?? []
  const [pending, setPending] = useState<readonly number[]>([])
  const nextPendingId = useRef(0)
  const [confirmedRow, setConfirmedRow] = useState<number | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  const flash = (row: number) => {
    setConfirmedRow(row)
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setConfirmedRow(null), CONFIRM_MS)
  }
  const placeholder = 'placeholder' in setting.control ? setting.control.placeholder : undefined
  if (!primary) return null
  const restartRequired =
    setting.restartScope === 'process_restart' ||
    setting.restartScope === 'mesh_restart' ||
    setting.mutability === 'restart-required'
  const savedText = restartRequired ? 'Saved. Takes effect when mesh-llm restarts.' : 'Saved.'
  const itemName = singularItemLabel(setting.label).toLowerCase()

  const emit = (next: readonly Record<string, unknown>[]) => onChange(JSON.stringify(next))
  const rest: ConfigurationSettingObjectSchema = {
    ...itemSchema,
    properties: (itemSchema.properties ?? []).filter((property) => property.name !== primary.name)
  }
  const stringOf = (item: Record<string, unknown>) => {
    const raw = item[primary.name]
    return typeof raw === 'string' ? raw : ''
  }

  // Keyed by value (and repeat count), so removing an earlier row never hands its draft to a later one.
  const seen = new Map<string, number>()
  const rowKeys = items.map((item) => {
    const text = stringOf(item)
    const count = seen.get(text) ?? 0
    seen.set(text, count + 1)
    return `row-${count}-${text}`
  })

  return (
    <div className="grid w-full min-w-0 gap-2.5" data-invalid={invalid ? 'true' : undefined}>
      {items.length === 0 && pending.length === 0 && placeholder ? (
        <div className="grid gap-1">
          <SetRow
            ariaLabel={`${setting.label} (not set)`}
            committed={undefined}
            confirmed={confirmedRow === 0}
            confirmedText={savedText}
            disabled={disabled}
            ghost
            onSet={(next) => {
              emit([{ [primary.name]: next }])
              flash(0)
            }}
            placeholder={placeholder}
          />
          <p className="type-caption text-fg-dim">{`Not set: no ${itemName} until you click Set.`}</p>
        </div>
      ) : null}
      {items.map((item, index) => (
        <div className="grid min-w-0 gap-2" key={rowKeys[index]}>
          <SetRow
            ariaLabel={`${setting.label} ${index + 1}`}
            committed={stringOf(item)}
            confirmed={confirmedRow === index}
            confirmedText={savedText}
            disabled={disabled}
            onRemove={() => emit(items.filter((_, itemIndex) => itemIndex !== index))}
            onSet={(next) => {
              emit(
                items.map((existing, itemIndex) =>
                  itemIndex === index ? { ...existing, [primary.name]: next } : existing
                )
              )
              flash(index)
            }}
            placeholder={undefined}
          />
          {rest.properties && rest.properties.length > 0 ? (
            <SchemaObjectFields
              ariaPrefix={`${setting.label} ${index + 1}`}
              disabled={disabled}
              invalid={invalid}
              onValueChange={(path, property, next) =>
                emit(
                  items.map((existing, itemIndex) =>
                    itemIndex === index
                      ? updateObjectPath(
                          existing,
                          path,
                          next.length === 0 && !property.required
                            ? undefined
                            : schemaValueFromInput(property.value_schema, next)
                        )
                      : existing
                  )
                )
              }
              schema={rest}
              value={item}
            />
          ) : null}
        </div>
      ))}
      {pending.map((id, index) => (
        <SetRow
          ariaLabel={`${setting.label} ${items.length + index + 1}`}
          committed={undefined}
          confirmed={false}
          confirmedText={savedText}
          disabled={disabled}
          key={`pending-${id}`}
          onSet={(next) => {
            emit([...items, { [primary.name]: next }])
            setPending((ids) => ids.filter((pendingId) => pendingId !== id))
            flash(items.length)
          }}
          placeholder={undefined}
        />
      ))}
      <Button
        className="justify-self-start"
        disabled={disabled}
        onClick={() => setPending((ids) => [...ids, nextPendingId.current++])}
        size="sm"
        type="button"
        variant="ghost"
      >
        <Plus aria-hidden="true" className="size-4" />
        Add your own
      </Button>
    </div>
  )
}

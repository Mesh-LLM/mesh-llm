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
  createSchemaObjectDefault,
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
  readonly ariaDescribedBy?: string
  readonly ariaLabel: string
  readonly committed: string | undefined
  readonly confirmed: boolean
  readonly confirmedText: string
  readonly disabled: boolean
  readonly ghost?: boolean
  readonly invalid: boolean
  readonly onRemove?: () => void
  readonly onSet: (value: string) => void
  readonly placeholder: string | undefined
}

// A value row: greyed until Set, black once Set, a brief checkmark after each Set.
function SetRow({
  ariaDescribedBy,
  ariaLabel,
  committed,
  confirmed,
  confirmedText,
  disabled,
  ghost = false,
  invalid,
  onRemove,
  onSet,
  placeholder
}: SetRowProps) {
  const [draft, setDraft] = useState(committed ?? '')

  const shown = ghost ? (placeholder ?? '') : draft
  const unset = ghost || committed === undefined
  const canSet = !disabled && shown.trim().length > 0 && (ghost || shown.trim() !== committed)

  return (
    <div className="flex min-w-0 items-center gap-2">
      <input
        aria-describedby={ariaDescribedBy}
        aria-invalid={invalid ? 'true' : undefined}
        aria-label={ariaLabel}
        autoCapitalize="off"
        autoCorrect="off"
        className={cn(
          TEXT_FIELD_BASE_CLASS,
          'w-full min-w-[200px]',
          ghost ? 'text-fg-dim' : 'text-foreground',
          disabled && 'cursor-not-allowed opacity-60',
          invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
        )}
        data-set={unset ? 'false' : 'true'}
        disabled={disabled}
        onChange={(event) => setDraft(event.currentTarget.value)}
        placeholder={placeholder}
        readOnly={ghost}
        spellCheck={false}
        value={shown}
      />
      <Button disabled={!canSet} onClick={() => onSet(shown.trim())} size="sm" type="button" variant="outline">
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
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  itemSchema,
  onChange,
  setting,
  value
}: SchemaSetValueArrayControlProps) {
  const primary = setValuePrimaryProperty(itemSchema)
  const parsed = parseSchemaObjectArrayValue(value)
  const items = parsed ?? []
  // Committed rows carry an id that survives removals, so a row never inherits another row's draft.
  // Row keys also include the stored value, so an outside change (Undo, Redo, Revert) remounts the
  // row with the restored value. If the list changes length from outside, the ids are re-issued.
  const [rows, setRows] = useState(() => ({ ids: items.map((_, index) => index), next: items.length }))
  let current = rows
  if (rows.ids.length !== items.length) {
    const ids = rows.ids.slice(0, items.length)
    let next = rows.next
    while (ids.length < items.length) ids.push(next++)
    current = { ids, next }
    setRows(current)
  }
  const rowIds = current.ids
  const [pending, setPending] = useState<readonly number[]>([])
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
  if (!parsed) {
    return (
      <p className="type-caption text-bad" role="alert">
        {setting.label} must be a valid JSON array of objects.
      </p>
    )
  }
  const restartRequired =
    setting.restartScope === 'process_restart' ||
    setting.restartScope === 'mesh_restart' ||
    setting.mutability === 'restart-required'
  // Set only edits the page's draft; the console writes it when the user clicks Save config.
  const savedText = restartRequired
    ? 'Set. Click Save config to keep it; takes effect when mesh-llm restarts.'
    : 'Set. Click Save config to keep it.'
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
  const removePending = (id: number) => setPending((ids) => ids.filter((pendingId) => pendingId !== id))

  return (
    <div className="grid w-full min-w-0 gap-2.5">
      {items.length === 0 && pending.length === 0 && placeholder ? (
        <div className="grid gap-1">
          <SetRow
            ariaDescribedBy={ariaDescribedBy}
            ariaLabel={`${setting.label} (not set)`}
            committed={undefined}
            confirmed={confirmedRow === 0}
            confirmedText={savedText}
            disabled={disabled}
            ghost
            invalid={invalid}
            onSet={(next) => {
              emit([{ ...createSchemaObjectDefault(itemSchema), [primary.name]: next }])
              flash(0)
            }}
            placeholder={placeholder}
          />
          <p className="type-caption text-fg-dim">{`Not set: no ${itemName} until you click Set.`}</p>
        </div>
      ) : null}
      {items.map((item, index) => (
        <div className="grid min-w-0 gap-2" key={`row-${rowIds[index]}-${stringOf(item)}`}>
          <SetRow
            ariaDescribedBy={ariaDescribedBy}
            ariaLabel={`${setting.label} ${index + 1}`}
            committed={stringOf(item)}
            confirmed={confirmedRow === index}
            confirmedText={savedText}
            disabled={disabled}
            invalid={invalid}
            onRemove={() => {
              setRows({ ids: rowIds.filter((_, rowIndex) => rowIndex !== index), next: current.next })
              setConfirmedRow(null)
              emit(items.filter((_, itemIndex) => itemIndex !== index))
            }}
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
              ariaDescribedBy={ariaDescribedBy}
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
          ariaDescribedBy={ariaDescribedBy}
          ariaLabel={`${setting.label} ${items.length + index + 1}`}
          committed={undefined}
          confirmed={false}
          confirmedText={savedText}
          disabled={disabled}
          invalid={invalid}
          key={`pending-${id}`}
          onRemove={() => removePending(id)}
          onSet={(next) => {
            setRows({ ids: [...rowIds, id], next: current.next })
            emit([...items, { ...createSchemaObjectDefault(itemSchema), [primary.name]: next }])
            removePending(id)
            flash(items.length)
          }}
          placeholder={undefined}
        />
      ))}
      <Button
        className="justify-self-start"
        disabled={disabled}
        onClick={() => {
          setRows({ ids: rowIds, next: current.next + 1 })
          setPending((ids) => [...ids, current.next])
        }}
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

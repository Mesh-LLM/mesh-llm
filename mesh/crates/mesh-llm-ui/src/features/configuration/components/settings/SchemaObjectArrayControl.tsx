import { ArrowDown, ArrowUp, Plus, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { ConfigurationSettingObjectProperty, ConfigurationSettingObjectSchema } from '@/features/app-tabs/types'
import { SchemaObjectFields } from '@/features/configuration/components/settings/SchemaObjectFields'
import type { SchemaSettingControlProps } from '@/features/configuration/components/settings/schema-control-utils'
import {
  createSchemaObjectDefault,
  isSchemaRecord,
  parseSchemaObjectArrayValue,
  schemaValueFromInput
} from '@/features/configuration/lib/schema-object-array'

type SchemaObjectArrayControlProps = SchemaSettingControlProps & {
  readonly itemSchema: ConfigurationSettingObjectSchema
}

function singularItemLabel(label: string): string {
  const noun = label.trim().split(/\s+/).at(-1) ?? 'item'
  const singular = noun.endsWith('ies')
    ? `${noun.slice(0, -3)}y`
    : noun.endsWith('s') && !noun.endsWith('ss')
      ? noun.slice(0, -1)
      : noun
  return `${singular.charAt(0).toUpperCase()}${singular.slice(1)}`
}

function updateObjectPath(
  value: Record<string, unknown>,
  path: readonly string[],
  nextValue: unknown
): Record<string, unknown> {
  const [name, ...remainingPath] = path
  if (!name) return value

  if (remainingPath.length === 0) {
    const next = { ...value }
    if (nextValue === undefined) delete next[name]
    else next[name] = nextValue
    return next
  }

  const child = value[name]
  return {
    ...value,
    [name]: updateObjectPath(isSchemaRecord(child) ? child : {}, remainingPath, nextValue)
  }
}

function moveItem(
  items: readonly Record<string, unknown>[],
  fromIndex: number,
  toIndex: number
): readonly Record<string, unknown>[] {
  const fromItem = items[fromIndex]
  const toItem = items[toIndex]
  if (!fromItem || !toItem) return items

  return items.map((item, index) => {
    if (index === fromIndex) return toItem
    if (index === toIndex) return fromItem
    return item
  })
}

export function SchemaObjectArrayControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  itemSchema,
  onChange,
  setting,
  value
}: SchemaObjectArrayControlProps) {
  const parsedItems = parseSchemaObjectArrayValue(value)
  const items = parsedItems ?? []
  const itemLabel = singularItemLabel(setting.label)
  const itemName = itemLabel.toLowerCase()
  const emitItems = (nextItems: readonly Record<string, unknown>[]) => onChange(JSON.stringify(nextItems))

  const changeField = (
    itemIndex: number,
    path: readonly string[],
    property: ConfigurationSettingObjectProperty,
    nextInput: string
  ) => {
    const nextValue =
      nextInput.length === 0 && !property.required ? undefined : schemaValueFromInput(property.value_schema, nextInput)
    emitItems(items.map((item, index) => (index === itemIndex ? updateObjectPath(item, path, nextValue) : item)))
  }

  return (
    <div className="grid w-full min-w-0 gap-2.5">
      {parsedItems ? (
        items.map((item, index) => {
          const position = index + 1
          const ariaPrefix = `${itemLabel} ${position}`
          return (
            <section
              aria-label={ariaPrefix}
              className="grid min-w-0 gap-3 rounded-[var(--radius)] border border-border-soft bg-panel-strong p-3"
              key={`${itemLabel}-${position}`}
            >
              <div className="flex min-w-0 items-center justify-between gap-2">
                <h4 className="type-panel-title text-foreground">{ariaPrefix}</h4>
                <div className="flex shrink-0 items-center gap-1">
                  <Button
                    aria-label={`Move ${itemName} ${position} up`}
                    disabled={disabled || index === 0}
                    onClick={() => emitItems(moveItem(items, index, index - 1))}
                    size="icon"
                    type="button"
                    variant="ghost"
                  >
                    <ArrowUp aria-hidden="true" className="size-4" />
                  </Button>
                  <Button
                    aria-label={`Move ${itemName} ${position} down`}
                    disabled={disabled || index === items.length - 1}
                    onClick={() => emitItems(moveItem(items, index, index + 1))}
                    size="icon"
                    type="button"
                    variant="ghost"
                  >
                    <ArrowDown aria-hidden="true" className="size-4" />
                  </Button>
                  <Button
                    aria-label={`Remove ${itemName} ${position}`}
                    disabled={disabled}
                    onClick={() => emitItems(items.filter((_, itemIndex) => itemIndex !== index))}
                    size="icon"
                    type="button"
                    variant="ghost"
                  >
                    <Trash2 aria-hidden="true" className="size-4" />
                  </Button>
                </div>
              </div>
              <SchemaObjectFields
                ariaDescribedBy={ariaDescribedBy}
                ariaPrefix={ariaPrefix}
                disabled={disabled}
                invalid={invalid}
                onValueChange={(path, property, nextValue) => changeField(index, path, property, nextValue)}
                schema={itemSchema}
                value={item}
              />
            </section>
          )
        })
      ) : (
        <p className="type-caption text-bad" role="alert">
          {setting.label} must be a valid JSON array of objects.
        </p>
      )}
      <Button
        aria-label={`Add ${itemName}`}
        className="justify-self-start"
        disabled={disabled || !parsedItems}
        onClick={() => emitItems([...items, createSchemaObjectDefault(itemSchema)])}
        size="sm"
        type="button"
        variant="outline"
      >
        <Plus aria-hidden="true" className="size-4" />
        Add {itemName}
      </Button>
    </div>
  )
}

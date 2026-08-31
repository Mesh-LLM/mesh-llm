import { Input } from '@/components/ui/input'
import { NativeSelect, type NativeSelectOption } from '@/components/ui/NativeSelect'
import type {
  ConfigurationSettingObjectProperty,
  ConfigurationSettingObjectSchema,
  ConfigurationSettingValueSchema
} from '@/features/app-tabs/types'
import { isSchemaRecord, schemaInputValue } from '@/features/configuration/lib/schema-object-array'
import { cn } from '@/lib/cn'

type SchemaObjectFieldsProps = {
  readonly ariaDescribedBy?: string
  readonly ariaPrefix: string
  readonly disabled: boolean
  readonly invalid: boolean
  readonly onValueChange: (path: readonly string[], property: ConfigurationSettingObjectProperty, value: string) => void
  readonly path?: readonly string[]
  readonly schema: ConfigurationSettingObjectSchema
  readonly value: Record<string, unknown>
}

function schemaOptions(schema: ConfigurationSettingValueSchema): readonly NativeSelectOption[] {
  switch (schema.kind) {
    case 'boolean':
      return [
        { value: 'true', label: 'On' },
        { value: 'false', label: 'Off' }
      ]
    case 'enum':
      return schema.values.map((value) => ({ value, label: value }))
    case 'one_of': {
      const variantOptions = schema.variants.map(schemaOptions)
      return variantOptions.some((options) => options.length === 0) ? [] : variantOptions.flat()
    }
    case 'array':
    case 'float':
    case 'integer':
    case 'object':
    case 'path':
    case 'socket_addr':
    case 'string':
    case 'url':
      return []
  }
}

function inputType(schema: ConfigurationSettingValueSchema): 'number' | 'text' {
  return schema.kind === 'integer' || schema.kind === 'float' ? 'number' : 'text'
}

type SchemaLeafFieldProps = {
  readonly ariaDescribedBy?: string
  readonly ariaLabel: string
  readonly disabled: boolean
  readonly invalid: boolean
  readonly onValueChange: (value: string) => void
  readonly property: ConfigurationSettingObjectProperty
  readonly value: unknown
}

function SchemaLeafField({
  ariaDescribedBy,
  ariaLabel,
  disabled,
  invalid,
  onValueChange,
  property,
  value
}: SchemaLeafFieldProps) {
  const schemaChoices = schemaOptions(property.value_schema)
  const options =
    property.required || schemaChoices.length === 0 ? schemaChoices : [{ value: '', label: 'Unset' }, ...schemaChoices]
  const inputValue = schemaInputValue(value)

  return (
    <label className="grid min-w-0 gap-1.5">
      <span className="type-label text-fg-faint">{property.label}</span>
      {options.length > 0 ? (
        <NativeSelect
          ariaDescribedBy={ariaDescribedBy}
          ariaLabel={ariaLabel}
          className="w-full min-w-0"
          disabled={disabled}
          invalid={invalid}
          name={property.name}
          onValueChange={onValueChange}
          options={options}
          value={inputValue}
        />
      ) : (
        <Input
          aria-describedby={ariaDescribedBy}
          aria-invalid={invalid ? 'true' : undefined}
          aria-label={ariaLabel}
          className={cn(
            'ui-control h-[32px] min-w-0 bg-surface font-mono text-[length:var(--density-type-control)]',
            invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
          )}
          disabled={disabled}
          name={property.name}
          onChange={(event) => onValueChange(event.currentTarget.value)}
          required={property.required}
          step={property.value_schema.kind === 'float' ? 'any' : undefined}
          type={inputType(property.value_schema)}
          value={inputValue}
        />
      )}
    </label>
  )
}

export function SchemaObjectFields({
  ariaDescribedBy,
  ariaPrefix,
  disabled,
  invalid,
  onValueChange,
  path = [],
  schema,
  value
}: SchemaObjectFieldsProps) {
  return (
    <div className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-2">
      {(schema.properties ?? []).map((property) => {
        const propertyPath = [...path, property.name]
        const propertyValue = value[property.name]
        if (property.value_schema.kind === 'object') {
          return (
            <fieldset
              className="grid min-w-0 gap-2 rounded-[var(--radius)] border border-border-soft p-2.5 sm:col-span-2"
              key={property.name}
            >
              <legend className="type-label px-1 text-fg-faint">{property.label}</legend>
              <SchemaObjectFields
                ariaDescribedBy={ariaDescribedBy}
                ariaPrefix={ariaPrefix}
                disabled={disabled}
                invalid={invalid}
                onValueChange={onValueChange}
                path={propertyPath}
                schema={property.value_schema}
                value={isSchemaRecord(propertyValue) ? propertyValue : {}}
              />
            </fieldset>
          )
        }

        return (
          <SchemaLeafField
            ariaDescribedBy={ariaDescribedBy}
            ariaLabel={`${ariaPrefix} ${property.label}`}
            disabled={disabled}
            invalid={invalid}
            key={property.name}
            onValueChange={(nextValue) => onValueChange(propertyPath, property, nextValue)}
            property={property}
            value={propertyValue}
          />
        )
      })}
    </div>
  )
}

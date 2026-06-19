import * as v from 'valibot'
import type {
  ConfigurationDefaultsSetting,
  ConfigurationSettingValueSchema,
  ConfigurationSettingValidationConstraint
} from '@/features/app-tabs/types'
import { acceptedValuesForSetting, numericMetadataForSetting } from './schema-control-utils'

export type SchemaFieldValidationResult = {
  readonly message?: string
  readonly valid: boolean
}

function arrayItems(value: string) {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean)
}

function numericValue(value: string) {
  if (value.trim().length === 0) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function normalizedChoiceValue(value: string) {
  if (value === 'true') return 'on'
  if (value === 'false') return 'off'
  return value
}

function firstIssueMessage(result: ReturnType<typeof v.safeParse>) {
  if (result.success) return undefined
  return result.issues[0]?.message
}

function validateNumber(value: string, setting: ConfigurationDefaultsSetting, integer: boolean) {
  const parsed = numericValue(value)
  const label = setting.label
  const numeric = numericMetadataForSetting(setting)
  const schema = integer
    ? v.pipe(
        v.number(`${label} must be a number.`),
        v.integer(`${label} must be a whole number.`),
        v.check(
          (input) => numeric.min === undefined || input >= numeric.min,
          `${label} must be at least ${numeric.min}.`
        ),
        v.check(
          (input) => numeric.max === undefined || input <= numeric.max,
          `${label} must be at most ${numeric.max}.`
        )
      )
    : v.pipe(
        v.number(`${label} must be a number.`),
        v.check(
          (input) => numeric.min === undefined || input >= numeric.min,
          `${label} must be at least ${numeric.min}.`
        ),
        v.check(
          (input) => numeric.max === undefined || input <= numeric.max,
          `${label} must be at most ${numeric.max}.`
        )
      )

  return firstIssueMessage(v.safeParse(schema, parsed))
}

function validateObject(value: string, label: string) {
  if (value.trim().length === 0) return undefined

  try {
    const parsed: unknown = JSON.parse(value)
    return firstIssueMessage(
      v.safeParse(
        v.pipe(
          v.unknown(),
          v.check(
            (input) => input !== null && typeof input === 'object' && !Array.isArray(input),
            `${label} must be a JSON object.`
          )
        ),
        parsed
      )
    )
  } catch {
    return `${label} must be valid JSON.`
  }
}

function validateSchemaKind(
  value: string,
  setting: ConfigurationDefaultsSetting,
  schema: ConfigurationSettingValueSchema
): string | undefined {
  const label = setting.label

  switch (schema.kind) {
    case 'boolean':
      return firstIssueMessage(
        v.safeParse(v.picklist(['on', 'off', 'auto', 'true', 'false'], `${label} must be on, off, or auto.`), value)
      )
    case 'integer':
      return validateNumber(value, setting, true)
    case 'float':
      return validateNumber(value, setting, false)
    case 'enum':
      return firstIssueMessage(
        v.safeParse(
          v.pipe(
            v.string(),
            v.check(
              (input) => schema.values.map(normalizedChoiceValue).includes(normalizedChoiceValue(input)),
              `${label} must be one of: ${schema.values.join(', ')}.`
            )
          ),
          value
        )
      )
    case 'one_of': {
      const messages = schema.variants.map((variant) => validateSchemaKind(value, setting, variant)).filter(Boolean)
      return messages.length === schema.variants.length ? messages[0] : undefined
    }
    case 'array': {
      const items = arrayItems(value)
      const itemError = items
        .map((item) => validateSchemaKind(item, setting, schema.items))
        .find((message): message is string => typeof message === 'string')
      return itemError ? `One ${label} item is invalid: ${itemError}` : undefined
    }
    case 'object':
      return validateObject(value, label)
    case 'url':
      if (value.trim().length === 0) return undefined
      return firstIssueMessage(v.safeParse(v.pipe(v.string(), v.url(`${label} must be a full URL.`)), value))
    case 'path':
    case 'socket_addr':
    case 'string':
      return firstIssueMessage(v.safeParse(v.string(`${label} must be text.`), value))
  }
}

function numericConstraintValue(value: string | undefined) {
  if (value === undefined) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function validateConstraint(
  value: string,
  setting: ConfigurationDefaultsSetting,
  constraint: ConfigurationSettingValidationConstraint
) {
  const label = setting.label

  switch (constraint.kind) {
    case 'non_empty':
      return firstIssueMessage(v.safeParse(v.pipe(v.string(), v.nonEmpty(`${label} cannot be empty.`)), value.trim()))
    case 'positive': {
      const parsed = numericValue(value)
      return firstIssueMessage(
        v.safeParse(v.pipe(v.number(`${label} must be a number.`), v.minValue(1, `${label} must be positive.`)), parsed)
      )
    }
    case 'range': {
      const parsed = numericValue(value)
      const min = numericConstraintValue(constraint.min)
      const max = numericConstraintValue(constraint.max)
      return firstIssueMessage(
        v.safeParse(
          v.pipe(
            v.number(`${label} must be a number.`),
            v.check((input) => min === undefined || input >= min, `${label} must be at least ${min}.`),
            v.check((input) => max === undefined || input <= max, `${label} must be at most ${max}.`)
          ),
          parsed
        )
      )
    }
    case 'allowed_values':
      return firstIssueMessage(
        v.safeParse(
          v.pipe(
            v.string(),
            v.check(
              (input) => constraint.values.map(normalizedChoiceValue).includes(normalizedChoiceValue(input)),
              `${label} must be one of: ${constraint.values.join(', ')}.`
            )
          ),
          value
        )
      )
    case 'requires':
      return undefined
  }
}

export function validateConfigurationSettingValue(
  setting: ConfigurationDefaultsSetting,
  value: string
): SchemaFieldValidationResult {
  const requiresValue = (setting.validationConstraints ?? []).some((constraint) => constraint.kind === 'non_empty')
  if (!requiresValue && value.trim().length === 0) return { valid: true }

  const schemaMessage = setting.valueSchema ? validateSchemaKind(value, setting, setting.valueSchema) : undefined
  if (schemaMessage) return { valid: false, message: schemaMessage }

  const acceptedValues = acceptedValuesForSetting(setting)
  if (acceptedValues.length > 0) {
    const acceptedMessage = validateConstraint(value, setting, { kind: 'allowed_values', values: acceptedValues })
    if (acceptedMessage) return { valid: false, message: acceptedMessage }
  }

  for (const constraint of setting.validationConstraints ?? []) {
    const constraintMessage = validateConstraint(value, setting, constraint)
    if (constraintMessage) return { valid: false, message: constraintMessage }
  }

  return { valid: true }
}

import * as v from 'valibot'
import type {
  ConfigurationDefaultsSetting,
  ConfigurationSettingValueSchema,
  ConfigurationSettingValidationConstraint
} from '@/features/app-tabs/types'
import { acceptedValuesForSetting, hasSchemaKind, numericMetadataForSetting } from './schema-control-utils'

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

function validateNumber(value: string, setting: ConfigurationDefaultsSetting, integer: boolean): string | undefined {
  if (value.trim().length === 0) return undefined
  const parsed = numericValue(value)
  if (parsed === undefined) {
    return `${setting.label} must be a number.`
  }
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

/**
 * Returns true when `value` parses as a URL with a well-formed hostname.
 * Rejects incomplete numeric hosts like `http://21.131.41:1311` that the
 * browser URL constructor silently pads to `21.131.0.41`.
 */
function isValidUrl(value: string): boolean {
  let parsed: URL
  try {
    parsed = new URL(value)
  } catch {
    return false
  }
  const host = parsed.hostname

  // IPv6 in brackets — always valid when the URL constructor accepted it.
  if (host.startsWith('[') && host.endsWith(']')) return true

  // All-numeric host. Reject unless every octet is 0-255 and there are exactly 4.
  if (/^\d+(\.\d+)*$/.test(host)) {
    const octets = host.split('.')
    if (octets.length !== 4) return false
    if (
      !octets.every((o) => {
        const n = Number(o)
        return Number.isInteger(n) && n >= 0 && n <= 255
      })
    )
      return false
    // The URL constructor normalizes incomplete numeric hosts (e.g. 21.131.41 → 21.131.0.41).
    // Compare the reconstructed host against the original authority to catch this.
    const originalAuthority = value.match(/^https?:\/\/([^/]+)/)?.[1] ?? ''
    const originalHost = originalAuthority.replace(/:\d+$/, '')
    if (originalHost !== host) return false
    return true
  }

  // Port must be in the valid TCP/UDP range 1–65535. Port 0 is rejected (not a
  // usable endpoint). The URL constructor silently accepts out-of-range and
  // negative ports, so we must check explicitly.
  if (parsed.port !== '' && (Number(parsed.port) < 1 || Number(parsed.port) > 65535)) return false

  return true
}

function validateUrl(value: string, label: string): string | undefined {
  if (value.trim().length === 0) return undefined
  if (!isValidUrl(value)) return `${label} must be a valid URL.`
  return undefined
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

function validateAllowedPattern(value: string, pattern: string, label: string): string | undefined {
  if (value.trim().length === 0) return undefined

  let compiled: RegExp
  try {
    compiled = new RegExp(pattern)
  } catch {
    return `${label} has an invalid validation pattern.`
  }

  return compiled.test(value) ? undefined : `${label} has an invalid format.`
}

function validateSchemaKind(
  value: string,
  setting: ConfigurationDefaultsSetting,
  schema: ConfigurationSettingValueSchema
): string | undefined {
  const label = setting.label

  // Skip schema-level validation for empty values — only validate format when a value is provided.
  if (value.trim().length === 0) return undefined

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
      return validateUrl(value, label)
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
      // Empty fields are allowed — validation applies only when a value is provided.
      return undefined
    case 'positive': {
      const parsed = numericValue(value)
      if (parsed === undefined) return undefined
      return firstIssueMessage(
        v.safeParse(v.pipe(v.number(`${label} must be a number.`), v.minValue(1, `${label} must be positive.`)), parsed)
      )
    }
    case 'range': {
      const parsed = numericValue(value)
      if (parsed === undefined) return undefined
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
      if (value.trim().length === 0) return undefined
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
    case 'allowed_pattern':
      return validateAllowedPattern(value, constraint.pattern, label)
    case 'requires':
      return undefined
  }
}

function shouldValidateAcceptedValues(setting: ConfigurationDefaultsSetting, value: string): boolean {
  const schema = setting.valueSchema
  if (!schema) return true

  const acceptsNumeric = hasSchemaKind(schema, 'integer') || hasSchemaKind(schema, 'float')
  if (acceptsNumeric && numericValue(value) !== undefined) return false

  const acceptsBoolean = hasSchemaKind(schema, 'boolean')
  if (acceptsBoolean && ['on', 'off', 'true', 'false'].includes(value)) return false

  return true
}

export function validateConfigurationSettingValue(
  setting: ConfigurationDefaultsSetting,
  value: string
): SchemaFieldValidationResult {
  if (value.trim().length === 0) return { valid: true }

  const schemaMessage = setting.valueSchema ? validateSchemaKind(value, setting, setting.valueSchema) : undefined
  if (schemaMessage) return { valid: false, message: schemaMessage }

  const acceptedValues = acceptedValuesForSetting(setting)
  if (acceptedValues.length > 0 && shouldValidateAcceptedValues(setting, value)) {
    const acceptedMessage = validateConstraint(value, setting, { kind: 'allowed_values', values: acceptedValues })
    if (acceptedMessage) return { valid: false, message: acceptedMessage }
  }

  for (const constraint of setting.validationConstraints ?? []) {
    const constraintMessage = validateConstraint(value, setting, constraint)
    if (constraintMessage) return { valid: false, message: constraintMessage }
  }

  return { valid: true }
}

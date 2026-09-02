import type { ConfigurationSettingObjectSchema, ConfigurationSettingValueSchema } from '@/features/app-tabs/types'

export function isSchemaRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

export function objectArrayItemSchema(
  schema: ConfigurationSettingValueSchema | undefined
): ConfigurationSettingObjectSchema | undefined {
  if (schema?.kind !== 'array' || schema.items.kind !== 'object') return undefined
  return schema.items
}

export function parseSchemaObjectArrayValue(value: string): readonly Record<string, unknown>[] | undefined {
  if (value.trim().length === 0) return []

  try {
    const parsed: unknown = JSON.parse(value)
    if (!Array.isArray(parsed) || !parsed.every(isSchemaRecord)) return undefined
    return parsed
  } catch (error) {
    if (error instanceof SyntaxError) return undefined
    throw error
  }
}

function requiredSchemaDefault(schema: ConfigurationSettingValueSchema): unknown {
  switch (schema.kind) {
    case 'boolean':
      return false
    case 'integer':
    case 'float':
      return 0
    case 'array':
      return []
    case 'object':
      return createSchemaObjectDefault(schema)
    case 'enum':
      return schema.values[0] ?? ''
    case 'one_of': {
      const firstVariant = schema.variants[0]
      return firstVariant ? requiredSchemaDefault(firstVariant) : ''
    }
    case 'path':
    case 'socket_addr':
    case 'string':
    case 'url':
      return ''
  }
}

export function createSchemaObjectDefault(schema: ConfigurationSettingObjectSchema): Record<string, unknown> {
  const value: Record<string, unknown> = {}
  for (const property of schema.properties ?? []) {
    if (property.required) value[property.name] = requiredSchemaDefault(property.value_schema)
  }
  return value
}

export function schemaValueFromInput(schema: ConfigurationSettingValueSchema, value: string): unknown {
  switch (schema.kind) {
    case 'boolean':
      return value === 'true' || value === 'on'
    case 'integer': {
      const parsed = Number(value)
      return value.trim().length > 0 && Number.isInteger(parsed) ? parsed : value
    }
    case 'float': {
      const parsed = Number(value)
      return value.trim().length > 0 && Number.isFinite(parsed) ? parsed : value
    }
    case 'one_of': {
      const numericVariant = schema.variants.find((variant) => variant.kind === 'integer' || variant.kind === 'float')
      if (numericVariant && value.trim().length > 0 && Number.isFinite(Number(value))) {
        return schemaValueFromInput(numericVariant, value)
      }
      const booleanVariant = schema.variants.find((variant) => variant.kind === 'boolean')
      if (booleanVariant && ['true', 'false', 'on', 'off'].includes(value)) {
        return schemaValueFromInput(booleanVariant, value)
      }
      return value
    }
    case 'array':
    case 'object':
      return value
    case 'enum':
    case 'path':
    case 'socket_addr':
    case 'string':
    case 'url':
      return value
  }
}

export function schemaInputValue(value: unknown): string {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

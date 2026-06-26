import { describe, expect, it } from 'vitest'
import type { ConfigurationDefaultsSetting, ConfigurationSettingValidationConstraint } from '@/features/app-tabs/types'
import { validateConfigurationSettingValue } from '@/features/configuration/components/settings/schema-field-validation'

function makeSetting(
  overrides: Partial<Pick<ConfigurationDefaultsSetting, 'valueSchema' | 'validationConstraints' | 'label'>> &
    Pick<ConfigurationDefaultsSetting, 'label'>
): ConfigurationDefaultsSetting {
  return {
    id: 'test-setting',
    categoryId: 'runtime',
    icon: 'cog',
    description: 'Test setting',
    inheritedLabel: overrides.label,
    canonicalPath: 'test.setting',
    control: { kind: 'text', name: 'test-setting', value: '', placeholder: '' },
    valueSchema: overrides.valueSchema,
    validationConstraints: overrides.validationConstraints,
    ...overrides
  }
}

describe('validateConfigurationSettingValue', () => {
  describe('empty values always pass validation', () => {
    it('passes for empty string with boolean schema', () => {
      const setting = makeSetting({ label: 'Bool Field', valueSchema: { kind: 'boolean' } })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for whitespace-only string with boolean schema', () => {
      const setting = makeSetting({ label: 'Bool Field', valueSchema: { kind: 'boolean' } })
      expect(validateConfigurationSettingValue(setting, '   ')).toEqual({ valid: true })
    })

    it('passes for empty string with integer schema', () => {
      const setting = makeSetting({ label: 'Int Field', valueSchema: { kind: 'integer' } })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for empty string with float schema', () => {
      const setting = makeSetting({ label: 'Float Field', valueSchema: { kind: 'float' } })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for empty string with enum schema', () => {
      const setting = makeSetting({ label: 'Enum Field', valueSchema: { kind: 'enum', values: ['a', 'b'] } })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for empty string with url schema', () => {
      const setting = makeSetting({ label: 'URL Field', valueSchema: { kind: 'url' } })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for empty string with string schema', () => {
      const setting = makeSetting({ label: 'String Field', valueSchema: { kind: 'string' } })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for empty string with path schema', () => {
      const setting = makeSetting({ label: 'Path Field', valueSchema: { kind: 'path' } })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for empty string with object schema', () => {
      const setting = makeSetting({ label: 'Object Field', valueSchema: { kind: 'object' } })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for empty string with array schema', () => {
      const setting = makeSetting({ label: 'Array Field', valueSchema: { kind: 'array', items: { kind: 'string' } } })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for empty string with one_of schema', () => {
      const setting = makeSetting({
        label: 'OneOf Field',
        valueSchema: { kind: 'one_of', variants: [{ kind: 'integer' }, { kind: 'string' }] }
      })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })
  })

  describe('non_empty constraint does not error on empty values', () => {
    it('passes for empty string when non_empty constraint is present', () => {
      const setting = makeSetting({
        label: 'Required Field',
        valueSchema: { kind: 'string' },
        validationConstraints: [{ kind: 'non_empty' }] as readonly ConfigurationSettingValidationConstraint[]
      })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for whitespace-only string when non_empty constraint is present', () => {
      const setting = makeSetting({
        label: 'Required Field',
        valueSchema: { kind: 'string' },
        validationConstraints: [{ kind: 'non_empty' }] as readonly ConfigurationSettingValidationConstraint[]
      })
      expect(validateConfigurationSettingValue(setting, '   ')).toEqual({ valid: true })
    })
  })

  describe('format validation applies when a value is provided', () => {
    it('rejects non-numeric value for integer schema', () => {
      const setting = makeSetting({ label: 'Cores', valueSchema: { kind: 'integer' } })
      const result = validateConfigurationSettingValue(setting, 'abc')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be a number')
    })

    it('rejects non-integer value for integer schema', () => {
      const setting = makeSetting({ label: 'Cores', valueSchema: { kind: 'integer' } })
      const result = validateConfigurationSettingValue(setting, '3.5')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('whole number')
    })

    it('accepts valid integer value for integer schema', () => {
      const setting = makeSetting({ label: 'Cores', valueSchema: { kind: 'integer' } })
      expect(validateConfigurationSettingValue(setting, '4')).toEqual({ valid: true })
    })

    it('accepts valid float value for float schema', () => {
      const setting = makeSetting({ label: 'Temperature', valueSchema: { kind: 'float' } })
      expect(validateConfigurationSettingValue(setting, '0.7')).toEqual({ valid: true })
    })

    it('rejects invalid boolean value', () => {
      const setting = makeSetting({ label: 'Enable', valueSchema: { kind: 'boolean' } })
      const result = validateConfigurationSettingValue(setting, 'maybe')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be on, off, or auto')
    })

    it('accepts valid boolean value "on"', () => {
      const setting = makeSetting({ label: 'Enable', valueSchema: { kind: 'boolean' } })
      expect(validateConfigurationSettingValue(setting, 'on')).toEqual({ valid: true })
    })

    it('accepts bool-or-auto segmented values', () => {
      const setting = makeSetting({
        label: 'Continuous batching',
        valueSchema: {
          kind: 'one_of',
          variants: [{ kind: 'boolean' }, { kind: 'enum', values: ['auto'] }]
        }
      })

      expect(validateConfigurationSettingValue(setting, 'on')).toEqual({ valid: true })
      expect(validateConfigurationSettingValue(setting, 'off')).toEqual({ valid: true })
      expect(validateConfigurationSettingValue(setting, 'auto')).toEqual({ valid: true })
    })

    it('accepts numeric values for integer-or-auto settings', () => {
      const setting = makeSetting({
        label: 'GPU layers',
        valueSchema: {
          kind: 'one_of',
          variants: [{ kind: 'integer' }, { kind: 'enum', values: ['auto'] }]
        }
      })

      expect(validateConfigurationSettingValue(setting, '-1')).toEqual({ valid: true })
      expect(validateConfigurationSettingValue(setting, '12')).toEqual({ valid: true })
      expect(validateConfigurationSettingValue(setting, 'auto')).toEqual({ valid: true })
    })

    it('rejects invalid enum value', () => {
      const setting = makeSetting({ label: 'Mode', valueSchema: { kind: 'enum', values: ['fast', 'slow'] } })
      const result = validateConfigurationSettingValue(setting, 'medium')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be one of')
    })

    it('accepts valid enum value', () => {
      const setting = makeSetting({ label: 'Mode', valueSchema: { kind: 'enum', values: ['fast', 'slow'] } })
      expect(validateConfigurationSettingValue(setting, 'fast')).toEqual({ valid: true })
    })

    it('rejects invalid URL value', () => {
      const setting = makeSetting({ label: 'Endpoint', valueSchema: { kind: 'url' } })
      const result = validateConfigurationSettingValue(setting, 'not-a-url')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be a valid URL')
    })

    it('accepts valid URL value', () => {
      const setting = makeSetting({ label: 'Endpoint', valueSchema: { kind: 'url' } })
      expect(validateConfigurationSettingValue(setting, 'https://example.com')).toEqual({ valid: true })
    })

    it('rejects incomplete IP address like http://21.131.41:1311', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      const result = validateConfigurationSettingValue(setting, 'http://21.131.41:1311')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be a valid URL')
    })

    it('rejects three-octet incomplete IP like http://1.2.3:8080', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      const result = validateConfigurationSettingValue(setting, 'http://1.2.3:8080')
      expect(result.valid).toBe(false)
    })

    it('accepts valid IPv4 URL like http://192.168.1.1:1311', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      expect(validateConfigurationSettingValue(setting, 'http://192.168.1.1:1311')).toEqual({ valid: true })
    })

    it('accepts valid IPv6 URL like http://[::1]:1311', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      expect(validateConfigurationSettingValue(setting, 'http://[::1]:1311')).toEqual({ valid: true })
    })

    it('accepts localhost URL', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      expect(validateConfigurationSettingValue(setting, 'http://localhost:4317')).toEqual({ valid: true })
    })

    it('rejects IP with out-of-range octet like http://999.1.1.1:8080', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      const result = validateConfigurationSettingValue(setting, 'http://999.1.1.1:8080')
      expect(result.valid).toBe(false)
    })

    it('rejects out-of-range port like http://121.212.231.132:65536', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      const result = validateConfigurationSettingValue(setting, 'http://121.212.231.132:65536')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be a valid URL')
    })

    it('accepts valid max port like http://localhost:65535', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      expect(validateConfigurationSettingValue(setting, 'http://localhost:65535')).toEqual({ valid: true })
    })

    it('rejects port 0 like http://localhost:0', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      const result = validateConfigurationSettingValue(setting, 'http://localhost:0')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be a valid URL')
    })

    it('rejects negative port like http://localhost:-1', () => {
      const setting = makeSetting({ label: 'OTLP Endpoint', valueSchema: { kind: 'url' } })
      const result = validateConfigurationSettingValue(setting, 'http://localhost:-1')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be a valid URL')
    })

    it('rejects invalid JSON object value', () => {
      const setting = makeSetting({ label: 'Config', valueSchema: { kind: 'object' } })
      const result = validateConfigurationSettingValue(setting, '[1,2]')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be a JSON object')
    })

    it('accepts valid JSON object value', () => {
      const setting = makeSetting({ label: 'Config', valueSchema: { kind: 'object' } })
      expect(validateConfigurationSettingValue(setting, '{"key":"value"}')).toEqual({ valid: true })
    })
  })

  describe('numeric constraints skip for empty values', () => {
    it('passes for empty string with positive constraint', () => {
      const setting = makeSetting({
        label: 'Port',
        valueSchema: { kind: 'integer' },
        validationConstraints: [{ kind: 'positive' }] as readonly ConfigurationSettingValidationConstraint[]
      })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('passes for empty string with range constraint', () => {
      const setting = makeSetting({
        label: 'Slots',
        valueSchema: { kind: 'integer' },
        validationConstraints: [
          { kind: 'range', min: '1', max: '16' }
        ] as readonly ConfigurationSettingValidationConstraint[]
      })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('accepts positive value with positive constraint', () => {
      const setting = makeSetting({
        label: 'Port',
        valueSchema: { kind: 'integer' },
        validationConstraints: [{ kind: 'positive' }] as readonly ConfigurationSettingValidationConstraint[]
      })
      expect(validateConfigurationSettingValue(setting, '5')).toEqual({ valid: true })
    })

    it('rejects zero with positive constraint', () => {
      const setting = makeSetting({
        label: 'Port',
        valueSchema: { kind: 'integer' },
        validationConstraints: [{ kind: 'positive' }] as readonly ConfigurationSettingValidationConstraint[]
      })
      const result = validateConfigurationSettingValue(setting, '0')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be positive')
    })

    it('rejects out-of-range value with range constraint', () => {
      const setting = makeSetting({
        label: 'Slots',
        valueSchema: { kind: 'integer' },
        validationConstraints: [
          { kind: 'range', min: '1', max: '16' }
        ] as readonly ConfigurationSettingValidationConstraint[]
      })
      const result = validateConfigurationSettingValue(setting, '20')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be at most')
    })
  })

  describe('allowed_values constraint skips for empty values', () => {
    it('passes for empty string with allowed_values constraint', () => {
      const setting = makeSetting({
        label: 'Backend',
        validationConstraints: [
          { kind: 'allowed_values', values: ['cuda', 'vulkan', 'cpu'] }
        ] as readonly ConfigurationSettingValidationConstraint[]
      })
      expect(validateConfigurationSettingValue(setting, '')).toEqual({ valid: true })
    })

    it('rejects invalid value with allowed_values constraint', () => {
      const setting = makeSetting({
        label: 'Backend',
        validationConstraints: [
          { kind: 'allowed_values', values: ['cuda', 'vulkan', 'cpu'] }
        ] as readonly ConfigurationSettingValidationConstraint[]
      })
      const result = validateConfigurationSettingValue(setting, 'rocm')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('must be one of')
    })

    it('accepts valid value with allowed_values constraint', () => {
      const setting = makeSetting({
        label: 'Backend',
        validationConstraints: [
          { kind: 'allowed_values', values: ['cuda', 'vulkan', 'cpu'] }
        ] as readonly ConfigurationSettingValidationConstraint[]
      })
      expect(validateConfigurationSettingValue(setting, 'cuda')).toEqual({ valid: true })
    })
  })

  describe('allowed_pattern constraint', () => {
    it('accepts valid service-name values', () => {
      const setting = makeSetting({
        label: 'Service name',
        valueSchema: { kind: 'string' },
        validationConstraints: [{ kind: 'allowed_pattern', pattern: '^[A-Za-z0-9_-]+$' }]
      })

      expect(validateConfigurationSettingValue(setting, 'valid_service-name-123')).toEqual({ valid: true })
    })

    it('rejects invalid service-name values', () => {
      const setting = makeSetting({
        label: 'Service name',
        valueSchema: { kind: 'string' },
        validationConstraints: [{ kind: 'allowed_pattern', pattern: '^[A-Za-z0-9_-]+$' }]
      })

      const result = validateConfigurationSettingValue(setting, '@@*(!111---aa')
      expect(result.valid).toBe(false)
      if (!result.valid) expect(result.message).toContain('invalid format')
    })
  })
})

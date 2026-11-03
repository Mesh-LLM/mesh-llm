import { describe, expect, it } from 'vitest'
import { createConfigurationAttestationSettingsFromSchema } from './config-adapter'
import { BACKEND_SCHEMA_REFERENCE } from './config-adapter-test-support'
import { validateConfigurationSettingValue } from '../components/settings/schema-field-validation'

describe('signed gossip creation policy', () => {
  it('uses the backend boolean control and preserves the new-mesh restriction', () => {
    const settings = createConfigurationAttestationSettingsFromSchema(BACKEND_SCHEMA_REFERENCE)
    const policy = settings.settings.find((setting) => setting.id === 'mesh_requirements.require_signed_gossip')

    expect(policy).toMatchObject({
      label: 'Require signed peer gossip',
      valueSchema: { kind: 'boolean' },
      mutability: 'restart-required',
      applyMode: 'static_on_load',
      restartScope: 'process_restart',
      control: { kind: 'choice', name: 'require_signed_gossip' }
    })
    expect(policy?.description).toContain('newly created mesh')
    expect(policy?.description).toContain('older versions cannot join')
    expect(validateConfigurationSettingValue(policy!, 'on')).toEqual({ valid: true })
    expect(validateConfigurationSettingValue(policy!, 'off')).toEqual({ valid: true })
    expect(validateConfigurationSettingValue(policy!, 'enabled').valid).toBe(false)
  })
})

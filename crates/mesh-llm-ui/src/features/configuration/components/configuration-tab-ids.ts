export type ConfigurationTabId =
  | 'general'
  | 'runtime'
  | 'models'
  | 'network'
  | 'local-deployment'
  | 'wake-policy'
  | 'signing'
  | 'plugins'
  | 'toml-review'
  | 'audit'

export const CONFIGURATION_TAB_IDS = [
  'general',
  'runtime',
  'models',
  'network',
  'local-deployment',
  'wake-policy',
  'signing',
  'plugins',
  'toml-review',
  'audit'
] as const satisfies readonly ConfigurationTabId[]

type ConfigurationTabAvailability = {
  pluginsEnabled?: boolean
  signingAttestationEnabled?: boolean
  wakePolicyEnabled?: boolean
  logsSettingsEnabled?: boolean
}

export function getEnabledConfigurationTabIds({
  pluginsEnabled = true,
  signingAttestationEnabled = true,
  wakePolicyEnabled = true,
  logsSettingsEnabled = true
}: ConfigurationTabAvailability = {}): ConfigurationTabId[] {
  return CONFIGURATION_TAB_IDS.filter((tabId) => {
    if (tabId === 'wake-policy') return wakePolicyEnabled
    if (tabId === 'signing') return signingAttestationEnabled
    if (tabId === 'plugins') return pluginsEnabled
    if (tabId === 'audit') return logsSettingsEnabled
    return true
  })
}

export function isConfigurationTabId(
  value: string | undefined,
  availability?: ConfigurationTabAvailability
): value is ConfigurationTabId {
  return value !== undefined && getEnabledConfigurationTabIds(availability).some((tabId) => tabId === value)
}

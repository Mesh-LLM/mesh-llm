import { useMemo, type ReactNode } from 'react'
import { Blocks, Brackets, Computer, Cpu, FileText, Network, ShieldCheck, SlidersHorizontal } from 'lucide-react'
import { ConfigurationTabs, type ConfigurationTabItem } from '@/features/configuration/components/ConfigurationTabs'
import { ConfigurationWakePolicyTab } from '@/features/configuration/components/ConfigurationWakePolicyTab'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'
import { DefaultsTab } from '@/features/configuration/components/DefaultsTab'
import { PluginIntegrationsPanel } from '@/features/configuration/components/PluginIntegrationsPanel'
import { ConfigurationPlaceholderPanel } from '@/features/configuration/layouts/ConfigurationLayout'
import { TomlView } from '@/features/configuration/components/TomlView'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import type { ConfigurationState } from '@/features/configuration/hooks/useConfigurationHistory'
import type { ConfigurationDefaultsHarnessData, ConfigurationHarnessData } from '@/features/app-tabs/types'

type ConfigurationEditorTabsProps = {
  activeTab: ConfigurationTabId
  defaultsValues: Record<string, string>
  displayData: ConfigurationHarnessData
  hasUnsavedChanges: boolean
  liveMode: boolean
  localAssigns: ConfigurationState['assigns']
  localDeployment: ReactNode
  localNodes: ConfigurationState['nodes']
  localSavedConfiguration: ConfigurationState
  logsSettingsEnabled: boolean
  modelSettingsData: ConfigurationDefaultsHarnessData
  onResetSettings: (settingsData: ConfigurationDefaultsHarnessData | undefined) => void
  onSettingValueChange: (settingId: string, value: string) => void
  onTabChange: (tab: ConfigurationTabId) => void
  pluginsEnabled: boolean
  pluginsSettingsData: ConfigurationDefaultsHarnessData | undefined
  pluginsDirty: boolean
  runtimeControlNotice?: ReactNode
  signingAttestationEnabled: boolean
  wakePolicyConfigurationEnabled: boolean
  auditDirty: boolean
  attestationDirty: boolean
  localDeploymentDirty: boolean
  meshllmDirty: boolean
  modelSettingsDirty: boolean
  networkDirty: boolean
  runtimeDirty: boolean
}

function combineSettingsData(
  ...groups: readonly (ConfigurationDefaultsHarnessData | undefined)[]
): ConfigurationDefaultsHarnessData {
  const categoryById = new Map<string, ConfigurationDefaultsHarnessData['categories'][number]>()
  const settingById = new Map<string, ConfigurationDefaultsHarnessData['settings'][number]>()
  const preview = groups.flatMap((group) => group?.preview ?? [])

  for (const group of groups) {
    for (const category of group?.categories ?? []) categoryById.set(String(category.id), category)
    for (const setting of group?.settings ?? []) settingById.set(setting.id, setting)
  }

  return {
    categories: Array.from(categoryById.values()),
    settings: Array.from(settingById.values()),
    preview
  }
}

export function ConfigurationEditorTabs({
  activeTab,
  defaultsValues,
  displayData,
  hasUnsavedChanges,
  liveMode,
  localAssigns,
  localDeployment,
  localNodes,
  localSavedConfiguration,
  logsSettingsEnabled,
  modelSettingsData,
  onResetSettings: resetSettings,
  onSettingValueChange: updateDefaultSetting,
  onTabChange,
  pluginsEnabled,
  pluginsSettingsData,
  pluginsDirty,
  runtimeControlNotice,
  signingAttestationEnabled,
  wakePolicyConfigurationEnabled,
  auditDirty,
  attestationDirty,
  localDeploymentDirty,
  meshllmDirty,
  modelSettingsDirty,
  networkDirty,
  runtimeDirty
}: ConfigurationEditorTabsProps) {
  const tomlSettings = useMemo(
    () =>
      combineSettingsData(
        displayData.meshllm,
        displayData.audit,
        displayData.runtimeSettings,
        modelSettingsData,
        displayData.network,
        displayData.attestation,
        pluginsSettingsData
      ),
    [
      displayData.attestation,
      displayData.audit,
      displayData.meshllm,
      displayData.network,
      displayData.runtimeSettings,
      modelSettingsData,
      pluginsSettingsData
    ]
  )
  const previousToml = useMemo(
    () =>
      buildTOML(localSavedConfiguration.nodes, localSavedConfiguration.assigns, displayData.catalog, {
        defaults: tomlSettings,
        defaultsValues: localSavedConfiguration.defaultsValues,
        modelPlacementPaths: displayData.modelPlacementPaths,
        modelConfigEntries: displayData.modelConfigEntries
      }),
    [
      displayData.catalog,
      displayData.modelConfigEntries,
      displayData.modelPlacementPaths,
      localSavedConfiguration.assigns,
      localSavedConfiguration.defaultsValues,
      localSavedConfiguration.nodes,
      tomlSettings
    ]
  )
  const pluginIntegrationMetadata = <PluginIntegrationsPanel metadataEnabled={liveMode && activeTab === 'plugins'} />
  const pluginSettingsContent = pluginsSettingsData?.settings.length ? (
    <DefaultsTab
      data={pluginsSettingsData}
      values={defaultsValues}
      onResetAll={() => resetSettings(pluginsSettingsData)}
      onSettingValueChange={updateDefaultSetting}
      configFilePath={displayData.configFilePath}
      readOnlyNotice={runtimeControlNotice}
      previewTitle="[[plugin]]"
      screenLabel="Configuration · plugins"
      summaryDescription={
        <>
          Plugin settings are generated from installed plugin schemas and written under each matching{' '}
          <span className="rounded border border-border-soft bg-surface px-1 font-mono text-foreground">
            [[plugin]]
          </span>{' '}
          entry. Host-owned fields such as command and startup policy stay separate from plugin-owned custom settings.
        </>
      }
      summaryStatus={
        pluginsDirty
          ? `${pluginsSettingsData.settings.length} settings · modified`
          : `${pluginsSettingsData.settings.length} settings`
      }
      summaryTitle="Plugin settings"
      summaryTitleId="plugins-summary-heading"
      summarySupplement={pluginIntegrationMetadata}
      previewTip={
        <>Plugin manifests own these fields; update or reinstall the plugin when a setting is missing from this list.</>
      }
    />
  ) : (
    <div className="space-y-[14px]">
      {pluginIntegrationMetadata}
      <ConfigurationPlaceholderPanel title="Plugins" icon={Blocks}>
        Plugin settings will appear here when an installed plugin publishes config schema metadata.
      </ConfigurationPlaceholderPanel>
    </div>
  )

  const tabs: ConfigurationTabItem[] = [
    {
      id: 'general',
      label: 'General',
      icon: Cpu,
      dirty: meshllmDirty,
      content: displayData.meshllm?.settings.length ? (
        <DefaultsTab
          data={displayData.meshllm}
          values={defaultsValues}
          onResetAll={() => resetSettings(displayData.meshllm)}
          onSettingValueChange={updateDefaultSetting}
          configFilePath={displayData.configFilePath}
          readOnlyNotice={runtimeControlNotice}
          previewTitle="[runtime] / [telemetry]"
          screenLabel="Configuration · meshllm"
          summaryDescription={
            <>
              Local process settings written directly to{' '}
              <span className="rounded border border-border-soft bg-surface px-1 font-mono text-foreground">
                config.toml
              </span>
              . These are process-owned settings, not per-model placement defaults.
            </>
          }
          summaryTitle="General settings"
          summaryTitleId="meshllm-summary-heading"
        />
      ) : (
        <ConfigurationPlaceholderPanel title="General settings" icon={Cpu}>
          No writable general process settings are exposed by the current runtime schema.
        </ConfigurationPlaceholderPanel>
      )
    },
    {
      id: 'runtime',
      label: 'Runtime',
      icon: SlidersHorizontal,
      dirty: runtimeDirty,
      content: displayData.runtimeSettings?.settings.length ? (
        <DefaultsTab
          data={displayData.runtimeSettings}
          values={defaultsValues}
          onResetAll={() => resetSettings(displayData.runtimeSettings)}
          onSettingValueChange={updateDefaultSetting}
          configFilePath={displayData.configFilePath}
          readOnlyNotice={runtimeControlNotice}
          previewTitle="[runtime] / [defaults.*]"
          screenLabel="Configuration · runtime"
          summaryDescription={
            <>
              Startup and reconciliation settings that the local process reads from the config file. Native runtime
              installation and hardware selection are intentionally not presented as switchable UI controls here.
            </>
          }
          summaryTitle="Runtime settings"
          summaryTitleId="runtime-summary-heading"
        />
      ) : (
        <ConfigurationPlaceholderPanel title="Runtime settings" icon={SlidersHorizontal}>
          No writable runtime settings are exposed by the current runtime schema.
        </ConfigurationPlaceholderPanel>
      )
    },
    {
      id: 'models',
      label: 'Models',
      icon: Computer,
      dirty: modelSettingsDirty,
      content: modelSettingsData.settings.length ? (
        <DefaultsTab
          data={modelSettingsData}
          values={defaultsValues}
          onResetAll={() => resetSettings(modelSettingsData)}
          onSettingValueChange={updateDefaultSetting}
          configFilePath={displayData.configFilePath}
          readOnlyNotice={runtimeControlNotice}
          previewTitle="[gpu] / [defaults.*]"
          screenLabel="Configuration · models"
          summaryDescription={
            <>
              GPU placement policy and model defaults are inherited by new{' '}
              <span className="rounded border border-border-soft bg-surface px-1 font-mono text-foreground">
                [[models]]
              </span>{' '}
              entries and can be overridden by individual deployments.
            </>
          }
          summaryTitle="Model settings"
          summaryTitleId="models-summary-heading"
        />
      ) : (
        <ConfigurationPlaceholderPanel title="Model settings" icon={Computer}>
          No writable model defaults are exposed by the current runtime schema.
        </ConfigurationPlaceholderPanel>
      )
    },
    {
      id: 'network',
      label: 'Network',
      icon: Network,
      dirty: networkDirty,
      content: displayData.network?.settings.length ? (
        <DefaultsTab
          data={displayData.network}
          values={defaultsValues}
          onResetAll={() => resetSettings(displayData.network)}
          onSettingValueChange={updateDefaultSetting}
          configFilePath={displayData.configFilePath}
          readOnlyNotice={runtimeControlNotice}
          previewTitle="[owner_control]"
          screenLabel="Configuration · network"
          summaryDescription={
            <>
              Network settings cover owner-control binding and advertised control endpoints that are applied when the
              local process starts.
            </>
          }
          summaryTitle="Network settings"
          summaryTitleId="network-summary-heading"
        />
      ) : (
        <ConfigurationPlaceholderPanel title="Network settings" icon={Network}>
          No writable network settings are exposed by the current runtime schema.
        </ConfigurationPlaceholderPanel>
      )
    },
    ...(logsSettingsEnabled
      ? [
          {
            id: 'audit',
            label: 'Logs',
            icon: FileText,
            dirty: auditDirty,
            content: displayData.audit?.settings.length ? (
              <DefaultsTab
                data={displayData.audit}
                values={defaultsValues}
                onResetAll={() => resetSettings(displayData.audit)}
                onSettingValueChange={updateDefaultSetting}
                configFilePath={displayData.configFilePath}
                readOnlyNotice={runtimeControlNotice}
                previewTitle="[audit] / [logging]"
                screenLabel="Configuration · logs"
                summaryDescription={
                  <>
                    Configure request history, payload capture, delivery, and optional security audit output for this
                    node. Changes are saved to the local MeshLLM configuration file.
                  </>
                }
                summaryTitle="Logs"
                summaryTitleId="logs-summary-heading"
              />
            ) : (
              <ConfigurationPlaceholderPanel title="Logs" icon={FileText}>
                No writable logs settings are exposed by the current runtime schema.
              </ConfigurationPlaceholderPanel>
            )
          } satisfies ConfigurationTabItem
        ]
      : []),
    {
      id: 'local-deployment',
      label: 'Model Deployment',
      icon: Computer,
      dirty: localDeploymentDirty,
      content: localDeployment
    },
    ...(wakePolicyConfigurationEnabled
      ? [
          {
            id: 'wake-policy',
            label: 'Reserves',
            icon: SlidersHorizontal,
            content: <ConfigurationWakePolicyTab />
          } satisfies ConfigurationTabItem
        ]
      : []),
    ...(signingAttestationEnabled
      ? [
          {
            id: 'signing',
            label: 'Signing / Attestation',
            icon: ShieldCheck,
            dirty: attestationDirty,
            content: displayData.attestation?.settings.length ? (
              <DefaultsTab
                data={displayData.attestation}
                values={defaultsValues}
                onResetAll={() => resetSettings(displayData.attestation)}
                onSettingValueChange={updateDefaultSetting}
                configFilePath={displayData.configFilePath}
                readOnlyNotice={runtimeControlNotice}
                previewTitle="[mesh_requirements]"
                screenLabel="Configuration · signing"
                summaryDescription={
                  <>
                    Attestation settings define certified-build admission requirements for meshes. They are written to{' '}
                    <span className="rounded border border-border-soft bg-surface px-1 font-mono text-foreground">
                      [mesh_requirements]
                    </span>{' '}
                    and enforced from the loaded config.
                  </>
                }
                summaryStatus={attestationDirty ? 'modified' : 'ready'}
                summaryTitle="Signing / Attestation"
                summaryTitleId="attestation-summary-heading"
                previewTip={
                  <>
                    These controls describe required build provenance. They do not claim remote hardware or native
                    runtime integrity beyond the attestation data the node can verify.
                  </>
                }
              />
            ) : (
              <ConfigurationPlaceholderPanel title="Signing / Attestation" icon={ShieldCheck}>
                No writable attestation settings are exposed by the current runtime schema.
              </ConfigurationPlaceholderPanel>
            )
          } satisfies ConfigurationTabItem
        ]
      : []),
    ...(pluginsEnabled
      ? [
          {
            id: 'plugins',
            label: 'Plugins',
            icon: Blocks,
            dirty: pluginsDirty,
            content: pluginSettingsContent
          } satisfies ConfigurationTabItem
        ]
      : []),
    {
      id: 'toml-review',
      label: 'TOML Output',
      icon: Brackets,
      dirty: hasUnsavedChanges,
      content: (
        <TomlView
          nodes={localNodes}
          assigns={localAssigns}
          models={displayData.catalog}
          defaults={tomlSettings}
          defaultsValues={defaultsValues}
          previousToml={previousToml}
          modelPlacementPaths={displayData.modelPlacementPaths}
          modelConfigEntries={displayData.modelConfigEntries}
          reviewMode
          validationEnabled={liveMode && activeTab === 'toml-review'}
          configPath={displayData.configFilePath}
          validationWarnings={displayData.validationWarnings}
          launchSummaryConfig={displayData.launchSummaryConfig}
        />
      )
    }
  ]
  const renderedActiveTab = tabs.some((tab) => tab.id === activeTab) ? activeTab : 'general'

  return <ConfigurationTabs value={renderedActiveTab} onValueChange={onTabChange} tabs={tabs} />
}

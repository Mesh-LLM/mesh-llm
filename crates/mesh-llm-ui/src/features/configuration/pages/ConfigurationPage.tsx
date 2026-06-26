import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, Blocks, Brackets, Computer, Cpu, Network, ShieldCheck, SlidersHorizontal } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { CatalogPopover } from '@/features/configuration/components/CatalogPopover'
import { ConfigurationHeader } from '@/features/configuration/components/ConfigurationHeader'
import { ConfigurationLiveDataBoundary } from '@/features/configuration/components/ConfigurationLiveDataBoundary'
import { ConfigurationReadOnlyNodesDivider } from '@/features/configuration/components/ConfigurationReadOnlyNodesDivider'
import { ConfigurationRuntimeControlBanner } from '@/features/configuration/components/ConfigurationRuntimeControlBanner'
import {
  formatRuntimeControlDisabledReason,
  formatRuntimeControlDisabledSaveError
} from '@/features/configuration/components/runtime-control-copy'
import { ConfigurationTabs, type ConfigurationTabItem } from '@/features/configuration/components/ConfigurationTabs'
import { ConfigurationWakePolicyTab } from '@/features/configuration/components/ConfigurationWakePolicyTab'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'
import { jumpCtxPower, stepCtx } from '@/features/configuration/components/ctx-slider-utils'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import { DefaultsTab } from '@/features/configuration/components/DefaultsTab'
import { NodeRail } from '@/features/configuration/components/NodeRail'
import { NodeSection } from '@/features/configuration/components/NodeSection'
import { TomlView } from '@/features/configuration/components/TomlView'
import { createDefaultsValues } from '@/features/configuration/hooks/useDefaultsSettingsState'
import {
  cloneConfigurationState,
  createConfigurationSnapshot,
  createInitialConfigurationState,
  hasInvalidAllocation,
  type ConfigurationState,
  useConfigurationHistory
} from '@/features/configuration/hooks/useConfigurationHistory'
import {
  ConfigurationDeploymentLayout,
  ConfigurationLayout,
  ConfigurationPlaceholderPanel
} from '@/features/configuration/layouts/ConfigurationLayout'
import { hasConfigurablePlacement } from '@/features/configuration/lib/config-math'
import {
  getNodeTargetContainerIdx,
  createSeparatePlacementSnapshot,
  restoreSeparatePlacement
} from '@/features/configuration/pages/ConfigurationPage.helpers'
import { KeyboardLegend } from '@/features/configuration/pages/ConfigurationPageKeyboardLegend'
import { UnsavedConfigurationNavigationBlocker } from '@/features/configuration/pages/ConfigurationPageNavigationBlocker'
import { useConfigurationPageSelection } from '@/features/configuration/pages/useConfigurationPageSelection'
import { useConfigurationPageKeyboardShortcuts } from '@/features/configuration/pages/useConfigurationPageKeyboardShortcuts'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { ConfigurationDefaultsHarnessData, ConfigurationHarnessData, Placement } from '@/features/app-tabs/types'
import { useConfigQuery } from '@/features/configuration/api/use-config-query'
import {
  runtimeControlApplyErrorMessage,
  type RuntimeControlApplyInput,
  type RuntimeControlApplyResponse,
  type RuntimeControlBootstrapPayload
} from '@/features/configuration/api/config-adapter'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'

type ConfigurationPageProps = {
  activeTab?: ConfigurationTabId
  enableNavigationBlocker?: boolean
  initialTab?: ConfigurationTabId
  onTabChange?: (tab: ConfigurationTabId) => void
}

type ConfigurationFixturePageProps = ConfigurationPageProps & {
  data?: ConfigurationHarnessData
}

type ConfigurationEditorPageProps = ConfigurationPageProps & {
  data: ConfigurationHarnessData
  liveMode: boolean
  runtimeControlBootstrap?: RuntimeControlBootstrapPayload
  runtimeControlConfigUnavailableReason?: string
  applyDefaults: (input: RuntimeControlApplyInput) => Promise<RuntimeControlApplyResponse | null>
}

const RUNTIME_CONTROL_SAVE_UNAVAILABLE_ERROR = 'Config was not saved. Runtime control config is unavailable.'

function valuesSnapshot(values: Record<string, string>, settingIds: readonly string[]) {
  return JSON.stringify(Object.fromEntries(settingIds.map((settingId) => [settingId, values[settingId] ?? null])))
}

function formatRuntimeControlFailure(error: unknown) {
  if (error instanceof Error && error.message.trim())
    return `Config was not saved. Runtime control failed: ${error.message}`
  return 'Config was not saved. Runtime control failed.'
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

export function ConfigurationPageContent({
  activeTab: controlledActiveTab,
  enableNavigationBlocker = true,
  initialTab = 'general',
  onTabChange
}: ConfigurationPageProps = {}) {
  const {
    data: liveData,
    isFetching,
    isError,
    isPending,
    modelsQuery,
    statusQuery,
    controlConfigQuery,
    applyDefaults
  } = useConfigQuery({ enabled: true })
  const runtimeControlBootstrap = controlConfigQuery.data?.bootstrap
  const runtimeControlDisabled = Boolean(runtimeControlBootstrap && !runtimeControlBootstrap.enabled)
  const runtimeControlConfigUnavailableReason =
    !runtimeControlDisabled && !controlConfigQuery.isFetching && !controlConfigQuery.data?.snapshot
      ? 'Runtime control config is unavailable'
      : undefined
  const livePluginSettingsData = liveData?.plugins ?? liveData?.integrations
  const retryLiveData = useCallback(() => {
    void Promise.all([statusQuery.refetch(), modelsQuery.refetch(), controlConfigQuery.refetch()])
  }, [controlConfigQuery, modelsQuery, statusQuery])
  const boundaryState = isError || (!isFetching && !isPending) ? 'error' : 'loading'

  if (!liveData) return <ConfigurationLiveDataBoundary state={boundaryState} onRetry={retryLiveData} />
  if (
    liveData.defaults.settings.length === 0 &&
    (liveData.meshllm?.settings.length ?? 0) === 0 &&
    (liveData.runtimeSettings?.settings.length ?? 0) === 0 &&
    (liveData.network?.settings.length ?? 0) === 0 &&
    (livePluginSettingsData?.settings.length ?? 0) === 0
  ) {
    return <ConfigurationLiveDataBoundary state="empty-schema" onRetry={retryLiveData} />
  }

  return (
    <ConfigurationEditorPage
      activeTab={controlledActiveTab}
      applyDefaults={applyDefaults}
      data={liveData}
      enableNavigationBlocker={enableNavigationBlocker}
      initialTab={initialTab}
      liveMode
      runtimeControlBootstrap={runtimeControlBootstrap}
      runtimeControlConfigUnavailableReason={runtimeControlConfigUnavailableReason}
      onTabChange={onTabChange}
    />
  )
}

export function ConfigurationFixturePage({
  data = CONFIGURATION_HARNESS,
  initialTab = 'models',
  ...props
}: ConfigurationFixturePageProps = {}) {
  return (
    <ConfigurationEditorPage
      {...props}
      applyDefaults={async () => null}
      data={data}
      initialTab={initialTab}
      liveMode={false}
    />
  )
}

function ConfigurationEditorPage({
  activeTab: controlledActiveTab,
  applyDefaults,
  data: displayData,
  enableNavigationBlocker = true,
  initialTab = 'general',
  liveMode,
  runtimeControlBootstrap,
  runtimeControlConfigUnavailableReason,
  onTabChange
}: ConfigurationEditorPageProps) {
  const signingAttestationEnabled = useBooleanFeatureFlag('configuration/signingAttestation')
  const pluginsEnabled = useBooleanFeatureFlag('configuration/integrations')
  const wakePolicyConfigurationEnabled = useBooleanFeatureFlag('configuration/wakePolicyConfiguration')
  const pluginsSettingsData = displayData.plugins ?? displayData.integrations
  const runtimeControlDisabled = liveMode && Boolean(runtimeControlBootstrap && !runtimeControlBootstrap.enabled)
  const runtimeControlDisabledReason = runtimeControlDisabled
    ? `Runtime control is disabled: ${formatRuntimeControlDisabledReason(runtimeControlBootstrap)}`
    : undefined
  const runtimeControlSaveDisabledReason = runtimeControlDisabledReason ?? runtimeControlConfigUnavailableReason

  const initialDefaultsValues = useMemo(
    () =>
      createDefaultsValues(
        displayData.defaults,
        displayData.meshllm,
        displayData.runtimeSettings,
        displayData.modelSettings,
        displayData.network,
        displayData.attestation,
        pluginsSettingsData
      ),
    [
      displayData.attestation,
      displayData.defaults,
      displayData.meshllm,
      displayData.modelSettings,
      displayData.network,
      displayData.runtimeSettings,
      pluginsSettingsData
    ]
  )
  const initialConfiguration = useMemo(
    () => createInitialConfigurationState(displayData.nodes, displayData.assigns, initialDefaultsValues),
    [displayData.assigns, displayData.nodes, initialDefaultsValues]
  )
  const configurationSourceKey = useMemo(
    () => createConfigurationSnapshot(displayData.nodes, displayData.assigns, initialDefaultsValues),
    [displayData.assigns, displayData.nodes, initialDefaultsValues]
  )
  const latestInitialConfigurationRef = useRef(initialConfiguration)
  useEffect(() => {
    latestInitialConfigurationRef.current = initialConfiguration
  }, [initialConfiguration])
  const {
    configuration,
    setAssigns,
    updateConfiguration,
    resetConfiguration,
    canUndo,
    canRedo,
    undoConfigurationChange,
    redoConfigurationChange
  } = useConfigurationHistory(initialConfiguration)
  const nodes = configuration.nodes
  const assigns = configuration.assigns
  const defaultsValues = configuration.defaultsValues
  const localNodeId = nodes[0]?.id ?? displayData.nodes[0]?.id ?? null
  const localNodes = useMemo(
    () => (localNodeId ? nodes.filter((node) => node.id === localNodeId) : []),
    [localNodeId, nodes]
  )
  const remoteNodes = useMemo(
    () => (localNodeId ? nodes.filter((node) => node.id !== localNodeId) : []),
    [localNodeId, nodes]
  )
  const localAssigns = useMemo(
    () => (localNodeId ? assigns.filter((assign) => assign.nodeId === localNodeId) : []),
    [assigns, localNodeId]
  )
  const [savedConfiguration, setSavedConfiguration] = useState<ConfigurationState>(() =>
    cloneConfigurationState(initialConfiguration)
  )
  const localInitialConfiguration = useMemo(
    () =>
      createInitialConfigurationState(
        initialConfiguration.nodes.filter((node) => node.id === localNodeId),
        initialConfiguration.assigns.filter((assign) => assign.nodeId === localNodeId),
        initialConfiguration.defaultsValues
      ),
    [initialConfiguration, localNodeId]
  )
  const localSavedConfiguration = useMemo(
    () =>
      createInitialConfigurationState(
        savedConfiguration.nodes.filter((node) => node.id === localNodeId),
        savedConfiguration.assigns.filter((assign) => assign.nodeId === localNodeId),
        savedConfiguration.defaultsValues
      ),
    [localNodeId, savedConfiguration]
  )
  const [activeTabState, setActiveTabState] = useState<ConfigurationTabId>(initialTab)
  const activeTab = controlledActiveTab ?? activeTabState
  const [collapsedMap, setCollapsedMap] = useState<Record<string, boolean>>({})
  const [isSavingConfiguration, setIsSavingConfiguration] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const appliedConfigurationSourceKeyRef = useRef(configurationSourceKey)
  const {
    selectedId,
    selectedNodeId,
    selectedContainerTarget,
    selectedAssignId,
    selectedAssign,
    catalogFor,
    catalogError,
    selectedCatalogNode,
    setNodeRef,
    setSelectedNodeId,
    closeCatalog,
    restorePreferredSelection,
    selectContainerTarget,
    openCatalogForNode,
    selectNodeByOffset,
    selectGpuSlotByOffset,
    selectModelInCurrentGpu,
    moveSelectedAssignByGpuOffset,
    removeAssignById,
    pickNodeAssignment,
    selectCatalogModel
  } = useConfigurationPageSelection({
    nodes: localNodes,
    assigns: localAssigns,
    models: displayData.catalog,
    initialConfiguration: localInitialConfiguration,
    preferredAssignId: displayData.preferredAssignId,
    setAssigns
  })

  const setNodePlacement = useCallback(
    (nodeId: string, placement: Placement) => {
      updateConfiguration((current) => {
        if (nodeId !== localNodeId) return current
        const node = current.nodes.find((item) => item.id === nodeId)
        if (!node || !hasConfigurablePlacement(node) || node.placement === placement) return current

        const separatePlacementSnapshot = current.separatePlacementSnapshots[nodeId] ?? {}
        const nextSeparatePlacementSnapshots =
          placement === 'pooled' && node.placement === 'separate'
            ? {
                ...current.separatePlacementSnapshots,
                [nodeId]: createSeparatePlacementSnapshot(current.assigns, nodeId)
              }
            : current.separatePlacementSnapshots
        const nextNodes = current.nodes.map((item) => (item.id === nodeId ? { ...item, placement } : item))
        const nextNode = nextNodes.find((item) => item.id === nodeId) ?? node
        const nextAssigns =
          placement === 'pooled'
            ? current.assigns.map((assign) => (assign.nodeId === nodeId ? { ...assign, containerIdx: 0 } : assign))
            : restoreSeparatePlacement(current.assigns, nextNode, separatePlacementSnapshot, displayData.catalog)

        return {
          nodes: nextNodes,
          assigns: nextAssigns,
          defaultsValues: current.defaultsValues,
          separatePlacementSnapshots: nextSeparatePlacementSnapshots
        }
      })
    },
    [displayData.catalog, localNodeId, updateConfiguration]
  )

  const hasInvalidNode = useMemo(
    () => hasInvalidAllocation(localNodes, localAssigns, displayData.catalog),
    [displayData.catalog, localAssigns, localNodes]
  )
  const currentSnapshot = useMemo(
    () => createConfigurationSnapshot(nodes, assigns, defaultsValues),
    [assigns, defaultsValues, nodes]
  )
  const savedSnapshot = useMemo(
    () =>
      createConfigurationSnapshot(
        savedConfiguration.nodes,
        savedConfiguration.assigns,
        savedConfiguration.defaultsValues
      ),
    [savedConfiguration]
  )
  const hasUnsavedChanges = currentSnapshot !== savedSnapshot
  useEffect(() => {
    if (appliedConfigurationSourceKeyRef.current === configurationSourceKey || hasUnsavedChanges) return

    const nextConfiguration = latestInitialConfigurationRef.current
    resetConfiguration(nextConfiguration)
    setSavedConfiguration(cloneConfigurationState(nextConfiguration))
    setSaveError(null)
    appliedConfigurationSourceKeyRef.current = configurationSourceKey
  }, [configurationSourceKey, hasUnsavedChanges, resetConfiguration])
  const saveAlertMessage = useMemo(() => {
    if (!hasUnsavedChanges || !saveError) return null
    if (saveError === RUNTIME_CONTROL_SAVE_UNAVAILABLE_ERROR) {
      return runtimeControlConfigUnavailableReason ? saveError : null
    }
    if (saveError.startsWith('Config was not saved. Runtime control is disabled:')) {
      return runtimeControlDisabledReason ? saveError : null
    }
    return saveError
  }, [hasUnsavedChanges, runtimeControlConfigUnavailableReason, runtimeControlDisabledReason, saveError])
  const settingsDirty = useCallback(
    (settingsData: { settings: readonly { id: string }[] } | undefined) => {
      const settingIds = settingsData?.settings.map((setting) => setting.id) ?? []
      return (
        settingIds.length > 0 &&
        valuesSnapshot(defaultsValues, settingIds) !== valuesSnapshot(savedConfiguration.defaultsValues, settingIds)
      )
    },
    [defaultsValues, savedConfiguration.defaultsValues]
  )
  const modelSettingsData = displayData.modelSettings ?? displayData.defaults
  const meshllmDirty = settingsDirty(displayData.meshllm)
  const runtimeDirty = settingsDirty(displayData.runtimeSettings)
  const modelSettingsDirty = settingsDirty(modelSettingsData)
  const networkDirty = settingsDirty(displayData.network)
  const attestationDirty = settingsDirty(displayData.attestation)
  const pluginsDirty = settingsDirty(pluginsSettingsData)
  const localDeploymentDirty = useMemo(
    () =>
      createConfigurationSnapshot(nodes, assigns, savedConfiguration.defaultsValues) !==
      createConfigurationSnapshot(
        savedConfiguration.nodes,
        savedConfiguration.assigns,
        savedConfiguration.defaultsValues
      ),
    [assigns, nodes, savedConfiguration]
  )

  const updateDefaultSetting = useCallback(
    (settingId: string, value: string) => {
      updateConfiguration((current) => ({
        ...current,
        defaultsValues: { ...current.defaultsValues, [settingId]: value }
      }))
    },
    [updateConfiguration]
  )

  const resetSettings = useCallback(
    (settingsData: { settings: readonly { id: string }[] } | undefined) => {
      const settingIds = new Set(settingsData?.settings.map((setting) => setting.id) ?? [])
      updateConfiguration((current) => ({
        ...current,
        defaultsValues: {
          ...current.defaultsValues,
          ...Object.fromEntries(
            Array.from(settingIds).map((settingId) => [settingId, initialDefaultsValues[settingId] ?? ''])
          )
        }
      }))
    },
    [initialDefaultsValues, updateConfiguration]
  )

  const stepSelectedContext = useCallback(
    (direction: -1 | 1, jumpToPower = false) => {
      if (!selectedAssign) return

      setAssigns((items) =>
        items.map((assign) =>
          assign.id === selectedAssign.id
            ? { ...assign, ctx: jumpToPower ? jumpCtxPower(assign.ctx, direction) : stepCtx(assign.ctx, direction) }
            : assign
        )
      )
    },
    [selectedAssign, setAssigns]
  )

  const revertConfiguration = useCallback(() => {
    const restoredConfiguration = cloneConfigurationState(savedConfiguration)

    resetConfiguration(restoredConfiguration)
    restorePreferredSelection(restoredConfiguration, displayData.preferredAssignId)
  }, [displayData.preferredAssignId, resetConfiguration, restorePreferredSelection, savedConfiguration])

  const saveConfiguration = useCallback(() => {
    if (isSavingConfiguration || hasInvalidNode || !hasUnsavedChanges) return

    setSaveError(null)

    if (runtimeControlDisabled) {
      setSaveError(formatRuntimeControlDisabledSaveError(runtimeControlBootstrap))
      return
    }

    if (runtimeControlConfigUnavailableReason) {
      setSaveError(RUNTIME_CONTROL_SAVE_UNAVAILABLE_ERROR)
      return
    }

    if (liveMode) {
      setIsSavingConfiguration(true)
      void applyDefaults({
        values: configuration.defaultsValues,
        nodes: localNodes.length > 0 ? localNodes : nodes,
        assigns: localAssigns,
        catalog: displayData.catalog,
        modelPlacementPaths: displayData.modelPlacementPaths
      })
        .then((response) => {
          if (!response?.success) {
            const message = runtimeControlApplyErrorMessage(response)
            setSaveError(
              message
                ? `Config was not saved. Runtime control rejected the update: ${message}`
                : RUNTIME_CONTROL_SAVE_UNAVAILABLE_ERROR
            )
            return
          }
          setSavedConfiguration(cloneConfigurationState(configuration))
        })
        .catch((error: unknown) => setSaveError(formatRuntimeControlFailure(error)))
        .finally(() => setIsSavingConfiguration(false))
      return
    }

    setSavedConfiguration(cloneConfigurationState(configuration))
  }, [
    applyDefaults,
    configuration,
    displayData.catalog,
    displayData.modelPlacementPaths,
    hasInvalidNode,
    hasUnsavedChanges,
    isSavingConfiguration,
    liveMode,
    localAssigns,
    localNodes,
    nodes,
    runtimeControlBootstrap,
    runtimeControlConfigUnavailableReason,
    runtimeControlDisabled
  ])

  const currentKeyboardNode = useMemo(
    () => localNodes.find((item) => item.id === (selectedNodeId ?? selectedAssign?.nodeId)) ?? null,
    [localNodes, selectedAssign, selectedNodeId]
  )

  const openCatalogForCurrentNode = useCallback(() => {
    if (currentKeyboardNode) openCatalogForNode(currentKeyboardNode)
  }, [currentKeyboardNode, openCatalogForNode])

  const setCurrentNodePlacement = useCallback(
    (placement: Placement) => {
      if (
        !currentKeyboardNode ||
        !hasConfigurablePlacement(currentKeyboardNode) ||
        currentKeyboardNode.placement === placement
      )
        return false

      setNodePlacement(currentKeyboardNode.id, placement)
      return true
    },
    [currentKeyboardNode, setNodePlacement]
  )
  const ignoreReadOnlyAction = useCallback(() => undefined, [])

  useConfigurationPageKeyboardShortcuts({
    canUndo,
    canRedo,
    selectedAssignId,
    saveConfiguration,
    revertConfiguration,
    undoConfigurationChange,
    redoConfigurationChange,
    selectNodeByOffset,
    selectGpuSlotByOffset,
    selectModelInCurrentGpu,
    moveSelectedAssignByGpuOffset,
    stepSelectedContext,
    openCatalogForCurrentNode,
    setCurrentNodePlacement,
    removeSelectedAssign: removeAssignById
  })

  const jump = (nodeId: string) =>
    document.getElementById(`node-${nodeId}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' })

  const keyboardHint: ReactNode = <KeyboardLegend />

  const rail = (
    <NodeRail
      nodes={nodes}
      assigns={assigns}
      models={displayData.catalog}
      collapsedMap={collapsedMap}
      setCollapsedMap={setCollapsedMap}
      onJump={jump}
      keyboardHint={keyboardHint}
    />
  )

  const localDeployment = (
    <ConfigurationDeploymentLayout rail={rail}>
      {localNodes.map((node) => (
        <div
          key={node.id}
          ref={(element) => {
            setNodeRef(node.id, element)
          }}
        >
          <NodeSection
            node={node}
            assigns={assigns}
            models={displayData.catalog}
            modelPlacementOptions={displayData.modelPlacementOptions}
            setAssigns={setAssigns}
            selectedId={selectedId}
            selectedContainerIdx={
              selectedContainerTarget?.nodeId === node.id ? selectedContainerTarget.containerIdx : null
            }
            selectedNode={selectedNodeId === node.id}
            onFocusNode={() => setSelectedNodeId(node.id)}
            onPick={(id) => pickNodeAssignment(node, id)}
            onSelectContainer={(containerIdx) =>
              selectContainerTarget(node.id, getNodeTargetContainerIdx(node, containerIdx))
            }
            collapsed={Boolean(collapsedMap[node.id])}
            setCollapsed={(collapsed) => setCollapsedMap((map) => ({ ...map, [node.id]: collapsed }))}
            onOpenCatalog={openCatalogForNode}
            onPlacementChange={setNodePlacement}
          />
        </div>
      ))}
      {remoteNodes.length > 0 ? <ConfigurationReadOnlyNodesDivider /> : null}
      {remoteNodes.map((node) => (
        <NodeSection
          key={node.id}
          node={node}
          assigns={assigns}
          models={displayData.catalog}
          modelPlacementOptions={displayData.modelPlacementOptions}
          setAssigns={setAssigns}
          selectedId={null}
          selectedContainerIdx={null}
          selectedNode={false}
          onFocusNode={ignoreReadOnlyAction}
          onPick={ignoreReadOnlyAction}
          onSelectContainer={ignoreReadOnlyAction}
          collapsed={Boolean(collapsedMap[node.id])}
          setCollapsed={(collapsed) => setCollapsedMap((map) => ({ ...map, [node.id]: collapsed }))}
          onOpenCatalog={ignoreReadOnlyAction}
          onPlacementChange={setNodePlacement}
          readOnly
        />
      ))}
    </ConfigurationDeploymentLayout>
  )
  const runtimeControlNotice =
    runtimeControlDisabled && runtimeControlBootstrap ? (
      <ConfigurationRuntimeControlBanner bootstrap={runtimeControlBootstrap} />
    ) : undefined
  const tomlSettings = useMemo(
    () =>
      combineSettingsData(
        displayData.meshllm,
        displayData.runtimeSettings,
        modelSettingsData,
        displayData.network,
        displayData.attestation,
        pluginsSettingsData
      ),
    [
      displayData.attestation,
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
            content: pluginsSettingsData?.settings.length ? (
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
                    entry. Host-owned fields such as command and startup policy stay separate from plugin-owned custom
                    settings.
                  </>
                }
                summaryStatus={
                  pluginsDirty
                    ? `${pluginsSettingsData.settings.length} settings · modified`
                    : `${pluginsSettingsData.settings.length} settings`
                }
                summaryTitle="Plugin settings"
                summaryTitleId="plugins-summary-heading"
                previewTip={
                  <>
                    Plugin manifests own these fields; update or reinstall the plugin when a setting is missing from
                    this list.
                  </>
                }
              />
            ) : (
              <ConfigurationPlaceholderPanel title="Plugins" icon={Blocks}>
                Plugin settings will appear here when an installed plugin publishes config schema metadata.
              </ConfigurationPlaceholderPanel>
            )
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

  const setActiveTab = useCallback(
    (tab: ConfigurationTabId) => {
      if (controlledActiveTab === undefined) setActiveTabState(tab)
      onTabChange?.(tab)
    },
    [controlledActiveTab, onTabChange]
  )

  return (
    <>
      <ConfigurationLayout
        header={
          <ConfigurationHeader
            title={displayData.title}
            description={displayData.description}
            nodes={nodes}
            canUndo={canUndo}
            canRedo={canRedo}
            hasUnsavedChanges={hasUnsavedChanges}
            hasInvalidNode={hasInvalidNode}
            isSaving={isSavingConfiguration}
            saveDisabledReason={runtimeControlSaveDisabledReason}
            onUndo={undoConfigurationChange}
            onRedo={redoConfigurationChange}
            onRevert={revertConfiguration}
            onSave={saveConfiguration}
          />
        }
      >
        {saveAlertMessage ? (
          <div className="px-5 pb-3">
            <Alert variant="destructive">
              <AlertTriangle aria-hidden="true" className="size-4 shrink-0 mt-0.5" />
              <AlertDescription className="min-w-0">
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <>{children}</>,
                    blockquote: ({ children }) => (
                      <div className="mt-1 border-l-2 border-destructive/40 pl-2 text-xs text-foreground/70">
                        {children}
                      </div>
                    ),
                    hr: () => <div className="my-2 border-t border-destructive/20" />,
                    strong: ({ children }) => <span className="font-semibold">{children}</span>,
                    code: ({ children }) => (
                      <code className="rounded bg-destructive/10 px-1 py-0.5 text-xs font-mono">{children}</code>
                    )
                  }}
                >
                  {saveAlertMessage}
                </ReactMarkdown>
              </AlertDescription>
            </Alert>
          </div>
        ) : null}
        <ConfigurationTabs value={renderedActiveTab} onValueChange={setActiveTab} tabs={tabs} />
      </ConfigurationLayout>
      {enableNavigationBlocker ? <UnsavedConfigurationNavigationBlocker hasUnsavedChanges={hasUnsavedChanges} /> : null}
      {catalogFor && selectedCatalogNode ? (
        <CatalogPopover
          open={Boolean(catalogFor)}
          onClose={closeCatalog}
          selectedNode={selectedCatalogNode}
          assigns={assigns}
          models={displayData.catalog}
          errorMessage={catalogError}
          onSelectModel={selectCatalogModel}
        />
      ) : null}
    </>
  )
}

export function ConfigurationPage(props: ConfigurationPageProps = {}) {
  return <ConfigurationPageContent {...props} />
}

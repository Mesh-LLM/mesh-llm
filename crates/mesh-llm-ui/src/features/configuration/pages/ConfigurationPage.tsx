import { useCallback } from 'react'
import { ConfigurationLiveDataBoundary } from '@/features/configuration/components/ConfigurationLiveDataBoundary'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { ConfigurationHarnessData } from '@/features/app-tabs/types'
import { useConfigQuery } from '@/features/configuration/api/use-config-query'
import { ConfigurationEditorPage } from '@/features/configuration/pages/ConfigurationEditorPage'

type ConfigurationPageProps = {
  activeTab?: ConfigurationTabId
  enableNavigationBlocker?: boolean
  initialTab?: ConfigurationTabId
  onTabChange?: (tab: ConfigurationTabId) => void
}

type ConfigurationFixturePageProps = ConfigurationPageProps & {
  data?: ConfigurationHarnessData
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
    (liveData.audit?.settings.length ?? 0) === 0 &&
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

export function ConfigurationPage(props: ConfigurationPageProps = {}) {
  return <ConfigurationPageContent {...props} />
}

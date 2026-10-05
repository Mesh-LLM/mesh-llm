import { LiveLoadingGhostRoot } from '@/components/ui/LiveLoadingGhostRoot'
import {
  ConfigurationDefaultsLoadingGhost,
  ConfigurationHeaderLoadingGhost,
  ConfigurationTabsLoadingGhost
} from '@/features/configuration/components/ConfigurationLoadingGhostSections'
import { ConfigurationLayout } from '@/features/configuration/layouts/ConfigurationLayout'

export function ConfigurationLiveLoadingGhost() {
  return (
    <LiveLoadingGhostRoot>
      <ConfigurationLayout header={<ConfigurationHeaderLoadingGhost />}>
        <ConfigurationTabsLoadingGhost />
        <ConfigurationDefaultsLoadingGhost />
      </ConfigurationLayout>
    </LiveLoadingGhostRoot>
  )
}

import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { ConfigurationLiveLoadingGhost } from '@/features/configuration/components/ConfigurationLiveLoadingGhost'

export type ConfigurationLiveDataBoundaryState = 'loading' | 'error' | 'empty-schema'

type ConfigurationLiveDataBoundaryProps = {
  state: ConfigurationLiveDataBoundaryState
  onRetry: () => void
}

const boundaryCopy: Record<
  Exclude<ConfigurationLiveDataBoundaryState, 'loading'>,
  {
    debugDescription: string
    debugTitle: string
    productionDescription: string
    statusLabel: string
    title: string
  }
> = {
  error: {
    debugDescription:
      'Configuration needs status, model catalog, runtime-control bootstrap, and /api/runtime/config-schema before rendering editable controls. Use the developer playground for fixture-only control inspection.',
    debugTitle: 'Could not load live configuration schema',
    productionDescription:
      'Configuration is waiting for the live node schema and config snapshot before rendering editable controls.',
    statusLabel: 'Configuration schema unavailable',
    title: 'Configuration schema is unavailable'
  },
  'empty-schema': {
    debugDescription:
      'The live configuration schema loaded, but it did not expose any supported Defaults controls. Check the schema export before editing configuration.',
    debugTitle: 'Live configuration schema has no Defaults controls',
    productionDescription:
      'The live configuration schema did not expose any editable Defaults controls. Retry after the service refreshes.',
    statusLabel: 'Configuration schema empty',
    title: 'Configuration schema is empty'
  }
}

export function ConfigurationLiveDataBoundary({ state, onRetry }: ConfigurationLiveDataBoundaryProps) {
  if (state === 'loading') return <ConfigurationLiveLoadingGhost />

  const copy = boundaryCopy[state]

  return (
    <LiveDataUnavailableOverlay
      debugDescription={copy.debugDescription}
      debugTitle={copy.debugTitle}
      productionDescription={copy.productionDescription}
      statusLabel={copy.statusLabel}
      title={copy.title}
      onRetry={onRetry}
    >
      <ConfigurationLiveLoadingGhost />
    </LiveDataUnavailableOverlay>
  )
}

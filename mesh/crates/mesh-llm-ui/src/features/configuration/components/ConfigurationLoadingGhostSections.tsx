import { LoadingGhostBlock } from '@/components/ui/LoadingGhostBlock'

const TAB_GHOST_ROWS = ['defaults', 'deployment', 'signing', 'integrations', 'toml']
const CATEGORY_GHOST_ROWS = ['runtime', 'memory', 'speculative', 'request', 'transport', 'multimodal', 'server']
const DEFAULT_SETTING_GHOST_ROWS = ['threads', 'batch', 'continuous', 'gpu-layers', 'parallel', 'attention']

export function ConfigurationHeaderLoadingGhost() {
  return (
    <header className="relative z-20 bg-transparent">
      <div className="flex min-h-[58px] flex-wrap items-center justify-between gap-x-4 gap-y-2 py-0">
        <div className="min-w-0">
          <LoadingGhostBlock className="h-5 w-32" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-[min(460px,70vw)]" shimmer />
        </div>
        <div className="flex items-center gap-1.5">
          <LoadingGhostBlock className="h-[30px] w-[30px]" shimmer />
          <LoadingGhostBlock className="h-[30px] w-[30px]" shimmer />
          <LoadingGhostBlock className="h-[30px] w-24" shimmer />
          <LoadingGhostBlock className="h-[30px] w-28" shimmer />
        </div>
      </div>
    </header>
  )
}

export function ConfigurationTabsLoadingGhost() {
  return (
    <div className="border-b border-border-soft" data-loading-region="configuration-tabs">
      <div className="flex min-h-[49px] items-end gap-6 overflow-hidden">
        {TAB_GHOST_ROWS.map((row, index) => (
          <div key={row} className="flex h-[49px] shrink-0 items-center gap-2 border-b-2 border-transparent">
            <LoadingGhostBlock className="size-4" shimmer />
            <LoadingGhostBlock className={index === 1 ? 'h-4 w-28' : 'h-4 w-20'} shimmer />
          </div>
        ))}
      </div>
    </div>
  )
}

export function ConfigurationDefaultsNoticeLoadingGhost() {
  return (
    <section
      className="panel-shell mt-[14px] rounded-[var(--radius-lg)] border border-border bg-panel p-4"
      data-loading-region="configuration-summary"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <LoadingGhostBlock className="h-5 w-36" shimmer />
            <LoadingGhostBlock className="h-5 w-24" shimmer />
          </div>
          <LoadingGhostBlock className="mt-3 h-3 w-full max-w-[920px]" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-[min(760px,82vw)]" shimmer />
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <LoadingGhostBlock className="h-[34px] w-28" shimmer />
          <LoadingGhostBlock className="h-[34px] w-24" shimmer />
        </div>
      </div>
    </section>
  )
}

function ConfigurationCategoryRailLoadingGhost() {
  return (
    <aside className="min-w-0" data-loading-region="configuration-categories">
      <LoadingGhostBlock className="h-3 w-24" shimmer />
      <div className="mt-3 space-y-1.5">
        {CATEGORY_GHOST_ROWS.map((row, index) => (
          <div
            key={row}
            className={index === 0 ? 'rounded-[var(--radius)] bg-selection/55 px-2.5 py-2' : 'px-2.5 py-2'}
          >
            <div className="flex items-center gap-2">
              <LoadingGhostBlock className="size-4" shimmer />
              <LoadingGhostBlock className="h-4 flex-1" shimmer />
              <LoadingGhostBlock className="h-3 w-6" shimmer />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-5 border-t border-border-soft pt-4">
        <LoadingGhostBlock className="h-3 w-32" shimmer />
        <LoadingGhostBlock className="mt-2 h-4 w-44" shimmer />
      </div>
    </aside>
  )
}

function ConfigurationSettingsCardLoadingGhost() {
  return (
    <section
      className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-4"
      data-loading-region="configuration-settings"
    >
      <div className="flex items-start gap-3">
        <LoadingGhostBlock className="size-9" shimmer />
        <div className="min-w-0 flex-1">
          <LoadingGhostBlock className="h-5 w-28" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-72 max-w-full" shimmer />
        </div>
      </div>
      <div className="mt-5 divide-y divide-border-soft">
        {DEFAULT_SETTING_GHOST_ROWS.map((row, index) => (
          <div key={row} className="grid gap-3 py-3 md:grid-cols-[minmax(0,1fr)_minmax(220px,360px)]">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <LoadingGhostBlock className="h-4 w-36" shimmer />
                {index < 5 ? <LoadingGhostBlock className="h-4 w-20 rounded-full" shimmer /> : null}
              </div>
              <LoadingGhostBlock className="mt-2 h-3 w-full max-w-[560px]" shimmer />
              <LoadingGhostBlock className="mt-1.5 h-3 w-2/3 max-w-[420px]" shimmer />
            </div>
            <div className="flex min-w-0 items-center justify-end">
              <LoadingGhostBlock className={index % 3 === 0 ? 'h-8 w-full' : 'h-8 w-[260px]'} shimmer />
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

function ConfigurationPreviewLoadingGhost() {
  return (
    <aside className="hidden min-w-0 space-y-[14px] xl:block" data-loading-region="configuration-preview">
      <section className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-3">
        <LoadingGhostBlock className="h-4 w-40" shimmer />
        <div className="mt-3 rounded-[var(--radius)] border border-border-soft bg-background p-3">
          <LoadingGhostBlock className="h-3 w-44" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-32" shimmer />
          <LoadingGhostBlock className="mt-5 h-3 w-48" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-36" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-40" shimmer />
          <LoadingGhostBlock className="mt-5 h-3 w-52" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-28" shimmer />
        </div>
      </section>
      <section className="rounded-[var(--radius-lg)] border border-dashed border-border-soft bg-panel/75 p-3">
        <LoadingGhostBlock className="h-3 w-10" shimmer />
        <LoadingGhostBlock className="mt-3 h-3 w-full" shimmer />
        <LoadingGhostBlock className="mt-2 h-3 w-3/4" shimmer />
      </section>
    </aside>
  )
}

export function ConfigurationDefaultsLoadingGhost() {
  return (
    <>
      <ConfigurationDefaultsNoticeLoadingGhost />
      <div className="grid min-w-0 gap-[14px] pt-[14px] xl:grid-cols-[250px_minmax(0,1fr)_minmax(280px,340px)]">
        <ConfigurationCategoryRailLoadingGhost />
        <ConfigurationSettingsCardLoadingGhost />
        <ConfigurationPreviewLoadingGhost />
      </div>
    </>
  )
}

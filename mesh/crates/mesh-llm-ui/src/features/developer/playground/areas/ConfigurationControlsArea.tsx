import { useState } from 'react'
import { Blocks, Brackets, Computer, Cpu, Network, ShieldCheck, SlidersHorizontal } from 'lucide-react'
import { NativeSelect } from '@/components/ui/NativeSelect'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { Slider } from '@/components/ui/Slider'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { ModelSelect } from '@/features/chat/components/ModelSelect'
import { CatalogPopover } from '@/features/configuration/components/CatalogPopover'
import { ConfigurationHeader } from '@/features/configuration/components/ConfigurationHeader'
import { ConfigurationTabs, type ConfigurationTabItem } from '@/features/configuration/components/ConfigurationTabs'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'
import { CtxSlider } from '@/features/configuration/components/CtxSlider'
import { DefaultsTab } from '@/features/configuration/components/DefaultsTab'
import { ModelConfigCard } from '@/features/configuration/components/ModelConfigCard'
import { NodeRail } from '@/features/configuration/components/NodeRail'
import { NodeSection } from '@/features/configuration/components/NodeSection'
import { PlacementToggle } from '@/features/configuration/components/PlacementToggle'
import { ReservedConfigCard } from '@/features/configuration/components/ReservedConfigCard'
import { TomlView } from '@/features/configuration/components/TomlView'
import PlaygroundTomlDiff from '@/features/app-shell/playground/components/PlaygroundTomlDiff'
import { VRAMBar } from '@/features/configuration/components/VRAMBar'
import { createDefaultsValues } from '@/features/configuration/hooks/useDefaultsSettingsState'
import {
  containerReservedGB,
  containerTotalGB,
  containerUsedGB,
  findModel
} from '@/features/configuration/lib/config-math'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { ConfigNode, Placement } from '@/features/app-tabs/types'
import { OptionGroup, PlaygroundPanel, SidebarTabs } from '@/features/developer/playground/primitives'
import type { DeveloperPlaygroundState } from '@/features/developer/playground/useDeveloperPlaygroundState'

export function ConfigurationControlsArea({ state }: { state: DeveloperPlaygroundState }) {
  const [activeConfigTab, setActiveConfigTab] = useState<ConfigurationTabId>('general')
  const [collapsedMap, setCollapsedMap] = useState<Record<string, boolean>>({})
  const [catalogNode, setCatalogNode] = useState<ConfigNode | null>(null)
  const [defaultsValues, setDefaultsValues] = useState(() => createDefaultsValues(CONFIGURATION_HARNESS.defaults))
  const [tomlMode, setTomlMode] = useState<'output' | 'diff'>('output')

  const sectionTabs: ConfigurationTabItem[] = [
    {
      id: 'general',
      label: 'General',
      icon: Cpu,
      dirty: true,
      content: (
        <div className="rounded-[var(--radius)] border border-border bg-background p-3 text-[length:var(--density-type-caption-lg)] text-fg-dim">
          General tab trigger with dirty accessory.
        </div>
      )
    },
    {
      id: 'runtime',
      label: 'Runtime',
      icon: SlidersHorizontal,
      dirty: true,
      content: (
        <div className="rounded-[var(--radius)] border border-border bg-background p-3 text-[length:var(--density-type-caption-lg)] text-fg-dim">
          Runtime tab trigger and navigation density.
        </div>
      )
    },
    {
      id: 'models',
      label: 'Models',
      icon: Computer,
      content: (
        <div className="rounded-[var(--radius)] border border-border bg-background p-3 text-[length:var(--density-type-caption-lg)] text-fg-dim">
          Model settings trigger.
        </div>
      )
    },
    {
      id: 'network',
      label: 'Network',
      icon: Network,
      content: (
        <div className="rounded-[var(--radius)] border border-border bg-background p-3 text-[length:var(--density-type-caption-lg)] text-fg-dim">
          Network settings trigger.
        </div>
      )
    },
    {
      id: 'local-deployment',
      label: 'Model Deployment',
      icon: Computer,
      content: (
        <div className="rounded-[var(--radius)] border border-border bg-background p-3 text-[length:var(--density-type-caption-lg)] text-fg-dim">
          Deployment tab trigger.
        </div>
      )
    },
    {
      id: 'signing',
      label: 'Signing / Attestation',
      icon: ShieldCheck,
      content: (
        <div className="rounded-[var(--radius)] border border-border bg-background p-3 text-[length:var(--density-type-caption-lg)] text-fg-dim">
          Reserved signing surface.
        </div>
      )
    },
    {
      id: 'plugins',
      label: 'Plugins',
      icon: Blocks,
      content: (
        <div className="rounded-[var(--radius)] border border-border bg-background p-3 text-[length:var(--density-type-caption-lg)] text-fg-dim">
          Plugin settings surface.
        </div>
      )
    },
    {
      id: 'toml-review',
      label: 'TOML Output',
      icon: Brackets,
      content: (
        <div className="rounded-[var(--radius)] border border-border bg-background p-3 text-[length:var(--density-type-caption-lg)] text-fg-dim">
          Review output trigger.
        </div>
      )
    }
  ]

  function setNodePlacement(nodeId: string, placement: Placement) {
    if (nodeId === state.primaryNode.id) state.handlePlacementChange(placement)
  }

  return (
    <>
      <SidebarTabs
        ariaLabel="Configuration control previews"
        defaultValue="layout"
        tabs={[
          {
            value: 'layout',
            label: 'Layout and VRAM',
            content: (
              <PlaygroundPanel
                title="Placement and VRAM state"
                description="Drive the same node, GPU, and assignment data through placement controls, focused context, and live VRAM bars."
              >
                <div className="grid gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
                  <div className="space-y-4">
                    <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                      <div className="type-label text-fg-faint">Placement</div>
                      <div className="mt-2">
                        <PlacementToggle
                          groupId="developer-playground-node-a"
                          placement={state.primaryNode.placement}
                          onChange={state.handlePlacementChange}
                        />
                      </div>
                    </div>

                    {state.focusedAssign ? (
                      <OptionGroup
                        label="Focused card"
                        value={state.focusedAssign.id}
                        options={state.primaryNodeAssigns.map((assign) => ({
                          value: assign.id,
                          label: findModel(assign.modelId, CONFIGURATION_HARNESS.catalog)?.name ?? assign.modelId
                        }))}
                        onChange={state.selectFocusedAssign}
                      />
                    ) : null}

                    <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                      <div className="type-label text-fg-faint">Target container</div>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {(state.primaryNode.placement === 'pooled'
                          ? [{ value: '0', label: `Pool · ${state.primaryNode.gpus.length} GPUs` }]
                          : state.primaryNode.gpus.map((gpu) => ({ value: `${gpu.idx}`, label: `GPU ${gpu.idx}` }))
                        ).map((option) => (
                          <button
                            key={option.value}
                            aria-pressed={`${state.effectiveSelectedContainerIdx}` === option.value}
                            className="ui-control inline-flex items-center rounded-[var(--radius)] border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium"
                            data-active={`${state.effectiveSelectedContainerIdx}` === option.value ? 'true' : undefined}
                            onClick={() => state.moveFocusedAssign(Number(option.value))}
                            type="button"
                          >
                            {option.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {state.focusedAssign && state.focusedConfigModel ? (
                      <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                        <div className="type-label text-fg-faint">Focused context</div>
                        <div className="mt-2">
                          <CtxSlider
                            invalid={state.focusedContainerFreeGB < 0}
                            maxCtx={Math.max(1024, state.focusedConfigModel.ctxMaxK * 1024)}
                            onChange={(nextValue) => state.updateAssignCtx(state.focusedAssign.id, nextValue)}
                            value={state.focusedAssign.ctx}
                          />
                        </div>
                      </div>
                    ) : null}

                    <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <div className="type-label text-fg-faint">GPU inventory</div>
                          <div className="mt-1 text-[length:var(--density-type-caption)] text-fg-dim">
                            Add or remove containers and watch the VRAM controls update immediately.
                          </div>
                        </div>
                        <button
                          className="ui-control-primary inline-flex items-center rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
                          onClick={state.addGpu}
                          type="button"
                        >
                          Add GPU
                        </button>
                      </div>
                      <div className="mt-3 space-y-2">
                        {state.primaryNode.gpus.map((gpu) => (
                          <div
                            key={gpu.idx}
                            className="flex items-center justify-between gap-2 rounded-[var(--radius)] border border-border-soft px-3 py-2"
                          >
                            <div>
                              <div className="font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
                                GPU {gpu.idx} · {gpu.name}
                              </div>
                              <div className="mt-0.5 text-[length:var(--density-type-label)] text-fg-faint">
                                {gpu.totalGB} GB total · {gpu.reservedGB ?? 0} GB reserved
                              </div>
                            </div>
                            <button
                              aria-label={`Remove GPU ${gpu.idx}`}
                              className="ui-control inline-flex items-center rounded-[var(--radius)] border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium"
                              disabled={state.primaryNode.gpus.length <= 1}
                              onClick={() => state.removeGpu(gpu.idx)}
                              type="button"
                            >
                              Remove
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {state.vramBars.map((bar) => (
                      <VRAMBar
                        key={`${state.primaryNode.id}-${bar.containerIdx}`}
                        assigns={state.configAssigns}
                        containerIdx={bar.containerIdx}
                        dragOver={state.dragOver}
                        label={bar.label}
                        models={CONFIGURATION_HARNESS.catalog}
                        node={state.primaryNode}
                        onPick={(selectionId) => {
                          state.setSelectedConfigSelectionId(selectionId)
                          if (selectionId && state.configAssigns.some((assign) => assign.id === selectionId)) {
                            state.selectFocusedAssign(selectionId)
                          }
                        }}
                        onSelectContainer={() => state.moveFocusedAssign(bar.containerIdx)}
                        reservedGB={bar.reservedGB}
                        selectedContainer={state.effectiveSelectedContainerIdx === bar.containerIdx}
                        selectedId={state.selectedConfigSelectionId}
                        setAssigns={state.setConfigAssigns}
                        setDragOver={state.setDragOver}
                        totalGB={bar.totalGB}
                      />
                    ))}
                  </div>
                </div>
              </PlaygroundPanel>
            )
          },
          {
            value: 'cards',
            label: 'Config cards',
            content: (
              <PlaygroundPanel
                title="Model config cards"
                description="Add, remove, and retarget card state against the live assignment list. Each card updates the same TOML and VRAM view."
                actions={
                  <button
                    className="ui-control-primary inline-flex items-center rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
                    onClick={state.addConfigCard}
                    type="button"
                  >
                    Add config card
                  </button>
                }
              >
                <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
                  <div className="space-y-4">
                    <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                      <div className="type-label text-fg-faint">New card model</div>
                      <div className="mt-2">
                        <ModelSelect
                          options={state.configCardOptions}
                          value={state.selectedConfigModelId}
                          onChange={state.setSelectedConfigModelId}
                        />
                      </div>
                    </div>
                    {state.focusedAssign ? (
                      <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                        <div className="type-label text-fg-faint">Focused assignment</div>
                        <div className="mt-2 font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
                          {findModel(state.focusedAssign.modelId, CONFIGURATION_HARNESS.catalog)?.name ??
                            state.focusedAssign.modelId}
                        </div>
                        <div className="mt-1 text-[length:var(--density-type-label)] text-fg-faint">
                          {state.primaryNode.placement === 'pooled'
                            ? 'pool'
                            : `GPU ${state.focusedAssign.containerIdx}`}{' '}
                          · {state.focusedAssign.ctx.toLocaleString()} ctx
                        </div>
                      </div>
                    ) : null}
                  </div>

                  <div className="space-y-4">
                    {state.primaryNodeAssigns.map((assign) => (
                      <div key={assign.id} className="space-y-2">
                        <button
                          aria-pressed={state.focusedAssignId === assign.id}
                          className="ui-control-ghost inline-flex items-center rounded-[var(--radius)] border border-transparent px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium data-[active=true]:ui-control"
                          data-active={state.focusedAssignId === assign.id ? 'true' : undefined}
                          onClick={() => state.selectFocusedAssign(assign.id)}
                          type="button"
                        >
                          Focus {findModel(assign.modelId, CONFIGURATION_HARNESS.catalog)?.name ?? assign.modelId}
                        </button>
                        <ModelConfigCard
                          assign={assign}
                          containerFreeGB={
                            containerTotalGB(
                              state.primaryNode,
                              state.primaryNode.placement === 'pooled' ? 0 : assign.containerIdx
                            ) -
                            containerReservedGB(
                              state.primaryNode,
                              state.primaryNode.placement === 'pooled' ? 0 : assign.containerIdx
                            ) -
                            containerUsedGB(
                              state.configAssigns.filter((item) => item.id !== assign.id),
                              state.primaryNode.id,
                              state.primaryNode.placement === 'pooled' ? 0 : assign.containerIdx,
                              CONFIGURATION_HARNESS.catalog
                            ) -
                            (findModel(assign.modelId, CONFIGURATION_HARNESS.catalog)?.sizeGB ?? 0)
                          }
                          models={CONFIGURATION_HARNESS.catalog}
                          node={state.primaryNode}
                          onCtxChange={(ctx) => state.updateAssignCtx(assign.id, ctx)}
                          onPick={() => state.selectFocusedAssign(assign.id)}
                          onRemove={() => state.removeConfigCard(assign.id)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </PlaygroundPanel>
            )
          },
          {
            value: 'toml',
            label: 'TOML',
            content: (
              <div className="flex h-full flex-col">
                <div className="flex items-center justify-between border-b border-border-soft px-3 py-2">
                  <SegmentedControl
                    ariaLabel="TOML view mode"
                    options={[
                      { value: 'output', label: 'Output' },
                      { value: 'diff', label: 'Diff' }
                    ]}
                    value={tomlMode}
                    variant="pill"
                    onValueChange={(v) => setTomlMode(v as 'output' | 'diff')}
                  />
                </div>
                <div className="flex-1 overflow-hidden">
                  {tomlMode === 'output' ? (
                    <TomlView
                      assigns={state.configAssigns}
                      models={CONFIGURATION_HARNESS.catalog}
                      nodes={state.configNodes}
                    />
                  ) : (
                    <PlaygroundTomlDiff />
                  )}
                </div>
              </div>
            )
          },
          {
            value: 'sections',
            label: 'Sections',
            content: (
              <>
                <PlaygroundPanel
                  title="Configuration header"
                  description="Undo, redo, revert, save, dirty, and invalid-allocation states are visible without mounting the full page route."
                >
                  <ConfigurationHeader
                    title="Configuration"
                    description="Local model deployment and inherited llama.cpp defaults."
                    nodes={state.configNodes}
                    canUndo
                    canRedo
                    hasUnsavedChanges
                    hasInvalidNode={state.focusedContainerFreeGB < 0}
                    onUndo={() => undefined}
                    onRedo={() => undefined}
                    onRevert={() => undefined}
                    onSave={() => undefined}
                  />
                </PlaygroundPanel>

                <PlaygroundPanel
                  title="Configuration tabs"
                  description="The specialized tab wrapper carries icons, dirty dots, and review destinations used by the route."
                >
                  <ConfigurationTabs value={activeConfigTab} onValueChange={setActiveConfigTab} tabs={sectionTabs} />
                </PlaygroundPanel>

                <div className="grid gap-4 xl:grid-cols-[260px_minmax(0,1fr)]">
                  <PlaygroundPanel
                    title="Node rail"
                    description="Rail density, collapse state, node counts, and keyboard hint slot."
                  >
                    <NodeRail
                      nodes={state.configNodes}
                      assigns={state.configAssigns}
                      models={CONFIGURATION_HARNESS.catalog}
                      collapsedMap={collapsedMap}
                      setCollapsedMap={setCollapsedMap}
                      onJump={(nodeId) => state.setSelectedConfigSelectionId(nodeId)}
                      keyboardHint={
                        <span className="text-[length:var(--density-type-caption)] text-fg-dim">
                          Arrows move between GPUs; A opens catalog.
                        </span>
                      }
                    />
                  </PlaygroundPanel>

                  <div className="space-y-4">
                    <NodeSection
                      node={state.primaryNode}
                      assigns={state.configAssigns}
                      models={CONFIGURATION_HARNESS.catalog}
                      setAssigns={state.setConfigAssigns}
                      selectedId={state.selectedConfigSelectionId}
                      selectedContainerIdx={state.effectiveSelectedContainerIdx}
                      selectedNode
                      onFocusNode={() => undefined}
                      onPick={(selectionId) => {
                        state.setSelectedConfigSelectionId(selectionId)
                        if (selectionId && state.primaryNodeAssigns.some((assign) => assign.id === selectionId))
                          state.selectFocusedAssign(selectionId)
                      }}
                      onSelectContainer={state.moveFocusedAssign}
                      collapsed={Boolean(collapsedMap[state.primaryNode.id])}
                      setCollapsed={(collapsed) =>
                        setCollapsedMap((map) => ({ ...map, [state.primaryNode.id]: collapsed }))
                      }
                      onOpenCatalog={setCatalogNode}
                      onPlacementChange={setNodePlacement}
                    />
                    <ReservedConfigCard
                      locationLabel={`${state.primaryNode.hostname} pool`}
                      reservedGB={state.primaryNode.gpus.reduce((sum, gpu) => sum + (gpu.reservedGB ?? 0), 0)}
                    />
                  </div>
                </div>
              </>
            )
          },
          {
            value: 'defaults',
            label: 'Defaults tab',
            content: (
              <DefaultsTab
                data={CONFIGURATION_HARNESS.defaults}
                values={defaultsValues}
                onResetAll={() => setDefaultsValues(createDefaultsValues(CONFIGURATION_HARNESS.defaults))}
                onSettingValueChange={(settingId, value) =>
                  setDefaultsValues((values) => ({ ...values, [settingId]: value }))
                }
                configFilePath={CONFIGURATION_HARNESS.configFilePath}
              />
            )
          },
          {
            value: 'live-state',
            label: 'Live state',
            content: (
              <LiveDataUnavailableOverlay
                debugTitle="Could not reach live configuration sources"
                title="Live configuration is unavailable"
                debugDescription="Configuration could not fetch the initial status and model catalog from the configured API target."
                productionDescription="Configuration is waiting for live node and model data before rendering editable controls."
                onRetry={() => undefined}
                onSwitchToTestData={() => undefined}
              >
                <TomlView
                  assigns={state.configAssigns}
                  models={CONFIGURATION_HARNESS.catalog}
                  nodes={state.configNodes}
                />
              </LiveDataUnavailableOverlay>
            )
          },
          {
            value: 'error-states',
            label: 'Error states',
            content: (
              <PlaygroundPanel
                title="Control error states"
                description="Each shared UI component rendered with its invalid prop active, showing the error ring, border, and aria-invalid attribute."
              >
                <div className="space-y-6">
                  <div>
                    <div className="type-label text-fg-faint mb-2">NativeSelect — invalid</div>
                    <NativeSelect
                      ariaLabel="Error demo select"
                      invalid
                      name="demo-select-error"
                      onValueChange={() => undefined}
                      options={[
                        { value: 'a', label: 'Option A' },
                        { value: 'b', label: 'Option B' }
                      ]}
                      value="a"
                    />
                  </div>
                  <div>
                    <div className="type-label text-fg-faint mb-2">SegmentedControl (pill) — invalid</div>
                    <SegmentedControl
                      ariaLabel="Error demo segmented"
                      invalid
                      name="demo-segmented-error"
                      onValueChange={() => undefined}
                      options={[
                        { value: 'on', label: 'On' },
                        { value: 'off', label: 'Off' },
                        { value: 'auto', label: 'Auto' }
                      ]}
                      value="on"
                      variant="pill"
                    />
                  </div>
                  <div>
                    <div className="type-label text-fg-faint mb-2">SegmentedControl (buttons) — valid (no error)</div>
                    <SegmentedControl
                      ariaLabel="Valid demo segmented"
                      name="demo-segmented-valid"
                      onValueChange={() => undefined}
                      options={[
                        { value: 'on', label: 'On' },
                        { value: 'off', label: 'Off' },
                        { value: 'auto', label: 'Auto' }
                      ]}
                      value="on"
                      variant="buttons"
                    />
                  </div>
                  <div>
                    <div className="type-label text-fg-faint mb-2">Slider — invalid</div>
                    <Slider
                      ariaLabel="Error demo slider"
                      invalid
                      max={100}
                      min={0}
                      name="demo-slider-error"
                      onValueChange={() => undefined}
                      value="50"
                    />
                  </div>
                  <div>
                    <div className="type-label text-fg-faint mb-2">Slider — valid (no error)</div>
                    <Slider
                      ariaLabel="Valid demo slider"
                      max={100}
                      min={0}
                      name="demo-slider-valid"
                      onValueChange={() => undefined}
                      value="50"
                    />
                  </div>
                </div>
              </PlaygroundPanel>
            )
          }
        ]}
      />
      <CatalogPopover
        open={catalogNode !== null}
        onClose={() => setCatalogNode(null)}
        selectedNode={catalogNode ?? state.primaryNode}
        assigns={state.configAssigns}
        models={CONFIGURATION_HARNESS.catalog}
        onSelectModel={(model) => {
          state.setSelectedConfigModelId(model.id)
          setCatalogNode(null)
          return true
        }}
      />
    </>
  )
}

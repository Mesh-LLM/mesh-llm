import { describe, expect, it } from 'vitest'
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConfigurationFixturePage as ConfigurationPage } from '@/features/configuration/pages/ConfigurationPage'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { ConfigurationHarnessData } from '@/features/app-tabs/types'
import {
  countTomlOccurrences,
  dispatchShortcut,
  getCarrackSection,
  openTomlOutput,
  render
} from './ConfigurationPage-test-support'

vi.mock('@tanstack/react-router', () => ({
  useBlocker: (...args: unknown[]) => globalThis.__meshConfigurationPageTestGlobals.useBlocker(...args)
}))

vi.mock('@/lib/feature-flags', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/feature-flags')>()

  return {
    ...actual,
    useBooleanFeatureFlag: vi.fn((path: string) =>
      globalThis.__meshConfigurationPageTestGlobals.useBooleanFeatureFlag(path)
    )
  }
})

vi.mock('@/features/plugins/api/plugin-web-ui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/features/plugins/api/plugin-web-ui')>()

  return {
    ...actual,
    usePluginSummariesQuery: vi.fn((...args: unknown[]) =>
      globalThis.__meshConfigurationPageTestGlobals.usePluginSummariesQuery(...args)
    ),
    useSetPluginWebUiEnabledMutation: vi.fn((...args: unknown[]) =>
      globalThis.__meshConfigurationPageTestGlobals.useSetPluginWebUiEnabledMutation(...args)
    ),
    usePluginWebUiConfigQuery: vi.fn((pluginName: string, ...args: unknown[]) =>
      globalThis.__meshConfigurationPageTestGlobals.usePluginWebUiConfigQuery(pluginName, ...args)
    ),
    usePluginWebUiConfigMutation: vi.fn((pluginName: string, ...args: unknown[]) =>
      globalThis.__meshConfigurationPageTestGlobals.usePluginWebUiConfigMutation(pluginName, ...args)
    )
  }
})

vi.mock('@/features/plugins/web-ui/bundle-loader', () => ({
  importPluginUiBundle: (...args: unknown[]) =>
    globalThis.__meshConfigurationPageTestGlobals.importPluginUiBundle(...args),
  assertPluginUiRegistration: vi.fn(),
  assertPluginUiMountHandle: vi.fn()
}))

describe('ConfigurationPage placement and history', () => {
  it('renders remote nodes as read-only context and keeps TOML in the output tab', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    expect(screen.getByRole('heading', { name: 'perseus.local' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'triton.lab' })).toBeInTheDocument()
    expect(screen.getByText('Peers')).toBeInTheDocument()
    expect(screen.getByText('read-only')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add model to perseus.local' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Add model to triton.lab' })).toBeDisabled()
    expect(screen.queryByRole('textbox', { name: /configuration toml source/i })).not.toBeInTheDocument()

    const tomlSource = await openTomlOutput(user)
    expect(tomlSource.value).toContain('version = 1')
    expect(tomlSource.value).toContain('model = "GLM-4.7-Flash-Q4_K_M"')
    expect(tomlSource.value).not.toContain('perseus.local')
    expect(tomlSource.value).not.toContain('triton.lab')
  })

  it('blocks remote VRAM chip and slot interactions from editing placement', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const carrackGpu3Capacity = within(getCarrackSection()).getAllByRole('region', {
      name: /rtx 6000 pro capacity/i
    })[2]
    if (!carrackGpu3Capacity) throw new Error('Expected carrack GPU 3 capacity region')

    await user.click(carrackGpu3Capacity)
    expect(carrackGpu3Capacity.closest('[data-config-container-selected="true"]')).toBeInTheDocument()

    const perseusSection = screen.getByRole('heading', { name: 'perseus.local' }).closest('section')
    if (!perseusSection) throw new Error('Expected perseus.local section')

    const remoteCapacity = within(perseusSection).getByRole('region', { name: /unified memory capacity/i })
    const remoteChip = within(remoteCapacity).getByRole('button', { name: /qwen3\.5-27b-ud-q4_k_xl, .* read-only/i })
    const remoteReservedLane = within(remoteCapacity).getByRole('button', { name: /system reserved space/i })

    expect(remoteChip).toBeDisabled()
    expect(remoteChip).toHaveAttribute('draggable', 'false')
    expect(remoteReservedLane).toBeDisabled()

    await user.click(remoteCapacity)
    expect(remoteCapacity.closest('[data-config-container-selected="true"]')).not.toBeInTheDocument()
    expect(carrackGpu3Capacity.closest('[data-config-container-selected="true"]')).toBeInTheDocument()

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 3/i })).toBeInTheDocument()
  })

  it('uses the clicked GPU container as the catalog add target', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const gpu3Capacity = within(getCarrackSection()).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[2]
    if (!gpu3Capacity) throw new Error('Expected carrack GPU 3 capacity region')

    await user.click(gpu3Capacity)

    const selectedGpu3Container = gpu3Capacity.closest('[data-config-container-selected="true"]')
    if (!(selectedGpu3Container instanceof HTMLElement))
      throw new Error('Expected clicked GPU 3 container to be selected')

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 3/i })).toBeInTheDocument()
    expect(within(gpu3Capacity).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
  })

  it('uses the arrow-key selected empty GPU slot as the catalog add target', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const gpu3Capacity = within(getCarrackSection()).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[2]
    if (!gpu3Capacity) throw new Error('Expected carrack GPU 3 capacity region')

    await user.keyboard('{ArrowDown}{ArrowDown}')

    const selectedGpu3Container = gpu3Capacity.closest('[data-config-container-selected="true"]')
    if (!(selectedGpu3Container instanceof HTMLElement)) throw new Error('Expected arrow-key selected GPU 3 container')

    const modelSelectionEvent = await dispatchShortcut('ArrowRight')
    expect(modelSelectionEvent.defaultPrevented).toBe(true)
    expect(gpu3Capacity.closest('[data-config-container-selected="true"]')).toBe(selectedGpu3Container)
    expect(screen.queryByRole('button', { name: /^remove /i })).not.toBeInTheDocument()

    await user.keyboard('a')
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 3/i })).toBeInTheDocument()
    expect(within(gpu3Capacity).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
  })

  it('keeps the current model selected when Tab has no other editable node target', async () => {
    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    expect(screen.getByRole('button', { name: /remove llama-3\.3-70b-q4_k_m from gpu 1/i })).toBeInTheDocument()

    const tabEvent = await dispatchShortcut('Tab')

    expect(tabEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /remove llama-3\.3-70b-q4_k_m from gpu 1/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /remove glm-4\.7-flash-q4_k_m from gpu 0/i })).not.toBeInTheDocument()
  })

  it('keeps model configuration open when clicking undo but closes it on page background', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    await user.keyboard('{ArrowDown}')
    const contextEvent = await dispatchShortcut('ArrowRight', { altKey: true })
    expect(contextEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /undo/i }))

    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()

    fireEvent.pointerDown(document.body)

    expect(screen.queryByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).not.toBeInTheDocument()
  })

  it('keeps keyboard edits scoped to the local node when remote assignments exist', async () => {
    const user = userEvent.setup()
    const data = {
      ...CONFIGURATION_HARNESS,
      assigns: [
        ...CONFIGURATION_HARNESS.assigns,
        { id: 'a6', modelId: 'phi4', nodeId: 'node-b', containerIdx: 0, ctx: 4096 }
      ],
      preferredAssignId: 'a2'
    }

    render(<ConfigurationPage initialTab="local-deployment" data={data} enableNavigationBlocker={false} />)

    await user.keyboard('{ArrowDown}')
    const contextEvent = await dispatchShortcut('ArrowRight', { altKey: true })

    expect(contextEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )

    const tomlSource = await openTomlOutput(user)
    expect(tomlSource.value).not.toContain('phi-4-mini')
  })

  it('selects models within the current GPU slot with left and right arrows', async () => {
    const user = userEvent.setup()
    const data = {
      ...CONFIGURATION_HARNESS,
      assigns: [
        ...CONFIGURATION_HARNESS.assigns,
        { id: 'a6', modelId: 'phi4', nodeId: 'node-a', containerIdx: 2, ctx: 4096 }
      ],
      preferredAssignId: 'a3'
    }

    render(<ConfigurationPage initialTab="local-deployment" data={data} enableNavigationBlocker={false} />)

    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()

    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 2/i })).toBeInTheDocument()

    await user.keyboard('{ArrowLeft}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()

    await user.keyboard('{Shift>}{ArrowRight}{/Shift}')
    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 2/i })).toBeInTheDocument()

    await user.keyboard('{Shift>}{ArrowLeft}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()
  })

  it('selects models from reserved lane selection within the current GPU slot', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const gpu2Capacity = within(getCarrackSection()).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[1]
    if (!gpu2Capacity) throw new Error('Expected carrack GPU 2 capacity region')

    await user.click(
      within(gpu2Capacity).getByRole('button', { name: /system reserved space, .* reserved on rtx 6000 pro/i })
    )
    expect(screen.queryByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).not.toBeInTheDocument()

    await user.keyboard('{ArrowRight}')

    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()
  })

  it('does not expose remote pooled placements as editable model buttons', () => {
    const data = {
      ...CONFIGURATION_HARNESS,
      assigns: [
        ...CONFIGURATION_HARNESS.assigns,
        { id: 'a6', modelId: 'phi4', nodeId: 'node-b', containerIdx: 0, ctx: 4096 }
      ],
      preferredAssignId: 'a5'
    }

    render(<ConfigurationPage initialTab="local-deployment" data={data} enableNavigationBlocker={false} />)

    expect(
      screen.queryByRole('button', { name: /remove qwen3\.5-27b-ud-q4_k_xl from perseus\.local pool/i })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: /remove phi-4-mini from perseus\.local pool/i })
    ).not.toBeInTheDocument()
    expect(screen.getByText('Qwen3.5-27B-UD-Q4_K_XL')).toBeInTheDocument()
    expect(screen.getByText('phi-4-mini')).toBeInTheDocument()
  })

  it('restores separate GPU assignments after previewing pooled placement', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const gpu2Capacity = within(getCarrackSection()).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[1]
    const rtx3080Capacity = within(getCarrackSection()).getByRole('region', { name: /rtx 3080 capacity/i })
    if (!gpu2Capacity) throw new Error('Expected carrack GPU 2 capacity region')

    expect(within(gpu2Capacity).getByRole('button', { name: /qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    expect(within(rtx3080Capacity).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'separate' }))

    const restoredGpu2Capacity = within(getCarrackSection()).getAllByRole('region', {
      name: /rtx 6000 pro capacity/i
    })[1]
    const restoredRtx3080Capacity = within(getCarrackSection()).getByRole('region', { name: /rtx 3080 capacity/i })
    if (!restoredGpu2Capacity) throw new Error('Expected restored carrack GPU 2 capacity region')

    expect(within(restoredGpu2Capacity).getByRole('button', { name: /qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    expect(within(restoredRtx3080Capacity).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
  })

  it('saves dirty changes and reverts back to the last saved configuration', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const saveButton = screen.getByRole('button', { name: /save config/i })
    const revertButton = screen.getByRole('button', { name: /revert/i })

    expect(saveButton).toHaveAttribute('aria-keyshortcuts', 'Control+S')
    expect(revertButton).toHaveAttribute('aria-keyshortcuts', 'Control+X')
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'No changes to save')

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    expect(saveButton).toBeEnabled()

    const saveEvent = await dispatchShortcut('s', { ctrlKey: true })
    expect(saveEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'No changes to save')

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()
    expect(saveButton).toBeEnabled()
    await openTomlOutput(user)
    expect(countTomlOccurrences('[models.hardware]')).toBe(0)

    const revertEvent = await dispatchShortcut('x', { ctrlKey: true })
    expect(revertEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    expect(countTomlOccurrences('[models.hardware]')).toBe(0)
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
  })

  it('preserves dirty edits when refreshed configuration data arrives', async () => {
    const user = userEvent.setup()
    const refreshedData: ConfigurationHarnessData = {
      ...CONFIGURATION_HARNESS,
      nodes: CONFIGURATION_HARNESS.nodes.map((node) =>
        node.id === 'carrack'
          ? {
              ...node,
              gpus: node.gpus.map((gpu) => ({ ...gpu, reservedGB: (gpu.reservedGB ?? 0) + 1 }))
            }
          : node
      )
    }

    const { rerender } = render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    expect(within(getCarrackSection()).getByRole('radio', { name: 'pooled' })).toBeChecked()
    expect(screen.getByRole('button', { name: /save config/i })).toBeEnabled()

    rerender(<ConfigurationPage data={refreshedData} initialTab="local-deployment" enableNavigationBlocker={false} />)

    expect(within(getCarrackSection()).getByRole('radio', { name: 'pooled' })).toBeChecked()
    expect(screen.getByRole('button', { name: /save config/i })).toBeEnabled()
  })

  it('tracks configuration history with Ctrl+Z and Ctrl+R', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const undoButton = screen.getByRole('button', { name: /undo/i })
    const redoButton = screen.getByRole('button', { name: /redo/i })

    expect(undoButton).toHaveAttribute('aria-keyshortcuts', 'Control+Z')
    expect(redoButton).toHaveAttribute('aria-keyshortcuts', 'Control+R')

    await user.keyboard('{ArrowDown}')
    const contextEvent = await dispatchShortcut('ArrowRight', { altKey: true })
    expect(contextEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )
    expect(undoButton).toBeEnabled()

    const undoEvent = await dispatchShortcut('z', { ctrlKey: true })
    expect(undoEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '16,384 ctx'
    )
    expect(redoButton).toBeEnabled()

    const redoEvent = await dispatchShortcut('r', { ctrlKey: true })
    expect(redoEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await openTomlOutput(user)
    expect(countTomlOccurrences('[models.hardware]')).toBe(0)

    await dispatchShortcut('z', { ctrlKey: true })
    expect(countTomlOccurrences('[models.hardware]')).toBe(4)

    await dispatchShortcut('r', { ctrlKey: true })
    expect(countTomlOccurrences('[models.hardware]')).toBe(0)
  })

  it('does not consume plain s when the selected node is already separate', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const shortcutEvent = await dispatchShortcut('s')

    expect(shortcutEvent.defaultPrevented).toBe(false)
    await openTomlOutput(user)
    expect(countTomlOccurrences('[models.hardware]')).toBe(4)
  })
})

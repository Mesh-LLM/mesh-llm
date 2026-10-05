import { describe, expect, it } from 'vitest'
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChatTab } from '@/features/chat/pages/ChatTab'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import { ConfigurationTab } from '@/features/configuration/pages/ConfigurationTab'
import { APP_STORAGE_KEYS, CFG_NODES, INITIAL_ASSIGNS } from '@/features/app-tabs/data'
import { FeatureFlagProvider } from '@/lib/feature-flags'
import { createMockDataTransfer, expectTomlOccurrences, render } from './app-tabs-test-support'

describe('chat and configuration app tabs', () => {
  it('renders chat and opens transparency from a message', async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(
      APP_STORAGE_KEYS.featureFlagOverrides,
      JSON.stringify({ chat: { transparencyTab: true } })
    )

    render(
      <FeatureFlagProvider>
        <ChatTab />
      </FeatureFlagProvider>
    )

    await user.click(screen.getByRole('button', { name: /inspect transparency/i }))
    expect(screen.getByText(/inbound route/i)).toBeInTheDocument()
    expect(screen.getByText(/link healthy/i)).toBeInTheDocument()
  })

  it('switches chat conversations from the sidebar', async () => {
    const user = userEvent.setup()
    render(<ChatTab />)
    expect(screen.getByText(/newsletter about local ai/i)).toBeInTheDocument()
    expect(screen.queryByText(/pooled placement plan/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /^model capacity draft/i }))
    expect(screen.getByText(/pooled placement plan/i)).toBeInTheDocument()
    expect(screen.getByText(/use pooled placement on perseus.local/i)).toBeInTheDocument()
    expect(screen.queryByText(/newsletter about local ai/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /^routing latency notes/i }))
    expect(screen.getByText(/newsletter about local ai/i)).toBeInTheDocument()
    expect(screen.queryByText(/pooled placement plan/i)).not.toBeInTheDocument()
  })

  it('renders configuration controls and keeps placement reflected in TOML output', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const configurationHeading = screen.getByRole('heading', { name: 'Configuration', level: 1 })
    expect(configurationHeading).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /toml output/i })).toBeInTheDocument()
    expect(screen.getByText('⌫')).toBeInTheDocument()
    expect(screen.getAllByText(/placement/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/add model/i).length).toBeGreaterThan(0)
    const configurationHeader = configurationHeading.closest('header')
    const nodeRail = screen.getByRole('navigation', { name: /configuration nodes/i })
    if (!configurationHeader) throw new Error('Expected configuration header')
    const keyboardShortcuts = within(nodeRail).getByRole('region', { name: /keyboard shortcuts/i })
    expect(within(keyboardShortcuts).queryByText('Keyboard:')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Navigate')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Toggle Section')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('␣')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Selected Model')).toBeInTheDocument()
    expect(within(keyboardShortcuts).queryByText('Adjust')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Actions')).toBeInTheDocument()
    const keyboardLegendText = keyboardShortcuts.textContent ?? ''
    expect(keyboardLegendText.indexOf('Select Model')).toBeLessThan(keyboardLegendText.indexOf('Selected Model'))
    expect(keyboardLegendText.indexOf('First/Last Model')).toBeLessThan(keyboardLegendText.indexOf('Selected Model'))
    expect(within(keyboardShortcuts).getByText('Adjust Context')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Jump Context')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Move GPU')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Add model')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Toggle Placement')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Undo')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Redo')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Save config')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Revert')).toBeInTheDocument()
    expect(keyboardLegendText.indexOf('Actions')).toBeLessThan(keyboardLegendText.indexOf('Add model'))
    expect(keyboardLegendText.indexOf('Add model')).toBeLessThan(keyboardLegendText.indexOf('Undo'))
    expect(within(keyboardShortcuts).queryByText('Alt')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).queryByText('Ctrl')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).queryByText('Shift')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).queryByText('Tab')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('⇥')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getAllByText('⌥')).toHaveLength(2)
    expect(within(keyboardShortcuts).getAllByText('⌃')).toHaveLength(4)
    expect(within(keyboardShortcuts).getAllByText('⇧')).toHaveLength(3)
    expect(within(keyboardShortcuts).getByText('Z')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('R')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getAllByText('S').length).toBeGreaterThan(0)
    expect(within(keyboardShortcuts).getByText('X')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Selected model')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('⌫')).toBeInTheDocument()
    expect(within(configurationHeader).queryByRole('region', { name: /keyboard shortcuts/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /undo/i })).toHaveAttribute('aria-keyshortcuts', 'Control+Z')
    expect(screen.getByRole('button', { name: /redo/i })).toHaveAttribute('aria-keyshortcuts', 'Control+R')
    expect(screen.getByRole('button', { name: /revert/i })).toHaveAttribute('aria-keyshortcuts', 'Control+X')
    expect(screen.getByRole('button', { name: /save config/i })).toHaveAttribute('aria-keyshortcuts', 'Control+S')
    expect(screen.getByRole('button', { name: /save config/i })).toBeDisabled()
    expect(buildTOML(CFG_NODES, INITIAL_ASSIGNS)).toContain('[models.hardware]')
    expect(screen.getByRole('button', { name: /remove llama-3\.3-70b-q4_k_m/i })).toBeInTheDocument()

    await user.click(configurationHeading)
    expect(screen.queryByRole('button', { name: /remove llama-3\.3-70b-q4_k_m/i })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /qwen3-4b-q4_k_m, 2\.6 gb weights/i }))
    expect(screen.getByRole('button', { name: /remove qwen3-4b-q4_k_m/i })).toBeInTheDocument()

    const carrackKeyboardTarget = screen.getByRole('button', {
      name: /collapse carrack\. use up and down arrows to select gpu slots/i
    })
    expect(carrackKeyboardTarget).toHaveTextContent('▾')
    expect(carrackKeyboardTarget).not.toHaveTextContent(/carrack/i)
    expect(carrackKeyboardTarget).not.toHaveClass('focus-visible:outline-accent')
    const carrackSection = carrackKeyboardTarget.closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')
    expect(carrackSection).toHaveAttribute('data-config-node-selected', 'true')
    expect(carrackSection.className).not.toContain('shadow-[0_0_0_1px_var(--color-accent)]')
    carrackKeyboardTarget.focus()
    expect(carrackKeyboardTarget).toHaveFocus()
    expect(screen.queryByRole('button', { name: /remove qwen3\.5-27b-ud-q4_k_xl/i })).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'perseus.local' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'triton.lab' })).toBeInTheDocument()
    expect(screen.getByText('Peers')).toBeInTheDocument()
    expect(screen.getByText('read-only')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i }))
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    await user.keyboard('{Alt>}{ArrowRight}{/Alt}')
    expect(
      screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights, 0\.4 gb context cache/i })
    ).toHaveTextContent('17,408 ctx')
    await user.keyboard('{Alt>}{ArrowLeft}{/Alt}')
    expect(
      screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights, 0\.4 gb context cache/i })
    ).toHaveTextContent('16,384 ctx')
    await user.keyboard('{Alt>}{Shift>}{ArrowRight}{/Shift}{/Alt}')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '32,768 ctx'
    )
    await user.keyboard('{Alt>}{Shift>}{ArrowLeft}{/Shift}{/Alt}')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '16,384 ctx'
    )
    await user.keyboard('{Shift>}{ArrowDown}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 3/i })).toBeInTheDocument()
    await user.keyboard('{Shift>}{ArrowUp}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()
    await user.keyboard('{ArrowUp}')
    expect(screen.getByRole('button', { name: /remove llama-3\.3-70b-q4_k_m/i })).toBeInTheDocument()
    await user.keyboard('{ArrowDown}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    await user.keyboard('{Delete}')
    expect(screen.queryByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /remove qwen3\.5-27b-q4_k_m/i })).not.toBeInTheDocument()

    await user.click(document.body)
    expect(screen.queryByRole('button', { name: /remove qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    const perseusSection = screen.getByRole('heading', { name: 'perseus.local' }).closest('section')
    const tritonSection = screen.getByRole('heading', { name: 'triton.lab' }).closest('section')

    if (!perseusSection || !tritonSection) throw new Error('Expected remote context sections')

    expect(screen.getByRole('button', { name: 'Add model to perseus.local' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Add model to triton.lab' })).toBeDisabled()
    expect(within(perseusSection).queryByText('read-only')).not.toBeInTheDocument()
    expect(within(tritonSection).queryByText('read-only')).not.toBeInTheDocument()

    const reservedLane = within(carrackSection).getAllByRole('button', { name: /system reserved space/i })[0]
    await user.click(reservedLane)
    expect(reservedLane).toHaveAttribute('aria-pressed', 'true')
    expect(within(carrackSection).getByRole('heading', { name: /system reserved space/i })).toBeInTheDocument()
    expect(
      within(carrackSection).getByText(/invariant system reserved space and has no configurable settings/i)
    ).toBeInTheDocument()
    expect(within(carrackSection).queryByRole('button', { name: /remove qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    expect(within(perseusSection).getByRole('radio', { name: 'separate' })).toBeDisabled()
    expect(within(perseusSection).getByRole('radio', { name: 'pooled' })).toBeDisabled()
    expect(within(tritonSection).getByRole('radio', { name: 'separate' })).toBeDisabled()
    expect(within(tritonSection).getByRole('radio', { name: 'pooled' })).toBeDisabled()

    const assignedModelDrag = createMockDataTransfer()
    const sameNodeDestination = within(carrackSection).getByRole('region', { name: /rtx 5090 capacity/i })
    const sourceContainer = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })

    fireEvent.dragStart(screen.getByRole('button', { name: /qwen3-4b-q4_k_m, 2\.6 gb weights/i }), {
      dataTransfer: assignedModelDrag
    })
    expect(assignedModelDrag.setData).toHaveBeenCalledWith('text/assign-id', 'a4')
    expect(assignedModelDrag.setData).toHaveBeenCalledWith('text/source-node', 'node-a')
    expect(assignedModelDrag.setData).toHaveBeenCalledWith('text/source-container', '7')
    expect(assignedModelDrag.setData).toHaveBeenCalledWith('application/x-mesh-source-container-node-a-7', 'node-a-7')

    fireEvent.dragEnter(sourceContainer, { dataTransfer: assignedModelDrag })
    fireEvent.dragOver(sourceContainer, { dataTransfer: assignedModelDrag })
    expect(within(sourceContainer).queryByText('Drop to assign')).not.toBeInTheDocument()
    expect(assignedModelDrag.dropEffect).toBe('none')

    fireEvent.drop(sourceContainer, { dataTransfer: assignedModelDrag })
    expect(within(sourceContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()

    fireEvent.dragEnter(sameNodeDestination, { dataTransfer: assignedModelDrag })
    fireEvent.dragOver(sameNodeDestination, { dataTransfer: assignedModelDrag })
    expect(within(sameNodeDestination).getByText('Drop to assign')).toBeInTheDocument()
    expect(assignedModelDrag.dropEffect).toBe('move')

    fireEvent.drop(sameNodeDestination, { dataTransfer: assignedModelDrag })
    await waitFor(() =>
      expect(within(sameNodeDestination).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
    )
    expect(within(sourceContainer).queryByRole('button', { name: /qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('radio', { name: 'pooled' }))
    expect(screen.getByRole('button', { name: /save config/i })).toBeEnabled()
    expect(within(carrackSection).getByRole('radio', { name: 'pooled' })).toBeChecked()
    carrackKeyboardTarget.focus()
    await user.keyboard('s')
    expect(within(carrackSection).getByRole('radio', { name: 'separate' })).toBeChecked()
    await user.keyboard('p')
    expect(within(carrackSection).getByRole('radio', { name: 'pooled' })).toBeChecked()

    const initialToml = buildTOML(CFG_NODES, INITIAL_ASSIGNS)
    expect(initialToml).toContain('version = 1')
    expect(initialToml).toContain('model = "GLM-4.7-Flash-Q4_K_M"')
    expect(initialToml).not.toContain('perseus.local')
    expect(initialToml).not.toContain('triton.lab')
    expect(initialToml).toContain('gpu_id = "cuda:0"')
    expect(initialToml).not.toContain('gpu_index =')
    expect(initialToml).not.toContain('[node]')

    carrackKeyboardTarget.focus()
    await user.keyboard('a')
    expect(screen.getByRole('dialog', { name: 'Model catalog' })).toBeInTheDocument()
    expect(screen.getAllByText('Fits').length).toBeGreaterThan(0)
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: /remove phi-4-mini/i })).toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'llava')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.click(screen.getByRole('button', { name: /llava-next-34b, 22 gb/i }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: /remove llava-next-34b/i })).toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    const dragTransfer = createMockDataTransfer()
    fireEvent.dragStart(screen.getByRole('button', { name: /qwen3-4b-q4_k_m, 2.6 gb/i }), {
      clientX: 12,
      clientY: 12,
      dataTransfer: dragTransfer
    })
    expect(dragTransfer.setData).toHaveBeenCalledWith('text/model', 'qwen4')
    expect(dragTransfer.setDragImage).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      expect.any(Number),
      expect.any(Number)
    )
    await user.click(screen.getByRole('button', { name: 'Close' }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.click(screen.getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument()
  })

  it('highlights the selected GPU container and targets it for catalog Enter adds', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')

    const qwen4Button = within(carrackSection).getByRole('button', { name: /qwen3-4b-q4_k_m, 2\.6 gb weights/i })
    await user.click(qwen4Button)

    let rtx3080Capacity = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    const selected3080Container = rtx3080Capacity.closest('[data-config-container-selected="true"]')
    if (!(selected3080Container instanceof HTMLElement)) throw new Error('Expected selected RTX 3080 container')
    expect(selected3080Container).toContainElement(qwen4Button)

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 7/i })).toBeInTheDocument()
    rtx3080Capacity = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    expect(within(rtx3080Capacity).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
    expect(rtx3080Capacity.closest('[data-config-container-selected="true"]')).toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'llava')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove llava-next-34b from gpu 2/i })).toBeInTheDocument()
    expect(within(rtx3080Capacity).queryByRole('button', { name: /llava-next-34b/i })).not.toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'mixtral')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')

    expect(screen.getByRole('textbox', { name: 'Command bar search' })).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(/mixtral-8x22b does not fit on any gpu in carrack/i)
    expect(screen.queryByRole('button', { name: /remove mixtral-8x22b/i })).not.toBeInTheDocument()
  })

  it('uses the clicked GPU container as the catalog add target', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')

    const rtx6000CapacityRegions = within(carrackSection).getAllByRole('region', { name: /rtx 6000 pro capacity/i })
    const gpu3Capacity = rtx6000CapacityRegions[2]
    if (!gpu3Capacity) throw new Error('Expected carrack GPU 3 capacity region')

    await user.click(gpu3Capacity)

    const selectedGpu3Container = gpu3Capacity.closest('[data-config-container-selected="true"]')
    if (!(selectedGpu3Container instanceof HTMLElement))
      throw new Error('Expected clicked GPU 3 container to be selected')

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 3/i })).toBeInTheDocument()
    expect(within(gpu3Capacity).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
  })

  it('restores separate GPU assignments after previewing pooled placement', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')

    const gpu2Capacity = within(carrackSection).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[1]
    const rtx3080Capacity = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    if (!gpu2Capacity) throw new Error('Expected carrack GPU 2 capacity region')

    expect(within(gpu2Capacity).getByRole('button', { name: /qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    expect(within(rtx3080Capacity).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('radio', { name: 'pooled' }))
    await user.click(within(carrackSection).getByRole('radio', { name: 'separate' }))

    const restoredGpu2Capacity = within(carrackSection).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[1]
    const restoredRtx3080Capacity = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    if (!restoredGpu2Capacity) throw new Error('Expected restored carrack GPU 2 capacity region')

    expect(within(restoredGpu2Capacity).getByRole('button', { name: /qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    expect(within(restoredRtx3080Capacity).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
  })

  it('keeps separate placement snapshots aligned with undo history', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const dispatchUndo = async () => {
      const event = new KeyboardEvent('keydown', { key: 'z', ctrlKey: true, bubbles: true, cancelable: true })

      await act(async () => {
        window.dispatchEvent(event)
      })
      expect(event.defaultPrevented).toBe(true)
    }
    const getCarrackSection = () => {
      const section = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
      if (!section) throw new Error('Expected carrack section')
      return section
    }
    const getRtx3080Capacity = () => within(getCarrackSection()).getByRole('region', { name: /rtx 3080 capacity/i })

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'separate' }))
    await user.click(within(getRtx3080Capacity()).getByRole('button', { name: /qwen3-4b-q4_k_m/i }))
    await user.keyboard('{Shift>}{ArrowUp}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3-4b-q4_k_m from gpu 6/i })).toBeInTheDocument()

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await dispatchUndo()
    await dispatchUndo()
    await dispatchUndo()
    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'separate' }))

    expect(within(getRtx3080Capacity()).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
  })

  it('enables save only for dirty configuration changes and supports the save shortcut', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const saveButton = screen.getByRole('button', { name: /save config/i })
    const revertButton = screen.getByRole('button', { name: /revert/i })
    expect(saveButton).toHaveAttribute('aria-keyshortcuts', 'Control+S')
    expect(revertButton).toHaveAttribute('aria-keyshortcuts', 'Control+X')
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'No changes to save')

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')
    await user.click(within(carrackSection).getByRole('radio', { name: 'pooled' }))
    expect(saveButton).toBeEnabled()

    const saveEvent = new KeyboardEvent('keydown', { key: 's', ctrlKey: true, bubbles: true, cancelable: true })
    await act(async () => {
      window.dispatchEvent(saveEvent)
    })
    expect(saveEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'No changes to save')
  })

  it('reverts dirty configuration changes with the revert shortcut', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const saveButton = screen.getByRole('button', { name: /save config/i })
    const getCarrackSection = () => {
      const section = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
      if (!section) throw new Error('Expected carrack section')
      return section
    }

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    expect(saveButton).toBeEnabled()
    const revertEvent = new KeyboardEvent('keydown', { key: 'x', ctrlKey: true, bubbles: true, cancelable: true })
    await act(async () => {
      window.dispatchEvent(revertEvent)
    })
    expect(revertEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    await expectTomlOccurrences(user, '[models.hardware]', 4)

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await expectTomlOccurrences(user, '[models.hardware]', 0)
    const saveEvent = new KeyboardEvent('keydown', { key: 's', ctrlKey: true, bubbles: true, cancelable: true })
    await act(async () => {
      window.dispatchEvent(saveEvent)
    })
    expect(saveEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()
    expect(saveButton).toBeEnabled()
    await expectTomlOccurrences(user, '[models.hardware]', 0)
    const revertToSavedEvent = new KeyboardEvent('keydown', {
      key: 'x',
      ctrlKey: true,
      bubbles: true,
      cancelable: true
    })
    await act(async () => {
      window.dispatchEvent(revertToSavedEvent)
    })
    expect(revertToSavedEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    await expectTomlOccurrences(user, '[models.hardware]', 0)
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
  })

  it('tracks full configuration history with Ctrl+Z and Ctrl+R', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const undoButton = screen.getByRole('button', { name: /undo/i })
    const redoButton = screen.getByRole('button', { name: /redo/i })
    const dispatchShortcut = async (key: string) => {
      const event = new KeyboardEvent('keydown', { key, ctrlKey: true, bubbles: true, cancelable: true })

      await act(async () => {
        window.dispatchEvent(event)
      })
      expect(event.defaultPrevented).toBe(true)
    }

    expect(undoButton).toHaveAttribute('aria-keyshortcuts', 'Control+Z')
    expect(redoButton).toHaveAttribute('aria-keyshortcuts', 'Control+R')

    await user.keyboard('{ArrowDown}')
    await user.keyboard('{Alt>}{ArrowRight}{/Alt}')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )
    expect(undoButton).toBeEnabled()

    await dispatchShortcut('z')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '16,384 ctx'
    )
    expect(redoButton).toBeEnabled()

    await dispatchShortcut('r')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )

    await user.keyboard('{Shift>}{ArrowDown}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 3/i })).toBeInTheDocument()
    await dispatchShortcut('z')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()
    await dispatchShortcut('r')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 3/i })).toBeInTheDocument()

    const getCarrackSection = () => {
      const section = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
      if (!section) throw new Error('Expected carrack section')
      return section
    }

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await expectTomlOccurrences(user, '[models.hardware]', 0)
    await dispatchShortcut('z')
    await expectTomlOccurrences(user, '[models.hardware]', 4)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toBeInTheDocument()
    await dispatchShortcut('r')
    await expectTomlOccurrences(user, '[models.hardware]', 0)

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()
    await dispatchShortcut('z')
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
    await dispatchShortcut('r')
    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /remove phi-4-mini/i }))
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
    await dispatchShortcut('z')
    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()
    await dispatchShortcut('r')
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
  })

  it('tracks drag and drop configuration history with Ctrl+Z and Ctrl+R', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const dispatchShortcut = async (key: string) => {
      const event = new KeyboardEvent('keydown', { key, ctrlKey: true, bubbles: true, cancelable: true })

      await act(async () => {
        window.dispatchEvent(event)
      })
      expect(event.defaultPrevented).toBe(true)
    }

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected configuration section')

    const sourceContainer = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    const destinationContainer = within(carrackSection).getByRole('region', { name: /rtx 5090 capacity/i })
    const assignedModelDrag = createMockDataTransfer()

    fireEvent.dragStart(within(sourceContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i }), {
      dataTransfer: assignedModelDrag
    })
    fireEvent.dragEnter(destinationContainer, { dataTransfer: assignedModelDrag })
    fireEvent.dragOver(destinationContainer, { dataTransfer: assignedModelDrag })
    fireEvent.drop(destinationContainer, { dataTransfer: assignedModelDrag })
    await waitFor(() =>
      expect(within(destinationContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
    )
    expect(within(sourceContainer).queryByRole('button', { name: /qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    await dispatchShortcut('z')
    await waitFor(() =>
      expect(within(sourceContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
    )
    expect(within(destinationContainer).queryByRole('button', { name: /qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()
    await dispatchShortcut('r')
    await waitFor(() =>
      expect(within(destinationContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
    )
    expect(within(sourceContainer).queryByRole('button', { name: /qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    const catalogDrag = createMockDataTransfer()
    fireEvent.dragStart(screen.getByRole('button', { name: /phi-4-mini, .* gb, .* context, fits/i }), {
      clientX: 12,
      clientY: 12,
      dataTransfer: catalogDrag
    })
    fireEvent.dragEnter(sourceContainer, { dataTransfer: catalogDrag })
    fireEvent.dragOver(sourceContainer, { dataTransfer: catalogDrag })
    fireEvent.drop(sourceContainer, { dataTransfer: catalogDrag })
    await waitFor(() =>
      expect(within(sourceContainer).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
    )

    await dispatchShortcut('z')
    await waitFor(() =>
      expect(within(sourceContainer).queryByRole('button', { name: /phi-4-mini/i })).not.toBeInTheDocument()
    )
    await dispatchShortcut('r')
    await waitFor(() =>
      expect(within(sourceContainer).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
    )
  })
})

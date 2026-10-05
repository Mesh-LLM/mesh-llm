import { beforeEach, describe, expect, it } from 'vitest'
import { act, fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DefaultsTab } from '@/features/configuration/components/DefaultsTab'
import type { ConfigurationDefaultsHarnessData } from '@/features/app-tabs/types'
import {
  CONFIGURATION_DEFAULTS,
  dependencyData,
  disabledInfoTrigger,
  previewSource,
  renderDefaultsTab,
  schemaDrivenControlData,
  SHOW_ADVANCED_STORAGE_KEY,
  settingInfoTrigger,
  settingsRow,
  slotDependencyData,
  slotDependencySettings
} from './DefaultsTab-test-support'

describe('DefaultsTab dependency and schema controls', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })
  it('keeps integrated dependency disable states and explanations working across real dependency pairs', async () => {
    const user = userEvent.setup()

    window.localStorage.setItem(SHOW_ADVANCED_STORAGE_KEY, 'true')

    const { rerender } = renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        'speculation-mode': 'off'
      }
    })

    const draftSelectionPolicyRow = settingsRow('Default draft selection policy')
    expect(draftSelectionPolicyRow).toHaveAttribute('data-settings-row-disabled', 'true')
    expect(screen.queryAllByText('Requires speculation-mode = draft')).toHaveLength(0)
    expect(screen.getByText('Prefill chunk size').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.queryByText('Requires prefill-chunking = fixed')).not.toBeInTheDocument()
    expect(screen.getByText('Prefill chunk schedule').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.queryByText('Requires prefill-chunking = schedule')).not.toBeInTheDocument()
    expect(screen.getByText('Mirostat entropy').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.getByText('Mirostat learning rate').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.queryAllByText('Requires mirostat-mode = 1 or 2')).toHaveLength(0)
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('prefill_chunk_size')
    expect(previewSource().value).not.toContain('prefill_chunk_schedule')
    expect(previewSource().value).not.toContain('mirostat_entropy')

    const draftSelectionPolicyTrigger = disabledInfoTrigger(draftSelectionPolicyRow)

    await user.hover(draftSelectionPolicyTrigger)
    expect(await screen.findByText('Requires speculation-mode = draft', { selector: 'div' })).toBeInTheDocument()
    await user.unhover(draftSelectionPolicyTrigger)

    rerender(
      <DefaultsTab
        data={CONFIGURATION_DEFAULTS}
        values={{
          'speculation-mode': 'draft',
          'mirostat-mode': '2',
          'prefill-chunking': 'fixed'
        }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    expect(screen.getByText('Default draft selection policy').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryAllByText('Requires speculation-mode = draft')).toHaveLength(0)
    expect(screen.getByText('Prefill chunk size').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryByText('Requires prefill-chunking = fixed')).not.toBeInTheDocument()
    expect(screen.getByText('Prefill chunk schedule').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.queryByText('Requires prefill-chunking = schedule')).not.toBeInTheDocument()
    expect(screen.getByText('Mirostat entropy').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.getByText('Mirostat learning rate').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryAllByText('Requires mirostat-mode = 1 or 2')).toHaveLength(0)
    expect(previewSource().value).toContain('mirostat_mode = 2')
    expect(previewSource().value).toContain('prefill_chunking = "fixed"')
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('prefill_chunk_size')
    expect(previewSource().value).not.toContain('mirostat_entropy')

    rerender(
      <DefaultsTab
        data={CONFIGURATION_DEFAULTS}
        values={{
          'speculation-mode': 'draft',
          'mirostat-mode': '2',
          'prefill-chunking': 'schedule',
          'prefill-chunk-schedule': '128,256'
        }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    expect(screen.getByText('Prefill chunk size').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.queryByText('Requires prefill-chunking = fixed')).not.toBeInTheDocument()
    expect(screen.getByText('Prefill chunk schedule').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryByText('Requires prefill-chunking = schedule')).not.toBeInTheDocument()
    expect(previewSource().value).toContain('prefill_chunk_schedule = "128,256"')
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('prefill_chunk_size')
    expect(previewSource().value).not.toContain('mirostat_entropy')
  })

  it('disables dependent settings until their dependency is satisfied behind inline info triggers', async () => {
    const user = userEvent.setup()

    const { rerender } = renderDefaultsTab({ data: dependencyData })

    const draftSelectionPolicyRow = settingsRow('Draft selection policy')
    const mirostatEntropyRow = settingsRow('Mirostat entropy')

    expect(draftSelectionPolicyRow).toHaveAttribute('data-settings-row-disabled', 'true')
    expect(mirostatEntropyRow).toHaveAttribute('data-settings-row-disabled', 'true')
    expect(screen.queryByText('Requires speculation-mode = draft_model')).not.toBeInTheDocument()
    expect(screen.queryByText('Requires mirostat-mode = 1 or 2')).not.toBeInTheDocument()
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('mirostat_entropy')

    const draftSelectionPolicyTrigger = disabledInfoTrigger(draftSelectionPolicyRow)

    await user.hover(draftSelectionPolicyTrigger)
    expect(await screen.findByText('Requires speculation-mode = draft_model', { selector: 'div' })).toBeInTheDocument()
    await user.unhover(draftSelectionPolicyTrigger)

    await act(async () => {
      disabledInfoTrigger(mirostatEntropyRow).focus()
    })
    expect(await screen.findByText('Requires mirostat-mode = 1 or 2', { selector: 'div' })).toBeInTheDocument()

    rerender(
      <DefaultsTab
        data={dependencyData}
        values={{ 'speculation-mode': 'draft_model', 'mirostat-mode': '2' }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    expect(screen.getByText('Draft selection policy').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryByText('Requires speculation-mode = draft_model')).not.toBeInTheDocument()
    expect(screen.getByText('Mirostat entropy').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryByText('Requires mirostat-mode = 1 or 2')).not.toBeInTheDocument()
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('mirostat_entropy')
  })

  it('keeps disabled slot-meter controls inert', () => {
    const onSettingValueChange = vi.fn()

    renderDefaultsTab({ data: slotDependencyData, onSettingValueChange })

    const slotRow = settingsRow('Default slots / parallel requests')
    expect(slotRow).toHaveAttribute('data-settings-row-disabled', 'true')
    expect(screen.getByRole('radio', { name: '4 slots' })).toBeChecked()
    expect(within(slotRow).queryByRole('spinbutton')).not.toBeInTheDocument()
    expect(within(slotRow).queryByRole('slider')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('radio', { name: '12 slots' }))
    fireEvent.pointerDown(screen.getByTestId('defaults-slot-meter'), {
      buttons: 1,
      clientX: 200,
      pointerId: 1
    })

    expect(onSettingValueChange).not.toHaveBeenCalled()
    expect(screen.getByRole('radio', { name: '4 slots' })).toBeChecked()
  })

  it('sizes slot-meter options and pointer selection from schema bounds', () => {
    const onSettingValueChange = vi.fn()
    const boundedSlotData = {
      ...slotDependencyData,
      settings: slotDependencySettings.map((setting) =>
        setting.id === 'parallel-slots'
          ? {
              ...setting,
              control: {
                ...setting.control,
                value: '3',
                min: 3,
                max: 6
              }
            }
          : setting
      )
    } satisfies ConfigurationDefaultsHarnessData

    renderDefaultsTab({
      data: boundedSlotData,
      values: { 'speculation-mode': 'draft_model' },
      onSettingValueChange
    })

    expect(screen.getByRole('radio', { name: '3 slots' })).toBeChecked()
    expect(screen.getByRole('radio', { name: '6 slots' })).toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: '2 slots' })).not.toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: '7 slots' })).not.toBeInTheDocument()

    const slotMeter = screen.getByTestId('defaults-slot-meter')
    vi.spyOn(slotMeter, 'getBoundingClientRect').mockReturnValue({
      bottom: 10,
      height: 10,
      left: 0,
      right: 400,
      top: 0,
      width: 400,
      x: 0,
      y: 0,
      toJSON: () => ({})
    })

    fireEvent.pointerDown(slotMeter, {
      buttons: 1,
      clientX: 100,
      pointerId: 1
    })

    expect(onSettingValueChange).toHaveBeenCalledWith('parallel-slots', '4')
  })

  it('renders schema-driven controls with bounds, hints, runtime notes, disabled framing, arrays, objects, and conflicts', async () => {
    const user = userEvent.setup()

    const { rerender } = renderDefaultsTab({ data: schemaDrivenControlData })

    expect(screen.getByRole('slider', { name: 'Context window' })).toHaveValue('4')
    expect(screen.queryByRole('spinbutton', { name: 'Context window' })).not.toBeInTheDocument()
    expect(screen.queryByText('Min 1 · Max 8 · Step 1 · Unit slots')).not.toBeInTheDocument()
    expect(screen.queryByText('Accepted: auto, pinned')).not.toBeInTheDocument()

    expect(screen.getByRole('textbox', { name: 'Projector path' })).toBeInTheDocument()
    expect(screen.queryByText('Paths are resolved on the machine running this MeshLLM node.')).not.toBeInTheDocument()

    expect(screen.getByRole('textbox', { name: 'Projector URL' })).toBeInTheDocument()
    expect(screen.queryByText('URL hint: enter a full URL including protocol.')).not.toBeInTheDocument()

    expect(screen.getByRole('radio', { name: 'CUDA 0' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'CUDA 1' })).toBeDisabled()
    expect(screen.queryByText('Unavailable: CUDA 1 — Reserved by another runtime')).not.toBeInTheDocument()

    const unavailableBackendRow = settingsRow('Unavailable backend')
    expect(unavailableBackendRow).toHaveAttribute('data-settings-row-disabled', 'true')
    expect(screen.queryByText('No native backend was detected.')).not.toBeInTheDocument()
    expect(screen.queryByText('Omit when disabled')).not.toBeInTheDocument()
    expect(screen.queryByText('The current value is kept in config but cannot be edited here.')).not.toBeInTheDocument()

    const unavailableBackendTrigger = disabledInfoTrigger(unavailableBackendRow)

    await user.hover(unavailableBackendTrigger)
    expect(await screen.findByText('No native backend was detected.', { selector: 'div' })).toBeInTheDocument()
    expect(
      await screen.findByText('The current value is kept in config but cannot be edited here.', {
        selector: 'div'
      })
    ).toBeInTheDocument()
    await user.unhover(unavailableBackendTrigger)

    const preservedDeviceRow = settingsRow('Pinned GPU device')
    expect(preservedDeviceRow).toHaveAttribute('data-settings-row-disabled', 'true')
    expect(
      within(preservedDeviceRow).queryByRole('button', { name: 'Reset Pinned GPU device to default' })
    ).not.toBeInTheDocument()
    expect(screen.queryByText('Requires gpu.assignment = pinned')).not.toBeInTheDocument()
    expect(within(preservedDeviceRow).queryByText('Preserve value on save')).not.toBeInTheDocument()

    await act(async () => {
      disabledInfoTrigger(preservedDeviceRow).focus()
    })
    expect(await screen.findByText('Requires gpu.assignment = pinned', { selector: 'div' })).toBeInTheDocument()

    rerender(
      <DefaultsTab
        data={schemaDrivenControlData}
        values={{ 'schema-preserved-device': 'cuda:1' }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    const dirtyPreservedDeviceRow = settingsRow('Pinned GPU device')
    const unavailableInfoTrigger = disabledInfoTrigger(dirtyPreservedDeviceRow)
    const dirtyPreservedReset = within(dirtyPreservedDeviceRow).getByRole('button', {
      name: 'Reset Pinned GPU device to default'
    })
    expect(
      unavailableInfoTrigger.compareDocumentPosition(dirtyPreservedReset) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()

    const arrayControl = screen.getByRole('textbox', { name: 'Allowed peers' })
    expect(arrayControl).toBeInTheDocument()
    expect(
      screen.queryByText('List input: enter one item per line. Saved as a TOML string array.')
    ).not.toBeInTheDocument()
    expect(screen.getByText('peer-a')).toBeInTheDocument()
    expect(screen.getByText('peer-b')).toBeInTheDocument()

    expect(screen.getByRole('textbox', { name: 'Telemetry headers' })).toBeInTheDocument()
    expect(screen.queryByText('Object input: enter a JSON object.')).not.toBeInTheDocument()
    expect(
      screen.queryByText('Conflict: Conflicts with draft_min_tokens values above the configured maximum.')
    ).not.toBeInTheDocument()

    const projectorPathRow = settingsRow('Projector path')
    await user.hover(settingInfoTrigger(projectorPathRow))
    expect(
      await screen.findByText('Paths are resolved on the machine running this MeshLLM node.', {
        selector: 'div'
      })
    ).toBeInTheDocument()
    await user.unhover(settingInfoTrigger(projectorPathRow))

    await act(async () => {
      settingInfoTrigger(settingsRow('Projector URL')).focus()
    })
    expect(
      await screen.findByText('URL hint: enter a full URL including protocol.', { selector: 'div' })
    ).toBeInTheDocument()

    await user.hover(settingInfoTrigger(settingsRow('Allowed peers')))
    expect(
      await screen.findByText('List input: enter one item per line. Saved as a TOML string array.', {
        selector: 'div'
      })
    ).toBeInTheDocument()
    await user.unhover(settingInfoTrigger(settingsRow('Allowed peers')))

    await act(async () => {
      settingInfoTrigger(settingsRow('Telemetry headers')).focus()
    })
    expect(await screen.findByText('Object input: enter a JSON object.', { selector: 'div' })).toBeInTheDocument()

    await user.hover(settingInfoTrigger(settingsRow('Draft pairing mode')))
    expect(
      await screen.findByText('Conflict: Conflicts with draft_min_tokens values above the configured maximum.', {
        selector: 'div'
      })
    ).toBeInTheDocument()
  })

  it('loads and authors schema-defined object array rows', async () => {
    const user = userEvent.setup()
    const onSettingValueChange = vi.fn()

    renderDefaultsTab({ data: schemaDrivenControlData, onSettingValueChange })

    expect(screen.getByRole('textbox', { name: 'Stage 1 Endpoint ID' })).toHaveValue('endpoint-a')
    expect(screen.getByRole('textbox', { name: 'Stage 2 Hostname' })).toHaveValue('worker-b')
    expect(screen.getByRole('spinbutton', { name: 'Stage 2 Layer start' })).toHaveValue(16)

    await user.hover(settingInfoTrigger(settingsRow('Topology stages')))
    expect(
      await screen.findByText('Structured list: add, remove, or reorder entries below.', { selector: 'div' })
    ).toBeInTheDocument()
    expect(
      screen.queryByText('List input: enter one item per line. Saved as a TOML string array.')
    ).not.toBeInTheDocument()
    expect(screen.queryByText('Object input: enter a JSON object.')).not.toBeInTheDocument()
    await user.unhover(settingInfoTrigger(settingsRow('Topology stages')))

    await user.click(screen.getByRole('button', { name: 'Move stage 2 up' }))
    expect(onSettingValueChange).toHaveBeenLastCalledWith(
      'topology-stages',
      '[{"node":{"hostname":"worker-b"},"layer_start":16,"layer_end":32},{"node":{"endpoint_id":"endpoint-a"},"layer_start":0,"layer_end":16}]'
    )

    await user.click(screen.getByRole('button', { name: 'Remove stage 1' }))
    expect(onSettingValueChange).toHaveBeenLastCalledWith(
      'topology-stages',
      '[{"node":{"hostname":"worker-b"},"layer_start":16,"layer_end":32}]'
    )

    await user.click(screen.getByRole('button', { name: 'Add stage' }))
    expect(onSettingValueChange).toHaveBeenLastCalledWith(
      'topology-stages',
      '[{"node":{"endpoint_id":"endpoint-a"},"layer_start":0,"layer_end":16},{"node":{"hostname":"worker-b"},"layer_start":16,"layer_end":32},{"node":{},"layer_start":0,"layer_end":0}]'
    )
  })
})

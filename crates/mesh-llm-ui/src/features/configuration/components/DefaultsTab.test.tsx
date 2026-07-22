import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'
import { DefaultsTab } from '@/features/configuration/components/DefaultsTab'
import type {
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues
} from '@/features/app-tabs/types'
import { env } from '@/lib/env'
import { SETTING_RESET_TOOLTIP } from '@/features/configuration/components/settings/SettingResetButton'

const SHOW_ADVANCED_STORAGE_KEY = `${env.storageNamespace}:configuration-defaults:show-advanced:v1`

const defaultSettings = [
  {
    id: 'runtime-mode',
    categoryId: 'runtime',
    icon: 'cpu',
    label: 'Runtime mode',
    description: 'Controls the standard runtime selection.',
    inheritedLabel: 'Inherited by default placements',
    control: {
      kind: 'choice',
      name: 'runtime_mode',
      value: 'auto',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'manual', label: 'manual' }
      ]
    }
  },
  {
    id: 'advanced-reasoning',
    categoryId: 'advanced',
    icon: 'cog',
    label: 'Reasoning budget',
    description: 'Advanced reasoning control.',
    inheritedLabel: 'Inherited by reasoning-capable placements',
    visibility: 'advanced',
    control: {
      kind: 'range',
      name: 'reasoning_budget',
      value: '128',
      min: 0,
      max: 512,
      step: 32,
      unit: 'tok'
    }
  },
  {
    id: 'advanced-note',
    categoryId: 'advanced',
    icon: 'filter',
    label: 'Advanced note',
    description: 'Extra advanced guidance.',
    inheritedLabel: 'Inherited by advanced defaults',
    visibility: 'advanced',
    control: {
      kind: 'text',
      name: 'advanced_note',
      value: '',
      placeholder: 'Optional note'
    }
  }
] satisfies readonly ConfigurationDefaultsSetting[]

const defaultsData = {
  categories: [
    { id: 'runtime', label: 'Runtime', summary: 'Standard defaults.', help: 'Runtime defaults' },
    { id: 'advanced', label: 'Reasoning', summary: 'Advanced defaults.', help: 'Reasoning defaults' }
  ],
  settings: defaultSettings,
  preview: []
} satisfies ConfigurationDefaultsHarnessData

const dependencySettings = [
  {
    id: 'speculation-mode',
    categoryId: 'speculative-decoding',
    icon: 'brain',
    label: 'Speculation mode',
    description: 'Controls whether draft-model speculation is active.',
    inheritedLabel: 'Inherited by speculative decoding defaults',
    control: {
      kind: 'choice',
      name: 'speculation_mode',
      value: 'off',
      presentation: 'segmented',
      options: [
        { value: 'off', label: 'off' },
        { value: 'draft_model', label: 'draft model' }
      ]
    }
  },
  {
    id: 'draft-selection-policy',
    categoryId: 'speculative-decoding',
    icon: 'filter',
    label: 'Draft selection policy',
    description: 'Chooses the draft selection behavior.',
    inheritedLabel: 'Only available when draft-model speculation is enabled',
    dependsOn: {
      settingId: 'speculation-mode',
      condition: (value: string) => value === 'draft_model'
    },
    control: {
      kind: 'choice',
      name: 'draft_selection_policy',
      value: 'auto',
      presentation: 'toggle',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'manual_only', label: 'Manual only' }
      ]
    }
  },
  {
    id: 'mirostat-mode',
    categoryId: 'request-defaults',
    icon: 'brain',
    label: 'Mirostat mode',
    description: 'Controls whether Mirostat is active.',
    inheritedLabel: 'Inherited by request defaults',
    control: {
      kind: 'choice',
      name: 'mirostat_mode',
      value: 'disabled',
      presentation: 'segmented',
      options: [
        { value: 'disabled', label: 'disabled' },
        { value: '1', label: '1' },
        { value: '2', label: '2' }
      ]
    }
  },
  {
    id: 'mirostat-entropy',
    categoryId: 'request-defaults',
    icon: 'gauge',
    label: 'Mirostat entropy',
    description: 'Depends on the Mirostat mode.',
    inheritedLabel: 'Only available when Mirostat is enabled',
    dependsOn: {
      settingId: 'mirostat-mode',
      condition: (value: string) => value !== 'disabled'
    },
    control: {
      kind: 'range',
      name: 'mirostat_entropy',
      value: '5',
      min: 0.1,
      max: 10,
      step: 0.1
    }
  }
] satisfies readonly ConfigurationDefaultsSetting[]

const dependencyData = {
  categories: [
    {
      id: 'speculative-decoding',
      label: 'Speculative Decoding',
      summary: 'Speculative defaults.',
      help: 'Speculative defaults'
    },
    {
      id: 'request-defaults',
      label: 'Request Defaults',
      summary: 'Sampling defaults.',
      help: 'Sampling defaults'
    }
  ],
  settings: dependencySettings,
  preview: []
} satisfies ConfigurationDefaultsHarnessData

const schemaDrivenControlSettings = [
  {
    id: 'schema-number',
    categoryId: 'runtime',
    icon: 'gauge',
    label: 'Context window',
    description: 'Schema numeric control.',
    inheritedLabel: 'Inherited by runtime defaults',
    valueSchema: { kind: 'integer' },
    controlBehavior: {
      numeric: { min: 1, max: 8, step: 1, unit: 'slots' }
    },
    control: {
      kind: 'range',
      name: 'context_window',
      value: '4',
      min: 1,
      max: 8,
      step: 1,
      unit: 'slots'
    }
  },
  {
    id: 'schema-path',
    categoryId: 'multimodal',
    icon: 'folder',
    label: 'Projector path',
    description: 'Schema path control.',
    inheritedLabel: 'Inherited by multimodal defaults',
    valueSchema: { kind: 'path' },
    control: {
      kind: 'text',
      name: 'projector_path',
      value: '',
      placeholder: './models/projector.gguf'
    }
  },
  {
    id: 'schema-url',
    categoryId: 'multimodal',
    icon: 'server',
    label: 'Projector URL',
    description: 'Schema URL control.',
    inheritedLabel: 'Inherited by multimodal defaults',
    valueSchema: { kind: 'url' },
    control: {
      kind: 'text',
      name: 'projector_url',
      value: '',
      placeholder: 'https://example.com/projector.gguf'
    }
  },
  {
    id: 'schema-runtime-choice',
    categoryId: 'runtime',
    icon: 'cpu',
    label: 'GPU device',
    description: 'Runtime choice control.',
    inheritedLabel: 'Inherited by runtime defaults',
    valueSchema: { kind: 'string' },
    controlBehavior: {
      options_source: 'runtime_gpus',
      write_policy: 'preserve_existing'
    },
    controlState: {
      enabled: true,
      source: 'runtime',
      write_policy: 'preserve_existing',
      options: [
        {
          value: { kind: 'string', value: 'cuda:0' },
          label: 'CUDA 0',
          note: '31.8 GiB VRAM',
          disabled: false,
          source: 'runtime_gpus'
        },
        {
          value: { kind: 'string', value: 'cuda:1' },
          label: 'CUDA 1',
          reason: 'Reserved by another runtime',
          disabled: true,
          source: 'runtime_gpus'
        }
      ]
    },
    control: {
      kind: 'choice',
      name: 'gpu_device',
      value: 'cuda:0',
      presentation: 'segmented',
      options: [{ value: 'cuda:0', label: 'CUDA 0' }]
    }
  },
  {
    id: 'schema-disabled',
    categoryId: 'runtime',
    icon: 'cpu',
    label: 'Unavailable backend',
    description: 'Disabled runtime control.',
    inheritedLabel: 'Inherited by runtime defaults',
    valueSchema: { kind: 'string' },
    controlBehavior: {
      options_source: 'runtime_native_backends',
      write_policy: 'omit_when_disabled'
    },
    controlState: {
      enabled: false,
      reason: 'No native backend was detected.',
      note: 'The current value is kept in config but cannot be edited here.',
      source: 'runtime',
      write_policy: 'omit_when_disabled'
    },
    control: {
      kind: 'text',
      name: 'native_backend',
      value: 'metal'
    }
  },
  {
    id: 'schema-pinned-assignment',
    categoryId: 'runtime',
    canonicalPath: 'gpu.assignment',
    icon: 'cpu',
    label: 'GPU assignment',
    description: 'Controls whether device selection is automatic or pinned.',
    inheritedLabel: 'Inherited by runtime defaults',
    valueSchema: { kind: 'enum', values: ['auto', 'pinned'] },
    control: {
      kind: 'choice',
      name: 'gpu_assignment',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'pinned', label: 'pinned' }
      ]
    }
  },
  {
    id: 'schema-preserved-device',
    categoryId: 'runtime',
    canonicalPath: 'defaults.hardware.device',
    icon: 'cpu',
    label: 'Pinned GPU device',
    description: 'Only editable when GPU assignment is pinned.',
    inheritedLabel: 'Inherited by runtime defaults',
    mutability: 'restart-required',
    valueSchema: { kind: 'string' },
    controlBehavior: {
      enable_when: [
        {
          path: { segments: ['gpu', 'assignment'] },
          operator: 'equals',
          values: [{ kind: 'string', value: 'pinned' }]
        }
      ],
      write_policy: 'preserve_existing'
    },
    control: {
      kind: 'text',
      name: 'device',
      value: 'cuda:0'
    },
    baselineValue: ''
  },
  {
    id: 'schema-array',
    categoryId: 'network',
    icon: 'server',
    label: 'Allowed peers',
    description: 'Schema array control.',
    inheritedLabel: 'Inherited by network defaults',
    valueSchema: { kind: 'array', items: { kind: 'string' } },
    control: {
      kind: 'text',
      name: 'allowed_peers',
      value: 'peer-a, peer-b'
    }
  },
  {
    id: 'schema-object',
    categoryId: 'telemetry',
    icon: 'filter',
    label: 'Telemetry headers',
    description: 'Schema object control.',
    inheritedLabel: 'Inherited by telemetry defaults',
    canonicalPath: 'telemetry.headers',
    tomlSection: 'telemetry',
    tomlKey: 'headers',
    valueSchema: { kind: 'object' },
    control: {
      kind: 'text',
      name: 'telemetry_headers',
      value: '{"x-trace": "abc"}'
    }
  },
  {
    id: 'schema-conflict',
    categoryId: 'advanced',
    icon: 'filter',
    label: 'Draft pairing mode',
    description: 'Conflict metadata control.',
    inheritedLabel: 'Inherited by advanced defaults',
    valueSchema: { kind: 'enum', values: ['warn_disable', 'fail_launch'] },
    controlBehavior: {
      conflicts: [
        {
          group: 'speculative-pairing',
          reason: 'Conflicts with draft_min_tokens values above the configured maximum.',
          condition: {
            path: { segments: ['defaults', 'speculative', 'draft_min_tokens'] },
            operator: 'present'
          }
        }
      ]
    },
    control: {
      kind: 'choice',
      name: 'draft_pairing_mode',
      value: 'warn_disable',
      presentation: 'segmented',
      options: [
        { value: 'warn_disable', label: 'warn_disable' },
        { value: 'fail_launch', label: 'fail_launch' }
      ]
    }
  }
] satisfies readonly ConfigurationDefaultsSetting[]

const schemaDrivenControlData = {
  categories: [
    { id: 'runtime', label: 'Runtime', summary: 'Runtime defaults.', help: 'Runtime defaults' },
    { id: 'multimodal', label: 'Multimodal', summary: 'Multimodal defaults.', help: 'Multimodal defaults' },
    { id: 'network', label: 'Network', summary: 'Network defaults.', help: 'Network defaults' },
    { id: 'telemetry', label: 'Telemetry', summary: 'Telemetry defaults.', help: 'Telemetry defaults' },
    { id: 'advanced', label: 'Advanced', summary: 'Advanced defaults.', help: 'Advanced defaults' }
  ],
  settings: schemaDrivenControlSettings,
  preview: []
} satisfies ConfigurationDefaultsHarnessData

const slotDependencySettings = [
  {
    id: 'speculation-mode',
    categoryId: 'speculative-decoding',
    icon: 'brain',
    label: 'Speculation mode',
    description: 'Controls whether draft-model speculation is active.',
    inheritedLabel: 'Inherited by speculative decoding defaults',
    control: {
      kind: 'choice',
      name: 'speculation_mode',
      value: 'off',
      presentation: 'segmented',
      options: [
        { value: 'off', label: 'off' },
        { value: 'draft_model', label: 'draft model' }
      ]
    }
  },
  {
    id: 'parallel-slots',
    categoryId: 'speculative-decoding',
    icon: 'gauge',
    label: 'Default slots / parallel requests',
    description: 'Parallel slot count.',
    inheritedLabel: 'Only available when draft-model speculation is enabled',
    rendererId: 'slot-meter',
    dependsOn: {
      settingId: 'speculation-mode',
      condition: (value: string) => value === 'draft_model'
    },
    control: {
      kind: 'range',
      name: 'parallel',
      value: '4',
      min: 1,
      max: 16,
      step: 1,
      unit: 'slots'
    }
  }
] satisfies readonly ConfigurationDefaultsSetting[]

const slotDependencyData = {
  categories: [
    {
      id: 'speculative-decoding',
      label: 'Speculative Decoding',
      summary: 'Speculative defaults.',
      help: 'Speculative defaults'
    }
  ],
  settings: slotDependencySettings,
  preview: []
} satisfies ConfigurationDefaultsHarnessData

const defaultValues: ConfigurationDefaultsValues = {}

function renderDefaultsTab(overrides: Partial<Parameters<typeof DefaultsTab>[0]> = {}) {
  return render(
    <DefaultsTab
      data={overrides.data ?? defaultsData}
      values={overrides.values ?? defaultValues}
      onSettingValueChange={overrides.onSettingValueChange ?? vi.fn()}
      onResetAll={overrides.onResetAll ?? vi.fn()}
      configFilePath={overrides.configFilePath}
    />
  )
}

function previewSource() {
  const source = screen.getByRole('textbox', { name: /\[defaults\] preview code/i })

  if (!(source instanceof HTMLTextAreaElement)) throw new Error('Expected TOML preview textarea')

  return source
}

function defaultsRail() {
  return within(screen.getByRole('navigation', { name: /defaults sections/i }))
}

function settingsRow(label: string) {
  const row = screen.getByText(label).closest('[data-settings-row="true"]')

  if (!(row instanceof HTMLElement)) throw new Error(`Expected settings row for ${label}`)

  return row
}

function disabledInfoTrigger(row: HTMLElement) {
  const trigger = within(row).getByRole('button', { name: /why unavailable/i })

  if (!(trigger instanceof HTMLButtonElement)) throw new Error('Expected disabled info trigger button')

  return trigger
}

function settingInfoTrigger(row: HTMLElement) {
  const trigger = within(row).getByRole('button', { name: /setting information/i })

  if (!(trigger instanceof HTMLButtonElement)) throw new Error('Expected setting info trigger button')

  return trigger
}

describe('DefaultsTab', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('marks modified setting rows with the same warning tone used by dirty tabs', () => {
    renderDefaultsTab({
      values: {
        'runtime-mode': 'manual'
      }
    })

    const row = settingsRow('Runtime mode')
    expect(row).toHaveAttribute('data-settings-row-dirty', 'true')
    expect(within(row).getByText('Runtime mode')).toHaveClass('text-warn')
  })

  it('uses network-panel typography tokens for configuration helper text', () => {
    renderDefaultsTab({ configFilePath: '/Users/test/.mesh-llm/config.toml' })

    const runtimeSection = screen.getByRole('heading', { name: 'Runtime' }).closest('section')
    if (!(runtimeSection instanceof HTMLElement)) throw new Error('Expected runtime settings section')

    expect(screen.getByRole('heading', { name: 'Runtime' })).toHaveClass('type-panel-title', 'text-foreground')
    expect(within(runtimeSection).getByText('Runtime defaults')).toHaveClass('type-caption', 'text-fg-dim')

    const row = settingsRow('Runtime mode')
    expect(within(row).getByText('Controls the standard runtime selection.')).toHaveClass('type-caption', 'text-fg-dim')
    expect(screen.getByText('TIP')).toHaveClass('type-label')
    expect(screen.getByText('Configuration Path')).toHaveClass('type-label', 'text-fg-faint')
    expect(screen.getByText('/Users/test/.mesh-llm/config.toml')).toHaveClass(
      'font-mono',
      'text-[length:var(--density-type-caption-lg)]',
      'text-fg-dim'
    )
  })

  it('hides advanced settings by default and persists the toggle', async () => {
    const user = userEvent.setup()

    renderDefaultsTab({
      values: {
        'advanced-reasoning': '256'
      }
    })

    expect(screen.getByRole('button', { name: /show advanced/i })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('heading', { name: 'Runtime' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Reasoning' })).not.toBeInTheDocument()
    expect(previewSource().value).not.toContain('runtime_mode = "auto"')
    expect(previewSource().value).toContain('reasoning_budget = 256')

    await user.click(screen.getByRole('button', { name: /show advanced/i }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /hide advanced/i })).toHaveAttribute('aria-pressed', 'true')
      expect(screen.getByRole('heading', { name: 'Reasoning' })).toBeInTheDocument()
      expect(previewSource().value).toContain('[defaults.runtime]')
      expect(previewSource().value).toContain('reasoning_budget = 256')
      expect(window.localStorage.getItem(SHOW_ADVANCED_STORAGE_KEY)).toBe('true')
    })

    await user.click(screen.getByRole('button', { name: /hide advanced/i }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /show advanced/i })).toHaveAttribute('aria-pressed', 'false')
      expect(screen.queryByRole('heading', { name: 'Reasoning' })).not.toBeInTheDocument()
      expect(previewSource().value).toContain('[defaults.runtime]')
      expect(previewSource().value).toContain('reasoning_budget = 256')
      expect(window.localStorage.getItem(SHOW_ADVANCED_STORAGE_KEY)).toBeNull()
    })
  })

  it('hydrates show advanced from localStorage', () => {
    window.localStorage.setItem(SHOW_ADVANCED_STORAGE_KEY, 'true')

    renderDefaultsTab()

    expect(screen.getByRole('button', { name: /hide advanced/i })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('heading', { name: 'Reasoning' })).toBeInTheDocument()
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('prefill_chunk_size')
    expect(previewSource().value).not.toContain('prefill_chunk_schedule')
    expect(previewSource().value).not.toContain('mirostat_entropy')
  })

  it('renders the real defaults inventory with integrated categories and section previews', async () => {
    const user = userEvent.setup()

    const { rerender } = renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        threads: '12',
        temperature: '0.8',
        'top-k': '55',
        'activation-wire-dtype': 'q8',
        'binary-stage-transport': 'on',
        'image-min-tokens': '64',
        'mmproj-offload': 'on',
        'server-alias': 'carrack-mesh',
        'direct-io': 'off',
        mlock: 'on'
      }
    })

    const rail = defaultsRail()
    expect(rail.getAllByRole('button')).toHaveLength(6)
    expect(rail.getByRole('button', { name: /runtime/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /request defaults/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /skippy transport/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /multimodal/i })).toBeInTheDocument()
    expect(rail.queryByRole('button', { name: /advanced server/i })).not.toBeInTheDocument()
    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
    expect(screen.queryByText('Memory lock')).not.toBeInTheDocument()

    await user.click(rail.getByRole('button', { name: /request defaults/i }))
    expect(rail.getByRole('button', { name: /request defaults/i })).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Request Defaults' })).toBeInTheDocument()
    expect(screen.getByText('Temperature')).toBeInTheDocument()
    expect(screen.getByText('Top-k')).toBeInTheDocument()

    await user.click(rail.getByRole('button', { name: /skippy transport/i }))
    expect(rail.getByRole('button', { name: /skippy transport/i })).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Skippy Transport' })).toBeInTheDocument()
    expect(screen.getByText('Activation wire dtype')).toBeInTheDocument()

    await user.click(rail.getByRole('button', { name: /multimodal/i }))
    expect(rail.getByRole('button', { name: /multimodal/i })).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Multimodal' })).toBeInTheDocument()
    expect(screen.getByText('MMProj offload')).toBeInTheDocument()

    expect(screen.getByRole('slider', { name: 'CPU threads' })).toHaveValue('12')
    expect(screen.getByRole('slider', { name: 'Temperature' })).toHaveValue('0.8')
    expect(screen.getByRole('slider', { name: 'Top-k' })).toHaveValue('55')
    expect(previewSource().value).toContain('threads = 12')
    expect(previewSource().value).toContain('[defaults.request_defaults]')
    expect(previewSource().value).toContain('temperature = 0.8')
    expect(previewSource().value).toContain('top_k = 55')
    expect(previewSource().value).toContain('[defaults.skippy]')
    expect(previewSource().value).toContain('activation_wire_dtype = "q8"')
    expect(previewSource().value).toContain('binary_stage_transport = "on"')
    expect(previewSource().value).toContain('mlock = true')
    expect(previewSource().value).toContain('[defaults.multimodal]')
    expect(previewSource().value).toContain('image_min_tokens = 64')
    expect(previewSource().value).toContain('mmproj_offload = true')
    expect(previewSource().value).toContain('[defaults.advanced.server]')
    expect(previewSource().value).toContain('alias = "carrack-mesh"')

    rerender(
      <DefaultsTab
        data={CONFIGURATION_DEFAULTS}
        values={{
          threads: '12',
          temperature: '0.8',
          'top-k': '55',
          'activation-wire-dtype': 'q8',
          'binary-stage-transport': 'on',
          'image-min-tokens': '64',
          'mmproj-offload': 'on',
          'direct-io': 'off',
          mlock: 'on'
        }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    expect(previewSource().value).not.toContain('[defaults.advanced.server]')
  })

  it('omits default-only metadata sections while keeping hidden advanced non-default values in preview', () => {
    renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        mlock: 'on',
        'server-alias': 'carrack-mesh'
      }
    })

    expect(screen.queryByText('Memory lock')).not.toBeInTheDocument()
    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
    expect(previewSource().value).toContain('[defaults.hardware]')
    expect(previewSource().value).toContain('mlock = true')
    expect(previewSource().value).toContain('[defaults.advanced.server]')
    expect(previewSource().value).toContain('alias = "carrack-mesh"')
    expect(previewSource().value).not.toContain('[defaults.skippy]')
    expect(previewSource().value).not.toContain('[defaults.multimodal]')
  })

  it('uses canonical inventory defaults when previewing hydrated live settings', () => {
    const liveHydratedDefaults = {
      ...CONFIGURATION_DEFAULTS,
      settings: CONFIGURATION_DEFAULTS.settings.map((setting) => {
        if (setting.id === 'activation-wire-dtype') {
          return {
            ...setting,
            baselineValue: setting.control.value,
            control: {
              ...setting.control,
              value: 'q8'
            }
          }
        }

        if (setting.id === 'image-min-tokens') {
          return {
            ...setting,
            baselineValue: setting.control.value,
            control: {
              ...setting.control,
              value: '64'
            }
          }
        }

        if (setting.id === 'server-alias') {
          return {
            ...setting,
            baselineValue: setting.control.value,
            control: {
              ...setting.control,
              value: 'carrack-mesh'
            }
          }
        }

        return setting
      })
    } satisfies ConfigurationDefaultsHarnessData

    renderDefaultsTab({
      data: liveHydratedDefaults,
      values: {}
    })

    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
    expect(previewSource().value).toContain('[defaults.skippy]')
    expect(previewSource().value).toContain('activation_wire_dtype = "q8"')
    expect(previewSource().value).toContain('[defaults.multimodal]')
    expect(previewSource().value).toContain('image_min_tokens = 64')
    expect(previewSource().value).toContain('[defaults.advanced.server]')
    expect(previewSource().value).toContain('alias = "carrack-mesh"')
    expect(previewSource().value).not.toContain('[defaults.request_defaults]')
  })

  it('shows reset actions beside controls for restart-required settings and resets only that setting', async () => {
    const user = userEvent.setup()
    const onSettingValueChange = vi.fn()

    const { rerender } = renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        threads: '12',
        'top-k': '55'
      },
      onSettingValueChange
    })

    const cpuThreadsRow = within(settingsRow('CPU threads'))
    const topKRow = within(settingsRow('Top-k'))
    const resetButton = cpuThreadsRow.getByRole('button', { name: 'Reset CPU threads to default' })

    expect(resetButton).toBeInTheDocument()
    expect(topKRow.queryByRole('button', { name: /reset top-k to default/i })).not.toBeInTheDocument()
    expect(
      screen.getByText('CPU threads').compareDocumentPosition(resetButton) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    expect(
      resetButton.compareDocumentPosition(screen.getByRole('slider', { name: 'CPU threads' })) &
        Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()

    await user.hover(resetButton)
    expect(await screen.findByText(SETTING_RESET_TOOLTIP, { selector: 'div' })).toBeInTheDocument()
    await user.unhover(resetButton)

    await act(async () => {
      resetButton.focus()
    })
    expect(await screen.findByText(SETTING_RESET_TOOLTIP, { selector: 'div' })).toBeInTheDocument()

    await user.click(resetButton)
    expect(onSettingValueChange).toHaveBeenCalledWith('threads', '0')
    expect(onSettingValueChange).not.toHaveBeenCalledWith('top-k', expect.anything())

    rerender(
      <DefaultsTab
        data={CONFIGURATION_DEFAULTS}
        values={{}}
        onSettingValueChange={onSettingValueChange}
        onResetAll={vi.fn()}
      />
    )
    expect(
      within(settingsRow('CPU threads')).queryByRole('button', { name: 'Reset CPU threads to default' })
    ).toBeNull()
  })

  it('keeps advanced filtering consistent across real category counts, rows, and section visibility', async () => {
    const user = userEvent.setup()

    renderDefaultsTab({ data: CONFIGURATION_DEFAULTS })

    const rail = defaultsRail()
    expect(rail.getAllByRole('button')).toHaveLength(6)
    expect(screen.queryByText('Mirostat mode')).not.toBeInTheDocument()
    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /show advanced/i }))

    await waitFor(() => {
      expect(rail.getAllByRole('button')).toHaveLength(7)
      expect(rail.getByRole('button', { name: /advanced server/i })).toBeInTheDocument()
      expect(screen.getByText('Mirostat mode')).toBeInTheDocument()
      expect(screen.getByText('Server alias')).toBeInTheDocument()
    })

    await user.click(rail.getByRole('button', { name: /advanced server/i }))
    expect(rail.getByRole('button', { name: /advanced server/i })).toHaveAttribute('aria-current', 'true')

    await user.click(screen.getByRole('button', { name: /hide advanced/i }))

    await waitFor(() => {
      expect(rail.getAllByRole('button')).toHaveLength(6)
      expect(rail.queryByRole('button', { name: /advanced server/i })).not.toBeInTheDocument()
      expect(screen.queryByText('Mirostat mode')).not.toBeInTheDocument()
      expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
      expect(rail.getByRole('button', { name: /runtime/i })).toHaveAttribute('aria-current', 'true')
    })
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
    expect(
      screen.queryByText('Path hint: enter a local filesystem path. No file picker is available here.')
    ).not.toBeInTheDocument()

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
      await screen.findByText('Path hint: enter a local filesystem path. No file picker is available here.', {
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
})

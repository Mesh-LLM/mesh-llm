/* eslint-disable react-refresh/only-export-components */
import { render, screen, within } from '@testing-library/react'
import { vi } from 'vitest'
import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'
import { DefaultsTab } from '@/features/configuration/components/DefaultsTab'
import type {
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues
} from '@/features/app-tabs/types'
import { env } from '@/lib/env'

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
    id: 'topology-stages',
    categoryId: 'runtime',
    icon: 'layers',
    label: 'Topology stages',
    description: 'Schema-defined locked topology stages.',
    inheritedLabel: 'Inherited by locked model placements',
    canonicalPath: 'defaults.topology.stages',
    tomlSection: 'defaults.topology',
    tomlKey: 'stages',
    valueSchema: {
      kind: 'array',
      items: {
        kind: 'object',
        properties: [
          {
            name: 'node',
            label: 'Node',
            required: true,
            value_schema: {
              kind: 'object',
              properties: [
                { name: 'endpoint_id', label: 'Endpoint ID', required: false, value_schema: { kind: 'string' } },
                { name: 'hostname', label: 'Hostname', required: false, value_schema: { kind: 'string' } }
              ]
            }
          },
          { name: 'layer_start', label: 'Layer start', required: true, value_schema: { kind: 'integer' } },
          { name: 'layer_end', label: 'Layer end', required: true, value_schema: { kind: 'integer' } }
        ]
      }
    },
    control: {
      kind: 'text',
      name: 'stages',
      value:
        '[{"node":{"endpoint_id":"endpoint-a"},"layer_start":0,"layer_end":16},{"node":{"hostname":"worker-b"},"layer_start":16,"layer_end":32}]'
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

export {
  CONFIGURATION_DEFAULTS,
  SHOW_ADVANCED_STORAGE_KEY,
  defaultSettings,
  defaultsData,
  dependencyData,
  dependencySettings,
  defaultsRail,
  disabledInfoTrigger,
  defaultValues,
  previewSource,
  renderDefaultsTab,
  schemaDrivenControlData,
  schemaDrivenControlSettings,
  settingInfoTrigger,
  settingsRow,
  slotDependencyData,
  slotDependencySettings
}

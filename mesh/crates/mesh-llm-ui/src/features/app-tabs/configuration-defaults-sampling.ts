import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'
import {
  DRAFT_MODEL_SPECULATION_DEPENDENCY,
  MIROSTAT_MODE_DEPENDENCY,
  REQUEST_DEFAULTS_TOML_SECTION
} from './configuration-defaults-constants'

export const CONFIGURATION_DEFAULT_SAMPLING_SETTINGS = [
  {
    id: 'speculation-mode',
    categoryId: 'speculative-decoding',
    tomlSection: 'defaults.speculative',
    tomlKey: 'mode',
    icon: 'brain',
    label: 'Default speculation mode',
    description: 'Choose the default speculation method, or leave the runtime in auto mode.',
    inheritedLabel: 'Inherited by compatible placements unless a model pins a mode',
    control: {
      kind: 'choice',
      name: 'mode',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'disabled', label: 'disabled' },
        { value: 'draft', label: 'draft' }
      ]
    }
  },
  {
    id: 'draft-selection-policy',
    categoryId: 'speculative-decoding',
    tomlSection: 'defaults.speculative',
    tomlKey: 'draft_selection_policy',
    icon: 'filter',
    label: 'Default draft selection policy',
    description: 'Choose how draft models are selected when draft-model speculation is active.',
    inheritedLabel: 'Controls whether Mesh chooses a draft from catalog metadata',
    dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
    control: {
      kind: 'choice',
      name: 'draft_selection_policy',
      value: 'auto',
      presentation: 'toggle',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'manual', label: 'manual' }
      ]
    }
  },
  {
    id: 'incompatible-pairing-behavior',
    categoryId: 'speculative-decoding',
    tomlSection: 'defaults.speculative',
    tomlKey: 'pairing_fault',
    icon: 'shield',
    label: 'Incompatible pairing behavior',
    description: 'Choose what happens when the draft and target models cannot pair.',
    inheritedLabel: 'Determines launch behavior when draft and target models cannot pair',
    dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
    control: {
      kind: 'choice',
      name: 'pairing_fault',
      value: 'warn_disable',
      presentation: 'toggle',
      options: [
        { value: 'warn_disable', label: 'Warn & Disable' },
        { value: 'fail_closed', label: 'Fail launch' }
      ]
    }
  },
  {
    id: 'draft-max-tokens',
    categoryId: 'speculative-decoding',
    tomlSection: 'defaults.speculative',
    tomlKey: 'draft_max_tokens',
    icon: 'gauge',
    label: 'Default draft max tokens',
    description: 'Limit how many draft tokens can be proposed before verification.',
    inheritedLabel: 'Higher values can improve throughput when acceptance stays high',
    dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
    control: { kind: 'range', name: 'draft_max_tokens', value: '16', min: 1, max: 64, step: 1, unit: 'tokens' }
  },
  {
    id: 'draft-min-tokens',
    categoryId: 'speculative-decoding',
    tomlSection: 'defaults.speculative',
    tomlKey: 'draft_min_tokens',
    icon: 'gauge',
    label: 'Default draft minimum tokens',
    description: 'Set the smallest draft batch attempted before verification.',
    inheritedLabel: '0 lets the runtime verify as soon as the draft becomes uncertain',
    mutability: 'restart-required',
    dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
    control: { kind: 'range', name: 'draft_min_tokens', value: '0', min: 0, max: 32, step: 1, unit: 'tokens' }
  },
  {
    id: 'draft-acceptance-threshold',
    categoryId: 'speculative-decoding',
    tomlSection: 'defaults.speculative',
    tomlKey: 'draft_acceptance_threshold',
    icon: 'gauge',
    label: 'Default draft acceptance threshold',
    description: 'Set the confidence needed before draft tokens are accepted.',
    inheritedLabel: 'Lower values speculate more aggressively; higher values reject earlier',
    visibility: 'advanced',
    mutability: 'restart-required',
    dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
    control: { kind: 'range', name: 'draft_acceptance_threshold', value: '0.70', min: 0, max: 1, step: 0.05 }
  },
  {
    id: 'temperature',
    categoryId: 'request-defaults',
    tomlSection: 'defaults.request_defaults',
    tomlKey: 'temperature',
    icon: 'gauge',
    label: 'Temperature',
    description: 'Fallback sampling temperature for requests that do not provide one.',
    inheritedLabel: 'Request payload temperature always wins when it is present',
    control: { kind: 'range', name: 'temperature', value: '0.70', min: 0, max: 2, step: 0.05 }
  },
  {
    id: 'top-p',
    categoryId: 'request-defaults',
    tomlSection: 'defaults.request_defaults',
    tomlKey: 'top_p',
    icon: 'gauge',
    label: 'Top-p',
    description: 'Fallback nucleus sampling threshold for requests that omit one.',
    inheritedLabel: 'Request payload top-p wins over this default',
    control: { kind: 'range', name: 'top_p', value: '0.95', min: 0, max: 1, step: 0.05 }
  },
  {
    id: 'reasoning-format',
    categoryId: 'request-defaults',
    tomlSection: 'defaults.request_defaults',
    tomlKey: 'reasoning_format',
    icon: 'cog',
    label: 'Reasoning format',
    description: 'Choose how thinking tokens appear in the response stream.',
    inheritedLabel: 'Inherited by model runtimes unless disabled per placement',
    control: {
      kind: 'choice',
      name: 'reasoning_format',
      value: 'auto',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'none', label: 'none' },
        { value: 'deepseek', label: 'deepseek' },
        { value: 'deepseek-legacy', label: 'deepseek-legacy' }
      ]
    }
  },
  {
    id: 'reasoning-budget',
    categoryId: 'request-defaults',
    tomlSection: 'defaults.request_defaults',
    tomlKey: 'reasoning_budget',
    icon: 'gauge',
    label: 'Reasoning budget',
    description: 'Cap the reasoning tokens reserved before the final answer.',
    inheritedLabel: 'Used only by runtimes with reasoning enabled',
    control: { kind: 'range', name: 'reasoning_budget', value: '0', min: 0, max: 4096, step: 128, unit: 'tok' }
  },
  {
    id: 'repeat-penalty',
    categoryId: 'request-defaults',
    tomlSection: 'defaults.request_defaults',
    tomlKey: 'repeat_penalty',
    icon: 'filter',
    label: 'Repeat penalty',
    description: 'Adjust how strongly repeated tokens are discouraged.',
    inheritedLabel: 'Safe fallback unless a placement tunes sampling',
    control: { kind: 'range', name: 'repeat_penalty', value: '1.1', min: 1, max: 2, step: 0.05 }
  },
  {
    id: 'repeat-last-n',
    categoryId: 'request-defaults',
    tomlSection: 'defaults.request_defaults',
    tomlKey: 'repeat_last_n',
    icon: 'layers',
    label: 'Repeat last-n window',
    description: 'Set how much recent token history the repeat penalty checks.',
    inheritedLabel: 'Inherited by placements with default sampling',
    control: { kind: 'range', name: 'repeat_last_n', value: '256', min: 0, max: 1024, step: 32, unit: 'tok' }
  },
  {
    id: 'top-k',
    categoryId: 'request-defaults',
    icon: 'filter',
    label: 'Top-k',
    description: 'Limit sampling to the top-k tokens.',
    inheritedLabel: 'Applied when a request does not override top-k',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: { kind: 'range', name: 'top_k', value: '40', min: 0, max: 100, step: 1 }
  },
  {
    id: 'min-p',
    categoryId: 'request-defaults',
    icon: 'filter',
    label: 'Min-p',
    description: 'Filter tokens below a dynamic probability floor.',
    inheritedLabel: 'Applied when a request does not override min-p',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: { kind: 'range', name: 'min_p', value: '0.05', min: 0, max: 1, step: 0.05 }
  },
  {
    id: 'presence-penalty',
    categoryId: 'request-defaults',
    icon: 'filter',
    label: 'Presence penalty',
    description: 'Increase or reduce the penalty for introducing new tokens.',
    inheritedLabel: 'Applied when a request does not override presence penalty',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: { kind: 'range', name: 'presence_penalty', value: '0', min: 0, max: 2, step: 0.1 }
  },
  {
    id: 'frequency-penalty',
    categoryId: 'request-defaults',
    icon: 'filter',
    label: 'Frequency penalty',
    description: 'Increase or reduce the penalty for repeated tokens.',
    inheritedLabel: 'Applied when a request does not override frequency penalty',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: { kind: 'range', name: 'frequency_penalty', value: '0', min: 0, max: 2, step: 0.1 }
  },
  {
    id: 'max-tokens',
    categoryId: 'request-defaults',
    icon: 'gauge',
    label: 'Max tokens',
    description: 'Cap the number of generated tokens for a request.',
    inheritedLabel: 'Applied when a request does not override the token cap',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: { kind: 'range', name: 'max_tokens', value: '0', min: 0, max: 32768, step: 256, unit: 'tokens' }
  },
  {
    id: 'seed',
    categoryId: 'request-defaults',
    icon: 'cog',
    label: 'Seed',
    description: 'Set the RNG seed for deterministic sampling when needed.',
    inheritedLabel: 'Applied when a request does not override the seed',
    visibility: 'advanced',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: { kind: 'text', name: 'seed', value: '-1', placeholder: '-1 (random)' }
  },
  {
    id: 'ignore-eos',
    categoryId: 'request-defaults',
    icon: 'filter',
    label: 'Ignore EOS',
    description: 'Choose whether the model should ignore end-of-sequence tokens.',
    inheritedLabel: 'Applied when a request does not override EOS handling',
    visibility: 'advanced',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: {
      kind: 'choice',
      name: 'ignore_eos',
      value: 'off',
      presentation: 'toggle',
      options: [
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'mirostat-mode',
    categoryId: 'request-defaults',
    icon: 'brain',
    label: 'Mirostat mode',
    description: 'Choose the Mirostat sampling mode, or disable it.',
    inheritedLabel: 'Applied when a request does not override Mirostat mode',
    visibility: 'advanced',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
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
    description: 'Set the Mirostat target entropy.',
    inheritedLabel: 'Applied when Mirostat mode is enabled',
    visibility: 'advanced',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    dependsOn: MIROSTAT_MODE_DEPENDENCY,
    control: { kind: 'range', name: 'mirostat_entropy', value: '5', min: 0.1, max: 10, step: 0.1 }
  },
  {
    id: 'mirostat-learning-rate',
    categoryId: 'request-defaults',
    icon: 'gauge',
    label: 'Mirostat learning rate',
    description: 'Set the Mirostat learning rate.',
    inheritedLabel: 'Applied when Mirostat mode is enabled',
    visibility: 'advanced',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    dependsOn: MIROSTAT_MODE_DEPENDENCY,
    control: { kind: 'range', name: 'mirostat_learning_rate', value: '0.1', min: 0.01, max: 1, step: 0.01 }
  },
  {
    id: 'samplers',
    categoryId: 'request-defaults',
    icon: 'filter',
    label: 'Samplers',
    description: 'Set the comma-separated sampler list.',
    inheritedLabel: 'Applied when a request does not override the sampler list',
    visibility: 'advanced',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: {
      kind: 'text',
      name: 'samplers',
      value: '',
      placeholder: 'top_k,tfs_z,typical_p,top_p,min_p,temperature'
    }
  },
  {
    id: 'sampler-sequence',
    categoryId: 'request-defaults',
    icon: 'layers',
    label: 'Sampler sequence',
    description: 'Set the sampler execution order.',
    inheritedLabel: 'Applied when a request does not override sampler ordering',
    visibility: 'advanced',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: {
      kind: 'text',
      name: 'sampler_sequence',
      value: '',
      placeholder: 'e.g. top_k;top_p;temperature'
    }
  },
  {
    id: 'stop',
    categoryId: 'request-defaults',
    icon: 'shield',
    label: 'Stop sequences',
    description: 'Set comma-separated stop sequences for a request.',
    inheritedLabel: 'Applied when a request does not override stop sequences',
    visibility: 'advanced',
    tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
    mutability: 'runtime',
    control: {
      kind: 'text',
      name: 'stop',
      value: '',
      placeholder: 'comma-separated stop sequences'
    }
  }
] as const satisfies readonly ConfigurationDefaultsSetting[]

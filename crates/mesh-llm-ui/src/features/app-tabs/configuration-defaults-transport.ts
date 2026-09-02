import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'
import {
  ADVANCED_SERVER_TOML_SECTION,
  MULTIMODAL_TOML_SECTION,
  PREFILL_CHUNKING_FIXED_DEPENDENCY,
  PREFILL_CHUNKING_SCHEDULE_DEPENDENCY,
  SKIPPY_TRANSPORT_TOML_SECTION
} from './configuration-defaults-constants'

export const CONFIGURATION_DEFAULT_TRANSPORT_SETTINGS = [
  {
    id: 'stage-model-path',
    categoryId: 'skippy-transport',
    icon: 'folder',
    label: 'Stage model path',
    description: 'Set the model or package path used for this skippy stage.',
    inheritedLabel: 'Inherited by stage chains without an explicit model path',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    visibility: 'advanced',
    mutability: 'restart-required',
    control: {
      kind: 'text',
      name: 'stage_model_path',
      value: '',
      placeholder: 'e.g. hf://meshllm/... or /path/to/stage.gguf'
    }
  },
  {
    id: 'stage-role',
    categoryId: 'skippy-transport',
    icon: 'server',
    label: 'Stage role',
    description: 'Choose the stage-chain role when topology is not inferred automatically.',
    inheritedLabel: 'Inherited by stage chains without an explicit role override',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    visibility: 'advanced',
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'stage_role',
      value: 'auto',
      presentation: 'select',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'prompt', label: 'prompt' },
        { value: 'stage', label: 'stage' }
      ]
    }
  },
  {
    id: 'stage-topology',
    categoryId: 'skippy-transport',
    icon: 'layers',
    label: 'Stage topology',
    description: 'Describe the stage chain topology when it is supplied as a text override.',
    inheritedLabel: 'Inherited by stage chains without a topology override',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    visibility: 'advanced',
    mutability: 'restart-required',
    control: {
      kind: 'text',
      name: 'stage_topology',
      value: '',
      placeholder: 'topology name or path'
    }
  },
  {
    id: 'prefill-chunking',
    categoryId: 'skippy-transport',
    icon: 'layers',
    label: 'Prefill chunking',
    description: 'Choose how prefill chunks are scheduled across a skippy stage chain.',
    inheritedLabel: 'Inherited by stage chains without a chunking override',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'prefill_chunking',
      value: 'auto',
      presentation: 'select',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'fixed', label: 'fixed' },
        { value: 'schedule', label: 'schedule' },
        { value: 'adaptive-ramp', label: 'adaptive-ramp' }
      ]
    }
  },
  {
    id: 'prefill-chunk-size',
    categoryId: 'skippy-transport',
    icon: 'gauge',
    label: 'Prefill chunk size',
    description: 'Set the fixed prefill chunk size. Use 0 to keep the backend auto sentinel.',
    inheritedLabel: 'Inherited by fixed chunking when a stage does not override the size',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    mutability: 'restart-required',
    dependsOn: PREFILL_CHUNKING_FIXED_DEPENDENCY,
    control: { kind: 'range', name: 'prefill_chunk_size', value: '0', min: 0, max: 8192, step: 64, unit: 'tokens' }
  },
  {
    id: 'prefill-chunk-schedule',
    categoryId: 'skippy-transport',
    icon: 'layers',
    label: 'Prefill chunk schedule',
    description: 'Provide a comma-separated schedule for scheduled prefill chunking.',
    inheritedLabel: 'Inherited by scheduled chunking when a stage does not override the schedule',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    visibility: 'advanced',
    mutability: 'restart-required',
    dependsOn: PREFILL_CHUNKING_SCHEDULE_DEPENDENCY,
    control: {
      kind: 'text',
      name: 'prefill_chunk_schedule',
      value: '',
      placeholder: 'e.g. 512,1024,2048'
    }
  },
  {
    id: 'binary-stage-transport',
    categoryId: 'skippy-transport',
    icon: 'binary',
    label: 'Binary stage transport',
    description: 'Choose whether the binary stage transport is enabled or left to auto selection.',
    inheritedLabel: 'Inherited by stage chains without a transport override',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'binary_stage_transport',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'lifecycle-startup-timeout-ms',
    categoryId: 'skippy-transport',
    icon: 'gauge',
    label: 'Lifecycle startup timeout',
    description: 'Set how long the orchestrator waits for a stage to become ready during startup.',
    inheritedLabel: 'Inherited by stage chains without a startup timeout override',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    visibility: 'advanced',
    mutability: 'restart-required',
    control: {
      kind: 'range',
      name: 'lifecycle_startup_timeout_ms',
      value: '30000',
      min: 1,
      max: 600000,
      step: 1000,
      unit: 'ms'
    }
  },
  {
    id: 'lifecycle-readiness-interval-ms',
    categoryId: 'skippy-transport',
    icon: 'gauge',
    label: 'Lifecycle readiness interval',
    description: 'Set how often readiness is re-checked while startup is in flight.',
    inheritedLabel: 'Inherited by stage chains without a readiness polling override',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    visibility: 'advanced',
    mutability: 'restart-required',
    control: {
      kind: 'range',
      name: 'lifecycle_readiness_interval_ms',
      value: '1000',
      min: 100,
      max: 60000,
      step: 100,
      unit: 'ms'
    }
  },
  {
    id: 'lifecycle-health-interval-ms',
    categoryId: 'skippy-transport',
    icon: 'shield',
    label: 'Lifecycle health interval',
    description: 'Set how often background health checks run after a stage is up.',
    inheritedLabel: 'Inherited by stage chains without a health polling override',
    tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
    visibility: 'advanced',
    mutability: 'restart-required',
    control: {
      kind: 'range',
      name: 'lifecycle_health_interval_ms',
      value: '15000',
      min: 100,
      max: 60000,
      step: 100,
      unit: 'ms'
    }
  },
  {
    id: 'mmproj-offload',
    categoryId: 'multimodal',
    icon: 'image',
    label: 'MMProj offload',
    description: 'Choose whether the multimodal projector stays auto-managed or explicitly on or off.',
    inheritedLabel: 'Inherited by placements without a projector-offload override',
    tomlSection: MULTIMODAL_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'mmproj_offload',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'image-min-tokens',
    categoryId: 'multimodal',
    icon: 'image',
    label: 'Image minimum tokens',
    description: 'Set the minimum token budget reserved for each image input.',
    inheritedLabel: 'Inherited by placements without an image minimum override',
    tomlSection: MULTIMODAL_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'image_min_tokens', value: '0', min: 0, max: 2048, step: 32, unit: 'tokens' }
  },
  {
    id: 'image-max-tokens',
    categoryId: 'multimodal',
    icon: 'image',
    label: 'Image maximum tokens',
    description: 'Set the maximum token budget allowed for each image input.',
    inheritedLabel: 'Inherited by placements without an image maximum override',
    tomlSection: MULTIMODAL_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'image_max_tokens', value: '2048', min: 0, max: 4096, step: 32, unit: 'tokens' }
  },
  {
    id: 'mmproj',
    categoryId: 'multimodal',
    icon: 'image',
    label: 'MMProj path',
    description: 'Set an explicit local path to the multimodal projector file.',
    inheritedLabel: 'Inherited by placements without an explicit projector path',
    visibility: 'advanced',
    tomlSection: MULTIMODAL_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'text', name: 'mmproj', value: '', placeholder: 'e.g. /path/to/mmproj.gguf' }
  },
  {
    id: 'mmproj-url',
    categoryId: 'multimodal',
    icon: 'image',
    label: 'MMProj URL',
    description: 'Set a URL used to download or reference the multimodal projector file.',
    inheritedLabel: 'Inherited by placements without a projector URL override',
    visibility: 'advanced',
    tomlSection: MULTIMODAL_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'text', name: 'mmproj_url', value: '', placeholder: 'e.g. https://example.com/mmproj.gguf' }
  },
  {
    id: 'server-alias',
    categoryId: 'advanced-server',
    icon: 'server',
    label: 'Server alias',
    description: 'Set a human-friendly alias for the server in advanced deployments.',
    inheritedLabel: 'Inherited by deployments without an explicit server alias',
    visibility: 'advanced',
    tomlSection: ADVANCED_SERVER_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'text', name: 'alias', value: '', placeholder: 'model alias' }
  }
] as const satisfies readonly ConfigurationDefaultsSetting[]

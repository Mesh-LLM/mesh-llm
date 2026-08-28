import type { ConfigurationDefaultsHarnessData } from '@/features/app-tabs/types'
import { CONFIGURATION_DEFAULT_RUNTIME_SETTINGS } from './configuration-defaults-runtime'
import { CONFIGURATION_DEFAULT_SAMPLING_SETTINGS } from './configuration-defaults-sampling'
import { CONFIGURATION_DEFAULT_TRANSPORT_SETTINGS } from './configuration-defaults-transport'
import {
  ADVANCED_SERVER_TOML_SECTION,
  MULTIMODAL_TOML_SECTION,
  SKIPPY_TRANSPORT_TOML_SECTION
} from './configuration-defaults-constants'

const CONFIGURATION_DEFAULT_SETTINGS = [
  ...CONFIGURATION_DEFAULT_RUNTIME_SETTINGS,
  ...CONFIGURATION_DEFAULT_SAMPLING_SETTINGS,
  ...CONFIGURATION_DEFAULT_TRANSPORT_SETTINGS
]

export const CONFIGURATION_DEFAULTS = {
  categories: [
    {
      id: 'runtime',
      label: 'Runtime',
      summary: 'Model fit, hardware, and throughput defaults.',
      help: 'Load-time runtime behavior and concurrency defaults'
    },
    {
      id: 'memory',
      label: 'Memory',
      summary: 'KV cache policy and fit headroom.',
      help: 'VRAM accounting and fit headroom'
    },
    {
      id: 'speculative-decoding',
      label: 'Speculative Decoding',
      summary: 'Draft acceleration defaults.',
      help: 'Speculative draft policy defaults',
      tomlSection: 'defaults.speculative'
    },
    {
      id: 'request-defaults',
      label: 'Request Defaults',
      summary: 'Sampling, reasoning, and request-time fallback defaults.',
      help: 'Request-time sampling and reasoning defaults'
    },
    {
      id: 'skippy-transport',
      label: 'Skippy Transport',
      summary: 'Prefill chunking, stage transport, and lifecycle timing.',
      help: 'Stage transport, chunking, and lifecycle defaults',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION
    },
    {
      id: 'multimodal',
      label: 'Multimodal',
      summary: 'Projector and image token defaults.',
      help: 'Vision projector and image token defaults',
      tomlSection: MULTIMODAL_TOML_SECTION
    },
    {
      id: 'advanced-server',
      label: 'Advanced Server',
      summary: 'Server identity and operator overrides.',
      help: 'Advanced server defaults and identity overrides',
      tomlSection: ADVANCED_SERVER_TOML_SECTION
    }
  ],

  settings: CONFIGURATION_DEFAULT_SETTINGS,
  preview: [
    { label: 'Scope', value: 'carrack only', meta: 'remote nodes are read-only context' },
    { label: 'Config path', value: '~/.mesh-llm/config.toml' },
    {
      label: 'Generated defaults',
      value: `${CONFIGURATION_DEFAULT_SETTINGS.length} settings`,
      meta: 'deployment overrides win'
    },
    { label: 'Signing', value: 'Unsigned', meta: 'attestation pending' }
  ]
} as const satisfies ConfigurationDefaultsHarnessData

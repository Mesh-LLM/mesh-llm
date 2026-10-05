import {
  NONE,
  PLUGIN_INSTALL_CONFLICTS,
  REDACTED_NAME,
  TUNE_CONFLICTS,
  TUNE_FLAGS,
  descriptor
} from './command-summary-descriptor-options'
import type { SummaryDescriptor } from './command-summary-descriptor-types'

export const PLUGIN_DESCRIPTORS: readonly SummaryDescriptor[] = [
  descriptor(
    ['mesh-llm', 'plugins', 'install'],
    NONE,
    ['reference', '--archive', '--name', '--version'],
    false,
    'none',
    PLUGIN_INSTALL_CONFLICTS
  ),
  descriptor(['mesh-llm', 'plugins', 'update'], NONE, REDACTED_NAME),
  descriptor(['mesh-llm', 'plugins', 'enable'], NONE, REDACTED_NAME),
  descriptor(['mesh-llm', 'plugins', 'disable'], NONE, REDACTED_NAME),
  descriptor(['mesh-llm', 'plugins', 'delete'], NONE, REDACTED_NAME),
  descriptor(['mesh-llm', 'plugins', 'info'], NONE, REDACTED_NAME),
  descriptor(['mesh-llm', 'plugins', 'search'], NONE, ['query']),
  descriptor(['mesh-llm', 'plugins', 'list'])
]

export const BENCHMARK_DESCRIPTORS: readonly SummaryDescriptor[] = [
  descriptor(
    ['mesh-llm', 'benchmark', 'tune'],
    TUNE_FLAGS,
    [
      '--model',
      '--models',
      '--ctx-sizes',
      '--batch-sizes',
      '--ubatch-sizes',
      '--mmap-values',
      '--mlock-values',
      '--flash-attention',
      '--speculative-types',
      '--spec-draft-models',
      '--spec-draft-max-tokens',
      '--spec-draft-min-tokens',
      '--spec-ngram-min',
      '--spec-ngram-max',
      '--spec-draft-acceptance-threshold',
      '--spec-draft-split-probability',
      '--throughput-tolerance-pct',
      '--max-tokens',
      '--startup-timeout-secs',
      '--request-timeout-secs',
      '--prompt'
    ],
    false,
    'none',
    TUNE_CONFLICTS
  ),
  descriptor(['mesh-llm', 'benchmark', 'import-prompts'], NONE, ['--source', '--limit', '--max-tokens', '--output'])
]

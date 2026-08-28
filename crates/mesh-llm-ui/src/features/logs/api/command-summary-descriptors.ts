import { AUTH_DESCRIPTORS } from './command-summary-descriptors-auth'
import { MODEL_DESCRIPTORS } from './command-summary-descriptors-models'
import { BENCHMARK_DESCRIPTORS, PLUGIN_DESCRIPTORS } from './command-summary-descriptors-plugins-benchmark'
import { RUNTIME_DESCRIPTORS } from './command-summary-descriptors-runtime'
import { TOP_LEVEL_DESCRIPTORS } from './command-summary-descriptors-top-level'
import type { SummaryDescriptor } from './command-summary-descriptor-types'

export type { SummaryDescriptor, SummaryRawKind } from './command-summary-descriptor-types'

export const SUMMARY_DESCRIPTORS: readonly SummaryDescriptor[] = [
  ...TOP_LEVEL_DESCRIPTORS,
  ...PLUGIN_DESCRIPTORS,
  ...MODEL_DESCRIPTORS,
  ...BENCHMARK_DESCRIPTORS,
  ...RUNTIME_DESCRIPTORS,
  ...AUTH_DESCRIPTORS
]

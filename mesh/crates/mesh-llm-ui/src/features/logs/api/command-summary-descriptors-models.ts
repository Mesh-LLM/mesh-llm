import {
  JSON_FLAGS,
  MODEL_CERTIFY_FLAGS,
  MODEL_PACKAGE_FLAGS,
  MODEL_SEARCH_CONFLICTS,
  REDACTED_MODEL,
  SEARCH_FLAGS,
  YES_JSON,
  descriptor
} from './command-summary-descriptor-options'
import type { SummaryDescriptor } from './command-summary-descriptor-types'

export const MODEL_DESCRIPTORS: readonly SummaryDescriptor[] = [
  descriptor(['mesh-llm', 'models', 'package'], MODEL_PACKAGE_FLAGS, [
    'source_repo',
    '--quant',
    '--target',
    '--model-id',
    '--flavor',
    '--timeout',
    '--mesh-llm-ref',
    '--status',
    '--logs',
    '--cancel'
  ]),
  descriptor(['mesh-llm', 'models', 'recommended'], JSON_FLAGS),
  descriptor(['mesh-llm', 'models', 'installed'], JSON_FLAGS),
  descriptor(['mesh-llm', 'models', 'cleanup'], YES_JSON, ['--unused-since']),
  descriptor(['mesh-llm', 'models', 'prune'], YES_JSON),
  descriptor(['mesh-llm', 'models', 'certify'], MODEL_CERTIFY_FLAGS, [
    'model',
    '--report-out',
    '--api-base',
    '--prompt',
    '--max-tokens'
  ]),
  descriptor(['mesh-llm', 'models', 'list'], JSON_FLAGS),
  descriptor(
    ['mesh-llm', 'models', 'search'],
    SEARCH_FLAGS,
    ['query', '--limit', '--sort'],
    false,
    'none',
    MODEL_SEARCH_CONFLICTS
  ),
  descriptor(['mesh-llm', 'models', 'show'], JSON_FLAGS, REDACTED_MODEL),
  descriptor(['mesh-llm', 'models', 'download'], ['--draft', '--direct', '--json'], REDACTED_MODEL),
  descriptor(['mesh-llm', 'models', 'updates'], ['--all', '--check', '--json'], ['repo']),
  descriptor(['mesh-llm', 'models', 'delete'], YES_JSON, REDACTED_MODEL)
]

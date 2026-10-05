import {
  JSON_FLAGS,
  NONE,
  REDACTED_APPLY_CONFIG,
  REDACTED_NAME,
  REDACTED_REMOTE,
  REDACTED_REMOTE_MODEL,
  REMOTE_MODEL_CONFLICTS,
  RUNTIME_LIST_CONFLICTS,
  RUNTIME_LIST_FLAGS,
  RUNTIME_PRUNE_FLAGS,
  descriptor
} from './command-summary-descriptor-options'
import type { SummaryDescriptor } from './command-summary-descriptor-types'

export const RUNTIME_DESCRIPTORS: readonly SummaryDescriptor[] = [
  descriptor(['mesh-llm', 'runtime', 'status'], NONE, NONE, true),
  descriptor(['mesh-llm', 'runtime']),
  descriptor(['mesh-llm', 'runtime', 'load'], NONE, REDACTED_NAME, true),
  descriptor(['mesh-llm', 'runtime', 'unload'], NONE, REDACTED_NAME, true),
  descriptor(['mesh-llm', 'runtime', 'guardrails'], JSON_FLAGS, NONE, true, 'mode'),
  descriptor(['mesh-llm', 'runtime', 'bootstrap'], JSON_FLAGS, NONE, true),
  descriptor(
    ['mesh-llm', 'runtime', 'list'],
    RUNTIME_LIST_FLAGS,
    ['--manifest', '--bundle-dir', '--cache-dir'],
    false,
    'none',
    RUNTIME_LIST_CONFLICTS
  ),
  descriptor(['mesh-llm', 'runtime', 'install'], JSON_FLAGS, [
    'runtime_ref',
    '--manifest',
    '--bundle-dir',
    '--cache-dir'
  ]),
  descriptor(['mesh-llm', 'runtime', 'remove'], JSON_FLAGS, ['native_runtime_id', '--mesh-version', '--cache-dir']),
  descriptor(['mesh-llm', 'runtime', 'prune'], RUNTIME_PRUNE_FLAGS, ['--mesh-version', '--cache-dir']),
  descriptor(['mesh-llm', 'runtime', 'remote'], JSON_FLAGS, REDACTED_REMOTE, true),
  descriptor(
    ['mesh-llm', 'runtime', 'remote-model'],
    JSON_FLAGS,
    [...REDACTED_REMOTE_MODEL, '--instance-id'],
    true,
    'none',
    REMOTE_MODEL_CONFLICTS
  ),
  descriptor(['mesh-llm', 'runtime', 'apply-config'], JSON_FLAGS, REDACTED_APPLY_CONFIG, true)
]

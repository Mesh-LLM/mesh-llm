import {
  DISCOVER_FLAGS,
  DRAFT,
  JSON_FLAGS,
  MODEL_PREPARE_FLAGS,
  NONE,
  REDACTED_NAME,
  SETUP_CONFLICTS,
  SETUP_FLAGS,
  SKILLS_INSTALL_CONFLICTS,
  UNINSTALL_CONFLICTS,
  UNINSTALL_FLAGS,
  UPDATE_CONFLICTS,
  descriptor
} from './command-summary-descriptor-options'
import type { SummaryDescriptor } from './command-summary-descriptor-types'

export const TOP_LEVEL_DESCRIPTORS: readonly SummaryDescriptor[] = [
  descriptor(['mesh-llm', 'setup'], SETUP_FLAGS, NONE, false, 'none', SETUP_CONFLICTS),
  descriptor(['mesh-llm', 'uninstall'], UNINSTALL_FLAGS, ['--binary-path'], false, 'none', UNINSTALL_CONFLICTS),
  descriptor(['mesh-llm', 'download'], DRAFT, REDACTED_NAME),
  descriptor(['mesh-llm', 'update'], ['--detect-flavor'], ['--version', '--flavor'], false, 'none', UPDATE_CONFLICTS),
  descriptor(['mesh-llm', 'status'], NONE, NONE, true),
  descriptor(['mesh-llm', 'load'], NONE, REDACTED_NAME, true),
  descriptor(['mesh-llm', 'unload'], NONE, REDACTED_NAME, true),
  descriptor(['mesh-llm', 'discover'], DISCOVER_FLAGS, ['--name', '--model', '--min-vram', '--region', '--relay']),
  descriptor(['mesh-llm', 'rotate-key']),
  descriptor(['mesh-llm', 'goose'], NONE, ['--model'], true),
  descriptor(['mesh-llm', 'claude'], NONE, ['--model'], true),
  descriptor(['mesh-llm', 'pi'], ['--write'], ['--model', '--host']),
  descriptor(['mesh-llm', 'opencode'], ['--write'], ['--model', '--host']),
  descriptor(['mesh-llm', 'stop']),
  descriptor(['mesh-llm', 'external-plugin'], NONE, ['argv']),
  descriptor(['mesh-llm', 'model-prepare'], MODEL_PREPARE_FLAGS, [
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
  descriptor(['mesh-llm', 'gpus'], JSON_FLAGS),
  descriptor(['mesh-llm', 'gpus', 'detect'], JSON_FLAGS),
  descriptor(['mesh-llm', 'gpus', 'run-benchmark'], JSON_FLAGS, NONE, false, 'backend'),
  descriptor(['mesh-llm', 'config', 'validate'], JSON_FLAGS, ['--config-path']),
  descriptor(['mesh-llm', 'doctor'], JSON_FLAGS),
  descriptor(['mesh-llm', 'doctor', 'split'], JSON_FLAGS, ['--model-ref', '--output-dir'], true),
  descriptor(
    ['mesh-llm', 'skills', 'install'],
    ['--all', '--dry-run', '--force'],
    ['--agent'],
    false,
    'none',
    SKILLS_INSTALL_CONFLICTS
  )
]

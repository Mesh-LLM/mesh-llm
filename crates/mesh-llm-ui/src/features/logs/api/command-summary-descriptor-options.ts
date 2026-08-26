import type { SummaryDescriptor, SummaryRawKind } from './command-summary-descriptor-types'

export const NONE = [] as const
export const JSON_FLAGS = ['--json'] as const
export const YES_JSON = ['--yes', '--json'] as const
export const SETUP_FLAGS = [
  '--yes',
  '--no-interactive',
  '--service',
  '--no-service',
  '--skip-runtime',
  '--verbose'
] as const
export const UNINSTALL_FLAGS = [
  '--dry-run',
  '--yes',
  '--keep-cache',
  '--keep-service-files',
  '--purge-config',
  '--keep-config',
  '--json',
  '--verbose'
] as const
export const DRAFT = ['--draft'] as const
export const DISCOVER_FLAGS = ['--auto'] as const
export const MODEL_PACKAGE_FLAGS = [
  '--experimental',
  '--dry-run',
  '--confirm',
  '--follow',
  '--list',
  '--update-script',
  '--json'
] as const
export const MODEL_PREPARE_FLAGS = [
  '--dry-run',
  '--confirm',
  '--follow',
  '--json',
  '--list',
  '--update-script'
] as const
export const RUNTIME_LIST_FLAGS = ['--available', '--installed', '--json'] as const
export const RUNTIME_PRUNE_FLAGS = ['--active-only', '--json'] as const
export const AUTH_INIT_FLAGS = ['--force', '--no-passphrase', '--keychain'] as const
export const AUTH_ROTATE_NODE_FLAGS = ['--revoke-current'] as const
export const TUNE_FLAGS = [
  '--json',
  '--no-speculative-tune',
  '--apply',
  '--replace-existing',
  '--launch-args',
  '--debug-telemetry'
] as const
export const SEARCH_FLAGS = ['--gguf', '--mlx', '--catalog', '--json'] as const
export const MODEL_CERTIFY_FLAGS = ['--json', '--package-only'] as const
export const SETUP_CONFLICTS = [['--service', '--no-service']] as const
export const UNINSTALL_CONFLICTS = [['--purge-config', '--keep-config']] as const
export const AUTH_INIT_CONFLICTS = [['--no-passphrase', '--keychain']] as const
export const UPDATE_CONFLICTS = [['--flavor', '--detect-flavor']] as const
export const PLUGIN_INSTALL_CONFLICTS = [['reference', '--archive']] as const
export const SKILLS_INSTALL_CONFLICTS = [['--agent', '--all']] as const
export const MODEL_SEARCH_CONFLICTS = [['--gguf', '--mlx']] as const
export const TUNE_CONFLICTS = [
  ['--model', '--models'],
  ['--no-speculative-tune', '--speculative-types'],
  ['--no-speculative-tune', '--spec-draft-models'],
  ['--no-speculative-tune', '--spec-draft-max-tokens'],
  ['--no-speculative-tune', '--spec-draft-min-tokens'],
  ['--no-speculative-tune', '--spec-draft-acceptance-threshold'],
  ['--no-speculative-tune', '--spec-draft-split-probability'],
  ['--no-speculative-tune', '--spec-ngram-min'],
  ['--no-speculative-tune', '--spec-ngram-max']
] as const
export const RUNTIME_LIST_CONFLICTS = [['--available', '--installed']] as const
export const REMOTE_MODEL_CONFLICTS = [['--model', '--instance-id']] as const

export const REDACTED_NAME = ['name'] as const
export const REDACTED_MODEL = ['model'] as const
export const REDACTED_REMOTE = ['--endpoint'] as const
export const REDACTED_REMOTE_MODEL = ['--endpoint', '--model', '--profile'] as const
export const REDACTED_APPLY_CONFIG = ['--endpoint', '--expected-revision', '--config'] as const

const REDACTED_NONE = [] as const

export const descriptor = (
  path: readonly string[],
  booleans: readonly string[] = NONE,
  redacted: readonly string[] = REDACTED_NONE,
  hasPort = false,
  raw: SummaryRawKind = 'none',
  conflicts: readonly (readonly string[])[] = NONE
): SummaryDescriptor => ({ path, booleans, redacted, conflicts, hasPort, raw })

import {
  AUTH_INIT_CONFLICTS,
  AUTH_INIT_FLAGS,
  AUTH_ROTATE_NODE_FLAGS,
  NONE,
  descriptor
} from './command-summary-descriptor-options'
import type { SummaryDescriptor } from './command-summary-descriptor-types'

export const AUTH_DESCRIPTORS: readonly SummaryDescriptor[] = [
  descriptor(['mesh-llm', 'auth', 'init'], AUTH_INIT_FLAGS, ['--owner-key'], false, 'none', AUTH_INIT_CONFLICTS),
  descriptor(['mesh-llm', 'auth', 'status'], NONE, ['--owner-key', '--node-key', '--node-ownership', '--trust-store']),
  descriptor(['mesh-llm', 'auth', 'sign-node'], NONE, [
    '--owner-key',
    '--node-key',
    '--out',
    '--hostname-hint',
    '--node-label',
    '--expires-in-hours'
  ]),
  descriptor(['mesh-llm', 'auth', 'renew-node'], NONE, [
    '--owner-key',
    '--node-key',
    '--out',
    '--hostname-hint',
    '--node-label',
    '--expires-in-hours'
  ]),
  descriptor(['mesh-llm', 'auth', 'verify-node'], NONE, [
    '--file',
    '--node-id',
    '--trust-store',
    '--verify-trust-policy'
  ]),
  descriptor(['mesh-llm', 'auth', 'rotate-node'], AUTH_ROTATE_NODE_FLAGS, [
    '--owner-key',
    '--node-key',
    '--out',
    '--hostname-hint',
    '--node-label',
    '--expires-in-hours',
    '--reason',
    '--trust-store'
  ]),
  descriptor(['mesh-llm', 'auth', 'revoke-owner'], NONE, ['owner_id', '--reason', '--trust-store']),
  descriptor(['mesh-llm', 'auth', 'revoke-node'], NONE, ['--cert-id', '--node-id', '--reason', '--trust-store']),
  descriptor(['mesh-llm', 'auth', 'rotate-owner'], ['--no-passphrase', '--force'], ['--owner-key']),
  descriptor(['mesh-llm', 'auth', 'trust', 'add'], NONE, ['owner_id', '--label', '--trust-store']),
  descriptor(['mesh-llm', 'auth', 'trust', 'remove'], NONE, ['owner_id', '--trust-store']),
  descriptor(['mesh-llm', 'auth', 'trust', 'list'], NONE, ['--trust-store'])
]

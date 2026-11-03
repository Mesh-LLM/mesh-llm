import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  applyRuntimeControlConfig,
  changedPluginCommandApprovals,
  describePluginCommandApprovals
} from './config-adapter-http'
import type { RuntimeControlApplyInput, RuntimeControlConfigSnapshot } from './config-adapter-types'
import { setOwnerControlCredential } from './owner-control-credential'

vi.mock('./config-adapter-merge', () => ({
  mergeConfigurationIntoMeshConfig: () => ({ plugin: [{ name: 'scanner', command: '/bin/scan' }] })
}))

afterEach(() => vi.unstubAllGlobals())

describe('changedPluginCommandApprovals', () => {
  it('reports a newly added plugin command', () => {
    const approvals = changedPluginCommandApprovals(
      { plugin: [{ name: 'metrics', command: '/usr/bin/old' }] },
      {
        plugin: [
          { name: 'metrics', command: '/usr/bin/old' },
          { name: 'scanner', command: '/usr/local/bin/scanner' }
        ]
      }
    )

    expect(approvals).toEqual([{ plugin: 'scanner', command: '/usr/local/bin/scanner', invocation: { args: [] } }])
  })

  it('reports a changed plugin command', () => {
    const approvals = changedPluginCommandApprovals(
      { plugin: [{ name: 'metrics', command: '/usr/bin/old' }] },
      { plugin: [{ name: 'metrics', command: '/usr/bin/new' }] }
    )

    expect(approvals).toEqual([{ plugin: 'metrics', command: '/usr/bin/new', invocation: { args: [] } }])
  })

  it('reports changed arguments for an unchanged command', () => {
    expect(
      changedPluginCommandApprovals(
        { plugin: [{ name: 'scanner', command: '/bin/scan', args: ['old'] }] },
        { plugin: [{ name: 'scanner', command: '/bin/scan', args: ['new'] }] }
      )
    ).toEqual([{ plugin: 'scanner', command: '/bin/scan', invocation: { args: ['new'] } }])
  })

  it('shows the arguments of every matching entry when names repeat', () => {
    const config = {
      plugin: [
        { name: 'scanner', command: '/bin/scan', args: ['first'] },
        { name: 'scanner', command: '/bin/scan', args: ['second'] }
      ]
    }
    expect(
      describePluginCommandApprovals(config, [
        { plugin: 'scanner', command: '/bin/scan', invocation: { args: ['first'] } },
        { plugin: 'scanner', command: '/bin/scan', invocation: { args: ['second'] } }
      ])
    ).toEqual(['scanner: /bin/scan ["first"]', 'scanner: /bin/scan ["second"]'])
  })

  it('ignores unchanged commands and plugins without commands', () => {
    const approvals = changedPluginCommandApprovals(
      {
        plugin: [{ name: 'metrics', command: '/usr/bin/same' }, { name: 'ui' }]
      },
      {
        plugin: [{ name: 'metrics', command: '/usr/bin/same' }, { name: 'ui' }, { name: 'headless' }]
      }
    )

    expect(approvals).toEqual([])
  })

  it('handles missing plugin lists', () => {
    expect(changedPluginCommandApprovals(undefined, null)).toEqual([])
    expect(changedPluginCommandApprovals(null, { plugin: [{ name: 'scanner', command: '/bin/scan' }] })).toEqual([
      { plugin: 'scanner', command: '/bin/scan', invocation: { args: [] } }
    ])
  })
})

it('does not declare approval for an unconfirmed plugin command', async () => {
  setOwnerControlCredential('test')
  const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
    expect(JSON.parse(String(init?.body))).not.toHaveProperty('plugin_command_approvals')
    return new Response(JSON.stringify({ success: true, current_revision: 2 }))
  })
  vi.stubGlobal('fetch', fetchMock)
  await applyRuntimeControlConfig(
    'control://owner',
    { revision: 1, config: {} } as RuntimeControlConfigSnapshot,
    {} as RuntimeControlApplyInput
  )
  expect(fetchMock).toHaveBeenCalledTimes(1)
})

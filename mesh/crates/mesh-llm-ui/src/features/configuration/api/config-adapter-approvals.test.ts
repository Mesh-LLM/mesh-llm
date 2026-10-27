import { describe, expect, it } from 'vitest'
import { changedPluginCommandApprovals } from './config-adapter-http'

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

    expect(approvals).toEqual([{ plugin: 'scanner', command: '/usr/local/bin/scanner' }])
  })

  it('reports a changed plugin command', () => {
    const approvals = changedPluginCommandApprovals(
      { plugin: [{ name: 'metrics', command: '/usr/bin/old' }] },
      { plugin: [{ name: 'metrics', command: '/usr/bin/new' }] }
    )

    expect(approvals).toEqual([{ plugin: 'metrics', command: '/usr/bin/new' }])
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
      { plugin: 'scanner', command: '/bin/scan' }
    ])
  })
})

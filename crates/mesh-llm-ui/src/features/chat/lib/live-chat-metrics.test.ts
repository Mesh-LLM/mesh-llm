import { describe, expect, it } from 'vitest'

import { liveChatActionMetrics } from '@/features/chat/lib/live-chat-metrics'
import type { StatusPayload } from '@/lib/api/types'

function status(overrides: Partial<StatusPayload>): StatusPayload {
  return {
    node_id: 'local-node',
    node_state: 'serving',
    model_name: 'model',
    peers: [],
    models: [],
    my_vram_gb: 0,
    gpus: [],
    serving_models: [],
    ...overrides
  }
}

describe('liveChatActionMetrics', () => {
  it('returns nothing until live status has loaded', () => {
    expect(liveChatActionMetrics(undefined)).toEqual([])
  })

  it('counts every node and sums the capacity each one advertises, not the rated GPU class', () => {
    const metrics = liveChatActionMetrics(
      status({
        my_vram_gb: 115.448725504,
        gpus: [{ name: 'Apple M4 Max', rated_vram_gb: 128, vram_bytes: 115_448_725_504 }],
        peers: [
          {
            id: 'carrack',
            role: 'Host',
            state: 'serving',
            models: [],
            vram_gb: 44.02970624,
            gpus: [
              { name: 'RTX 5090', rated_vram_gb: 32, vram_bytes: 34_190_917_632 },
              { name: 'RTX 3080', rated_vram_gb: 10, vram_bytes: 10_737_418_240 }
            ]
          }
        ]
      })
    )

    expect(metrics).toEqual([
      { id: 'nodes', icon: 'cpu', label: '2 nodes' },
      { id: 'vram', icon: 'hard-drive', label: '159.5 GB' }
    ])
  })

  it('uses the singular label for a lone node', () => {
    expect(liveChatActionMetrics(status({ my_vram_gb: 24 }))).toEqual([
      { id: 'nodes', icon: 'cpu', label: '1 node' },
      { id: 'vram', icon: 'hard-drive', label: '24.0 GB' }
    ])
  })

  it('counts client nodes but excludes their capacity, like the scheduler does', () => {
    const metrics = liveChatActionMetrics(
      status({
        my_vram_gb: 115.4,
        peers: [
          { id: 'host', role: 'Host', state: 'serving', models: [], vram_gb: 44 },
          { id: 'laptop', role: 'Client', state: 'client', models: [], vram_gb: 24 }
        ]
      })
    )

    expect(metrics).toEqual([
      { id: 'nodes', icon: 'cpu', label: '3 nodes' },
      { id: 'vram', icon: 'hard-drive', label: '159.4 GB' }
    ])
  })

  it('suppresses local capacity when this node is a client', () => {
    const metrics = liveChatActionMetrics(
      status({
        node_state: 'client',
        my_vram_gb: 115.4,
        peers: [{ id: 'host', role: 'Host', state: 'serving', models: [], vram_gb: 44 }]
      })
    )

    expect(metrics[1]).toEqual({ id: 'vram', icon: 'hard-drive', label: '44.0 GB' })
  })
})

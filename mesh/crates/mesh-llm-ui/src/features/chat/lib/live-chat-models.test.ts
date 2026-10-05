import { statusBackedChatModels } from '@/features/chat/lib/live-chat-models'
import type { StatusPayload } from '@/lib/api/types'

function status(overrides: Partial<StatusPayload>): StatusPayload {
  return {
    node_id: 'local-node',
    node_state: 'client',
    model_name: '',
    peers: [],
    models: [],
    my_vram_gb: 0,
    gpus: [],
    serving_models: [],
    ...overrides
  }
}

describe('statusBackedChatModels', () => {
  it('uses peer-hosted models and only falls back to assigned models for legacy peers', () => {
    const models = statusBackedChatModels(
      status({
        peers: [
          {
            hosted_models: [],
            hosted_models_known: true,
            serving_models: ['assigned-but-not-hosted']
          },
          {
            hosted_models: [],
            hosted_models_known: false,
            serving_models: ['legacy-routable']
          },
          {
            hosted_models: [],
            serving_models: ['legacy-status-api-routable']
          },
          {
            hosted_models: ['hosted-routable'],
            hosted_models_known: true,
            serving_models: ['assigned-copy']
          }
        ]
      })
    )

    expect(models.map((model) => model.name)).toEqual([
      'hosted-routable',
      'legacy-routable',
      'legacy-status-api-routable'
    ])
  })

  it('includes only warm local model entries while retaining legacy strings', () => {
    const models = statusBackedChatModels(
      status({
        node_state: 'serving',
        serving_models: [
          { name: 'warm-local', node_id: 'local-node', status: 'warm' },
          { name: 'loading-local', node_id: 'local-node', status: 'loading' },
          { name: 'unloading-local', node_id: 'local-node', status: 'unloading' },
          'legacy-local',
          'mesh'
        ]
      })
    )

    expect(models.map((model) => model.name)).toEqual(['legacy-local', 'warm-local'])
  })
})

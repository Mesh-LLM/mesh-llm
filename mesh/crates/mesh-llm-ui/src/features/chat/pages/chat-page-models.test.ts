import { describe, expect, it } from 'vitest'

import type { ModelSummary } from '@/features/app-tabs/types'
import { isChatSelectableModel } from '@/features/chat/pages/chat-page-models'

function model(overrides: Partial<ModelSummary> = {}): ModelSummary {
  return {
    name: 'meshllm/chat-model:Q4_K_M',
    family: 'meshllm',
    size: '8.2 GB',
    context: '128K',
    status: 'warm',
    tags: [],
    ...overrides
  }
}

describe('isChatSelectableModel', () => {
  it('keeps a warm model that advertises causal generation', () => {
    expect(isChatSelectableModel(model({ workloadClass: 'causal_generation' }))).toBe(true)
  })

  it('keeps a warm encoder-decoder model, which the host admits for generation', () => {
    expect(isChatSelectableModel(model({ workloadClass: 'encoder_decoder' }))).toBe(true)
  })

  it('keeps a warm model from a peer that predates workload advertisement', () => {
    expect(isChatSelectableModel(model())).toBe(true)
  })

  it('leaves a warm decision-only System One model out of the picker', () => {
    const laya = model({
      name: 'meshllm/laya-multilingual-F16-GGUF:F16',
      fullId: 'meshllm/laya-multilingual-F16-GGUF:F16',
      workloadClass: 'decision'
    })

    expect(isChatSelectableModel(laya)).toBe(false)
  })

  it.each(['embedding', 'rerank', 'speech_synthesis', 'speech_recognition', 'unknown'])(
    'leaves a warm %s model out of the picker',
    (workloadClass) => {
      expect(isChatSelectableModel(model({ workloadClass }))).toBe(false)
    }
  )

  it('still leaves non-warm models out regardless of workload class', () => {
    expect(isChatSelectableModel(model({ status: 'cold' as ModelSummary['status'] }))).toBe(false)
    expect(isChatSelectableModel(model({ status: 'offline' }))).toBe(false)
    expect(isChatSelectableModel(model({ status: 'warming' }))).toBe(false)
  })
})

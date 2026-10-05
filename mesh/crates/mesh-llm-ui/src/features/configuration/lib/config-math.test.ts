import { describe, expect, it } from 'vitest'
import {
  canFitModelInContainer,
  containerAvailableGB,
  containerTotalGB,
  findModel,
  findPreferredModelFitContainerIdx,
  hasConfigurablePlacement,
  modelNeedGB,
  modelWeightsGB,
  nodeReservedGB,
  nodeUsableGB
} from '@/features/configuration/lib/config-math'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

describe('configuration memory math', () => {
  it('treats reserved VRAM as unavailable capacity for fit checks', () => {
    const model = findModel('qwen4')
    const node: ConfigNode = {
      id: 'node-test',
      hostname: 'test-node',
      region: 'lab',
      status: 'online',
      cpu: 'test cpu',
      ramGB: 64,
      placement: 'separate',
      gpus: [{ idx: 0, name: 'test gpu', totalGB: 3, reservedGB: 0.5 }]
    }

    if (!model) throw new Error('Expected qwen4 test model')

    expect(nodeReservedGB(node)).toBe(0.5)
    expect(nodeUsableGB(node)).toBe(2.5)
    expect(containerAvailableGB([], node, 0)).toBe(2.5)
    expect(canFitModelInContainer(model, node, [], 0, 4096)).toBe(false)
  })

  it('uses system-reported capacity for fit math while preserving rated total separately', () => {
    const node: ConfigNode = {
      id: 'node-test',
      hostname: 'test-node',
      region: 'lab',
      status: 'online',
      cpu: 'test cpu',
      ramGB: 64,
      placement: 'separate',
      gpus: [{ idx: 0, name: 'RTX 5090', totalGB: 32, systemTotalGB: 34.36, reservedGB: 0.54 }]
    }

    expect(containerTotalGB(node, 0)).toBe(34.36)
    expect(nodeUsableGB(node)).toBeCloseTo(33.82)
  })

  it('only allows placement changes for discrete multi-GPU nodes', () => {
    const singleGpuNode: ConfigNode = {
      id: 'node-single',
      hostname: 'single-node',
      region: 'lab',
      status: 'online',
      cpu: 'test cpu',
      ramGB: 64,
      placement: 'separate',
      memoryTopology: 'discrete',
      gpus: [{ idx: 0, name: 'test gpu', totalGB: 24 }]
    }
    const multiGpuNode: ConfigNode = {
      ...singleGpuNode,
      id: 'node-multi',
      hostname: 'multi-node',
      gpus: [...singleGpuNode.gpus, { idx: 1, name: 'test gpu', totalGB: 24 }]
    }
    const unifiedNode: ConfigNode = {
      ...singleGpuNode,
      id: 'node-unified',
      hostname: 'unified-node',
      region: 'unified',
      placement: 'pooled',
      memoryTopology: 'unified',
      gpus: [{ idx: 0, name: 'unified memory', totalGB: 48 }]
    }

    expect(hasConfigurablePlacement(singleGpuNode)).toBe(false)
    expect(hasConfigurablePlacement(multiGpuNode)).toBe(true)
    expect(hasConfigurablePlacement(unifiedNode)).toBe(false)
  })

  it('prefers the selected GPU when adding a catalog model', () => {
    const model = findModel('qwen4')
    const node: ConfigNode = {
      id: 'node-test',
      hostname: 'test-node',
      region: 'lab',
      status: 'online',
      cpu: 'test cpu',
      ramGB: 64,
      placement: 'separate',
      gpus: [
        { idx: 0, name: 'small gpu', totalGB: 8 },
        { idx: 1, name: 'selected gpu', totalGB: 8 }
      ]
    }

    if (!model) throw new Error('Expected qwen4 test model')

    expect(findPreferredModelFitContainerIdx(model, node, [], 1)).toBe(1)
  })

  it('uses the shared model weight helper when catalog memory size is missing', () => {
    const model: ConfigModel = {
      id: 'local-gguf/sha256-example',
      name: 'local-gguf/sha256-example',
      family: 'local-gguf',
      paramsB: 0,
      quant: 'Q4_K_M',
      sizeGB: 0,
      diskGB: 17.4,
      ctxMaxK: 256,
      moe: false,
      vision: false,
      tags: []
    }

    expect(modelWeightsGB(model)).toBe(17.4)
    expect(modelNeedGB(model, 4096)).toBeGreaterThan(17.4)
  })

  it('falls back to the first fitting GPU when the selected GPU lacks capacity', () => {
    const model = findModel('phi4')
    const node: ConfigNode = {
      id: 'node-test',
      hostname: 'test-node',
      region: 'lab',
      status: 'online',
      cpu: 'test cpu',
      ramGB: 64,
      placement: 'separate',
      gpus: [
        { idx: 0, name: 'full gpu', totalGB: 5, reservedGB: 0.1 },
        { idx: 1, name: 'open gpu', totalGB: 8 }
      ]
    }

    if (!model) throw new Error('Expected phi4 test model')

    expect(findPreferredModelFitContainerIdx(model, node, [], 0)).toBe(1)
  })

  it('returns null when no GPU can fit a catalog model', () => {
    const model = findModel('mixtral')
    const node: ConfigNode = {
      id: 'node-test',
      hostname: 'test-node',
      region: 'lab',
      status: 'online',
      cpu: 'test cpu',
      ramGB: 64,
      placement: 'separate',
      gpus: [
        { idx: 0, name: 'small gpu', totalGB: 24 },
        { idx: 1, name: 'other gpu', totalGB: 24 }
      ]
    }
    const assigns: ConfigAssign[] = []

    if (!model) throw new Error('Expected mixtral test model')

    expect(findPreferredModelFitContainerIdx(model, node, assigns, 0)).toBeNull()
  })
})

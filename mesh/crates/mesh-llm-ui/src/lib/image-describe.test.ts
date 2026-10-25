import { afterEach, describe, expect, it, vi } from 'vitest'
import { extractObjectLabels, loadWithBackendFallback, visionBackendCandidates } from '@/lib/image-describe'

describe('extractObjectLabels', () => {
  it('returns [] for empty input', () => {
    expect(extractObjectLabels('', '')).toEqual([])
    expect(extractObjectLabels('   ', 'caption')).toEqual([])
  })

  it('splits on newlines, commas, and semicolons', () => {
    const raw = 'red car\nblue bicycle, pedestrian; traffic light'
    expect(extractObjectLabels(raw, '')).toEqual(['red car', 'blue bicycle', 'pedestrian', 'traffic light'])
  })

  it('treats florence bbox markers as label separators', () => {
    const raw = 'golden retriever<loc_12><loc_34><loc_56><loc_78>blue collar<loc_1><loc_2><loc_3><loc_4>'
    expect(extractObjectLabels(raw, '')).toEqual(['golden retriever', 'blue collar'])
  })

  it('dedupes case-insensitively', () => {
    const raw = 'Cat\ncat\nCAT\ndog'
    expect(extractObjectLabels(raw, '')).toEqual(['Cat', 'dog'])
  })

  it('drops labels already present in the caption', () => {
    const caption = 'A golden retriever sits on a green lawn.'
    const raw = 'golden retriever, blue collar, red brick wall'
    expect(extractObjectLabels(raw, caption)).toEqual(['blue collar', 'red brick wall'])
  })

  it('caps the list at 12 labels', () => {
    const raw = Array.from({ length: 20 }, (_, i) => `item ${i}`).join(', ')
    expect(extractObjectLabels(raw, '')).toHaveLength(12)
  })

  it('returns [] when everything is in the caption', () => {
    const caption = 'red car driving past a blue bicycle'
    const raw = 'red car; blue bicycle'
    expect(extractObjectLabels(raw, caption)).toEqual([])
  })
})

describe('visionBackendCandidates', () => {
  afterEach(() => {
    delete (navigator as { gpu?: unknown }).gpu
  })

  it('offers only WASM when WebGPU is absent', () => {
    expect(visionBackendCandidates()).toEqual([{ device: 'wasm' }])
  })

  it('prefers WebGPU with a WASM fallback when WebGPU exists', () => {
    Object.defineProperty(navigator, 'gpu', { value: {}, configurable: true })
    expect(visionBackendCandidates()).toEqual([{ device: 'webgpu' }, { device: 'wasm' }])
  })

  it('ignores a null navigator.gpu', () => {
    Object.defineProperty(navigator, 'gpu', { value: null, configurable: true })
    expect(visionBackendCandidates()).toEqual([{ device: 'wasm' }])
  })
})

describe('loadWithBackendFallback', () => {
  it('returns the first successful load without trying later candidates', async () => {
    const load = vi.fn().mockResolvedValue('pipeline')
    await expect(loadWithBackendFallback([{ device: 'webgpu' }, { device: 'wasm' }], load)).resolves.toBe('pipeline')
    expect(load).toHaveBeenCalledTimes(1)
    expect(load).toHaveBeenCalledWith({ device: 'webgpu' })
  })

  it('falls through to the next candidate when one fails', async () => {
    const attempts: string[] = []
    const result = await loadWithBackendFallback([{ device: 'webgpu' }, { device: 'wasm' }], async (backend) => {
      attempts.push(backend.device)
      if (backend.device === 'webgpu') throw new Error('no adapter')
      return 'pipeline'
    })
    expect(result).toBe('pipeline')
    expect(attempts).toEqual(['webgpu', 'wasm'])
  })

  it('rethrows the original error when a single candidate fails', async () => {
    const wasmError = new Error('wasm failed')
    await expect(
      loadWithBackendFallback([{ device: 'wasm' }], async () => {
        throw wasmError
      })
    ).rejects.toBe(wasmError)
  })

  it('aggregates every error when multiple candidates fail', async () => {
    const webgpuError = new Error('no adapter')
    const wasmError = new Error('wasm failed')
    const failure = loadWithBackendFallback([{ device: 'webgpu' }, { device: 'wasm' }], async (backend) => {
      throw backend.device === 'webgpu' ? webgpuError : wasmError
    })
    await expect(failure).rejects.toBeInstanceOf(AggregateError)
    await expect(failure).rejects.toMatchObject({ errors: [webgpuError, wasmError] })
  })

  it('rejects with an Error when there are no candidates', async () => {
    const load = vi.fn()
    await expect(loadWithBackendFallback([], load)).rejects.toThrow('No vision backend candidates')
    expect(load).not.toHaveBeenCalled()
  })
})

describe('describeImage inference-time fallback', () => {
  afterEach(() => {
    delete (navigator as { gpu?: unknown }).gpu
    vi.restoreAllMocks()
    vi.resetModules()
  })

  function stubWebGpuPresent() {
    Object.defineProperty(navigator, 'gpu', { value: {}, configurable: true })
  }

  function mockTransformersModule(
    generateForDevice: (device: string) => Promise<unknown>,
    mockOptions?: { failImageLoad?: boolean }
  ) {
    const fromPretrained = vi.fn(async (_modelId: string, options: { device: string }) => ({
      generate: () => generateForDevice(options.device)
    }))
    vi.doMock('@huggingface/transformers', () => ({
      Florence2ForConditionalGeneration: { from_pretrained: fromPretrained },
      AutoProcessor: { from_pretrained: vi.fn(async () => async () => ({ input_ids: { dims: [1, 4] } })) },
      AutoTokenizer: { from_pretrained: vi.fn(async () => ({ batch_decode: () => ['a caption'] })) },
      RawImage: {
        fromURL: vi.fn(async () => {
          if (mockOptions?.failImageLoad) throw new Error('bad image')
          return {}
        })
      }
    }))
    return fromPretrained
  }

  it('drops the WebGPU pipeline and retries on WASM when generate fails', async () => {
    stubWebGpuPresent()
    const fromPretrained = mockTransformersModule(async (device) => {
      if (device === 'webgpu') throw new Error('GPU OOM')
      return { slice: () => [] }
    })
    const { describeImage } = await import('@/lib/image-describe')

    const result = await describeImage('blob:fake')

    expect(result.description).toBe('a caption')
    expect(fromPretrained.mock.calls.map((call) => call[1].device)).toEqual(['webgpu', 'wasm'])
  })

  it('propagates the WASM error when inference fails on both backends', async () => {
    stubWebGpuPresent()
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    mockTransformersModule(async (device) => {
      throw new Error(`${device} generate failed`)
    })
    const { describeImage } = await import('@/lib/image-describe')

    await expect(describeImage('blob:fake')).rejects.toThrow('wasm generate failed')
    expect(warn).toHaveBeenCalledWith(
      '[image-describe] WebGPU inference failed, retrying on WASM:',
      expect.objectContaining({ message: 'webgpu generate failed' })
    )
  })

  it('propagates image-load failures without a backend retry', async () => {
    stubWebGpuPresent()
    const fromPretrained = mockTransformersModule(async () => ({ slice: () => [] }), {
      failImageLoad: true
    })
    const { describeImage } = await import('@/lib/image-describe')

    await expect(describeImage('blob:fake')).rejects.toThrow('bad image')
    expect(fromPretrained).toHaveBeenCalledTimes(1)
    expect(fromPretrained.mock.calls[0]?.[1].device).toBe('webgpu')
  })
})

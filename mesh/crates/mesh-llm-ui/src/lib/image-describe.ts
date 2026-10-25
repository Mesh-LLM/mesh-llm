/**
 * Browser-side image description via Transformers.js + Florence-2.
 *
 * The model is loaded only when an image attachment needs description. Its
 * output is injected as text so regular text models can still reason about
 * attached images, matching the legacy console behavior.
 *
 * Bundle policy: the Transformers.js JS module is dynamic-imported (lazy code
 * split). The ONNX Runtime WASM (~21 MB) is NOT bundled — the
 * stripBundledOnnxWasm Vite plugin keeps it out of dist/, and Transformers.js
 * fetches it from jsDelivr on first use. This keeps the embedded UI dist small
 * (the WASM is the single biggest file in the binary's `include_dir!` blob) at
 * the cost of one CDN fetch the first time a user attaches an image. The model
 * weights themselves were already CDN-streamed from HuggingFace; this just
 * applies the same policy to the runtime.
 *
 * Do not override `env.backends.onnx.wasm.wasmPaths`: upstream already
 * defaults it to the correct jsDelivr URL for its onnxruntime-web build. A
 * hand-pinned override here drifted from package.json and pointed at a dist/
 * that never hosted the WASM, 404ing every image description.
 *
 * Device policy: prefer WebGPU (GPU offload instead of saturating every CPU
 * core per image); fall back to WASM when WebGPU is unavailable or fails to
 * initialize. Both paths run the same fp32 weights, so results are equivalent
 * and the fallback reuses already-downloaded files. A WebGPU session that
 * loads but later fails at inference time (driver bug, GPU OOM) is dropped
 * and retried once on WASM. fp16 is deliberately not used: the upstream fp16
 * export of this model fails onnxruntime validation ("Subgraph output
 * (logits) is an outer scope value being returned directly"), and the failed
 * attempt still downloads the full fp16 weight set before throwing — every
 * WebGPU user would pay hundreds of MB of wasted downloads on first use.
 */

import type { Tensor } from '@huggingface/transformers'

let pipelineCache: DescriptionPipeline | null = null
let loadingPromise: Promise<DescriptionPipeline> | null = null
let activeDevice: VisionBackend['device'] | null = null

const MODEL_ID = 'onnx-community/Florence-2-base-ft'

type DescriptionPipeline = Awaited<ReturnType<typeof createDescriptionPipeline>>
type LoadedImage = Awaited<ReturnType<DescriptionPipeline['RawImage']['fromURL']>>
type SliceableTensor = { slice: (start: null, end: [number, null]) => Tensor }

export type VisionBackend = {
  device: 'webgpu' | 'wasm'
}

const WASM_BACKEND: VisionBackend = { device: 'wasm' }

export function visionBackendCandidates(): VisionBackend[] {
  const hasWebGpu = typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu != null
  return hasWebGpu ? [{ device: 'webgpu' }, WASM_BACKEND] : [WASM_BACKEND]
}

export async function loadWithBackendFallback<T>(
  candidates: VisionBackend[],
  load: (backend: VisionBackend) => Promise<T>
): Promise<T> {
  if (candidates.length === 0) throw new Error('No vision backend candidates')
  const errors: unknown[] = []
  for (const candidate of candidates) {
    try {
      return await load(candidate)
    } catch (error) {
      errors.push(error)
    }
  }
  // A single candidate fails often (WASM-only browsers) — keep its original
  // error. With several, keep them all: the WebGPU error usually explains why
  // the preferred path failed and is the one wanted in bug reports.
  if (errors.length === 1) throw errors[0]
  throw new AggregateError(errors, 'All vision backends failed')
}

async function createDescriptionPipeline(backend: VisionBackend) {
  const { Florence2ForConditionalGeneration, AutoProcessor, AutoTokenizer, RawImage } =
    await import('@huggingface/transformers')

  const [model, processor, tokenizer] = await Promise.all([
    Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: 'fp32',
      device: backend.device
    }),
    AutoProcessor.from_pretrained(MODEL_ID),
    AutoTokenizer.from_pretrained(MODEL_ID)
  ])

  return { model, processor, tokenizer, RawImage }
}

function resetDescriptionPipeline() {
  pipelineCache = null
  loadingPromise = null
  activeDevice = null
}

async function getDescriptionPipeline(
  candidates: VisionBackend[] = visionBackendCandidates()
): Promise<DescriptionPipeline> {
  if (pipelineCache) return pipelineCache
  if (loadingPromise) return loadingPromise

  loadingPromise = (async () => {
    try {
      pipelineCache = await loadWithBackendFallback(candidates, async (backend) => {
        const pipeline = await createDescriptionPipeline(backend)
        activeDevice = backend.device
        return pipeline
      })
      return pipelineCache
    } catch (error) {
      resetDescriptionPipeline()
      throw error
    }
  })()

  return loadingPromise
}

export type ImageDescriptionResult = {
  description: string
  ocrText: string | null
  objects: string[]
  combinedText: string
}

let pipelineQueue: Promise<unknown> = Promise.resolve()
function enqueue<T>(task: () => Promise<T>): Promise<T> {
  const next = pipelineQueue.then(task, task)
  pipelineQueue = next.catch(() => undefined)
  return next
}

export async function describeImage(
  imageSource: string,
  onProgress?: (message: string) => void
): Promise<ImageDescriptionResult> {
  return enqueue(() => describeImageInternal(imageSource, onProgress))
}

async function describeImageInternal(
  imageSource: string,
  onProgress?: (message: string) => void,
  candidates: VisionBackend[] = visionBackendCandidates()
): Promise<ImageDescriptionResult> {
  const modelAlreadyLoaded = pipelineCache != null
  if (!modelAlreadyLoaded) onProgress?.('Downloading vision model...')
  const pipeline = await getDescriptionPipeline(candidates)
  if (!modelAlreadyLoaded) onProgress?.('Starting local vision model...')

  // Decode outside the retry boundary: an image-load failure is
  // backend-independent and must not trigger the WASM retry below.
  const image = await pipeline.RawImage.fromURL(imageSource)

  try {
    return await runDescription(pipeline, image, onProgress)
  } catch (error) {
    // A WebGPU session can load fine and still fail at inference time (driver
    // bug, GPU OOM). Drop it and retry once on WASM; a WASM failure or a
    // non-backend error propagates unchanged.
    if (activeDevice !== 'webgpu') throw error
    console.warn('[image-describe] WebGPU inference failed, retrying on WASM:', error)
    resetDescriptionPipeline()
    return describeImageInternal(imageSource, onProgress, [WASM_BACKEND])
  }
}

async function runDescription(
  { model, processor, tokenizer }: DescriptionPipeline,
  image: LoadedImage,
  onProgress?: (message: string) => void
): Promise<ImageDescriptionResult> {
  onProgress?.('Processing image...')

  const captionPrompt = '<MORE_DETAILED_CAPTION>'
  const captionInputs = await processor(image, captionPrompt)
  const captionIds = await model.generate({
    ...captionInputs,
    max_new_tokens: 256
  })
  const captionGenerated = sliceGeneratedIds(captionIds, captionInputs.input_ids.dims.at(-1))
  const description = tokenizer.batch_decode(captionGenerated, { skip_special_tokens: true })[0]?.trim() ?? ''

  let objects: string[] = []
  try {
    const regionPrompt = '<DENSE_REGION_CAPTION>'
    const regionInputs = await processor(image, regionPrompt)
    const regionIds = await model.generate({
      ...regionInputs,
      max_new_tokens: 256
    })
    const regionGenerated = sliceGeneratedIds(regionIds, regionInputs.input_ids.dims.at(-1))
    const regionText = tokenizer.batch_decode(regionGenerated, { skip_special_tokens: true })[0]?.trim() ?? ''
    objects = extractObjectLabels(regionText, description)
  } catch {
    // Region captioning is best-effort; caption + OCR still land.
  }

  let ocrText: string | null = null
  try {
    const ocrPrompt = '<OCR>'
    const ocrInputs = await processor(image, ocrPrompt)
    const ocrIds = await model.generate({
      ...ocrInputs,
      max_new_tokens: 256
    })
    const ocrGenerated = sliceGeneratedIds(ocrIds, ocrInputs.input_ids.dims.at(-1))
    const raw = tokenizer.batch_decode(ocrGenerated, { skip_special_tokens: true })[0]?.trim() ?? ''
    if (raw.length > 3) ocrText = raw
  } catch {
    // OCR is best-effort.
  }

  const parts: string[] = []
  if (description) parts.push(`[Image description: ${description}]`)
  if (objects.length) parts.push(`[Visible objects: ${objects.join('; ')}]`)
  if (ocrText) parts.push(`[Text visible in image: ${ocrText}]`)

  return {
    description,
    ocrText,
    objects,
    combinedText: parts.join('\n') || '[Unable to describe image]'
  }
}

function isSliceableTensor(value: unknown): value is SliceableTensor {
  return typeof value === 'object' && value !== null && 'slice' in value && typeof value.slice === 'function'
}

function sliceGeneratedIds(value: unknown, promptTokenCount: number | undefined): Tensor {
  if (!isSliceableTensor(value)) {
    throw new Error('Unexpected vision model output')
  }

  return value.slice(null, [promptTokenCount ?? 0, null])
}

export function extractObjectLabels(raw: string, caption: string): string[] {
  if (!raw) return []

  const stripped = raw.replace(/<loc_\d+>/g, '\n')
  const captionLower = caption.toLowerCase()
  const seen = new Set<string>()
  const labels: string[] = []

  for (const rawLabel of stripped.split(/[\n,;]+/)) {
    const label = rawLabel.replace(/\s+/g, ' ').trim()
    if (!label) continue

    const key = label.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    if (captionLower.includes(key)) continue

    labels.push(label)
    if (labels.length >= 12) break
  }

  return labels
}

export function canRunBrowserVision(): boolean {
  return typeof WebAssembly !== 'undefined'
}

export function isModelLoaded(): boolean {
  return pipelineCache != null
}

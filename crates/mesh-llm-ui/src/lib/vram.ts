const DECIMAL_GB_BYTES = 1_000_000_000
const GIB_BYTES = 1024 ** 3

const RATED_CAPACITY_GB_CLASSES = [
  1, 2, 3, 4, 6, 8, 10, 11, 12, 16, 18, 20, 22, 24, 32, 36, 40, 44, 48, 64, 80, 96, 128, 144, 160, 192, 256, 384, 512
] as const

export type VramGpuInput = {
  total_vram_gb?: number
  rated_vram_gb?: number
  vram_bytes?: number
  reserved_bytes?: number
  allocatable_vram_bytes?: number
}

function finitePositive(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

function nearestRatedClass(valueGB: number): number {
  return RATED_CAPACITY_GB_CLASSES.reduce((best, candidate) => {
    const bestError = Math.abs(valueGB - best) / best
    const candidateError = Math.abs(valueGB - candidate) / candidate
    return candidateError < bestError ? candidate : best
  })
}

function candidateRelativeError(bytes: number, capacityGB: number, unitBytes: number): number {
  return Math.abs(bytes - capacityGB * unitBytes) / (capacityGB * unitBytes)
}

export function decimalVramGBFromBytes(bytes: number | null | undefined): number | null {
  const value = finitePositive(bytes)
  return value == null ? null : value / DECIMAL_GB_BYTES
}

export function allocatableVramBytes(
  systemReportedBytes: number | null | undefined,
  reservedBytes?: number | null
): number | null {
  const total = finitePositive(systemReportedBytes)
  if (total == null) return null
  const reserved = finitePositive(reservedBytes) ?? 0
  return Math.max(0, total - reserved)
}

export function ratedVramGBFromBytes(bytes: number | null | undefined): number | null {
  const value = finitePositive(bytes)
  if (value == null) return null

  const decimalCandidate = nearestRatedClass(value / DECIMAL_GB_BYTES)
  const binaryCandidate = nearestRatedClass(value / GIB_BYTES)
  const decimalError = candidateRelativeError(value, decimalCandidate, DECIMAL_GB_BYTES)
  const binaryError = candidateRelativeError(value, binaryCandidate, GIB_BYTES)
  return decimalError <= binaryError ? decimalCandidate : binaryCandidate
}

export function gpuRatedVramGB(gpu: VramGpuInput): number | null {
  return finitePositive(gpu.rated_vram_gb) ?? finitePositive(gpu.total_vram_gb) ?? ratedVramGBFromBytes(gpu.vram_bytes)
}

export function gpuSystemReportedVramGB(gpu: VramGpuInput): number | null {
  return decimalVramGBFromBytes(gpu.vram_bytes) ?? finitePositive(gpu.total_vram_gb)
}

export function gpuReservedVramGB(gpu: VramGpuInput): number {
  return decimalVramGBFromBytes(gpu.reserved_bytes) ?? 0
}

export function gpuAllocatableVramGB(gpu: VramGpuInput): number | null {
  if (gpu.allocatable_vram_bytes != null) {
    return decimalVramGBFromBytes(gpu.allocatable_vram_bytes) ?? 0
  }
  const allocatable = allocatableVramBytes(gpu.vram_bytes, gpu.reserved_bytes)
  return decimalVramGBFromBytes(allocatable)
}

export function formatRatedVramGB(valueGB: number | null | undefined): string {
  const value = finitePositive(valueGB)
  if (value == null) return 'Unknown'
  return `${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)} GB`
}

export function formatRatedVramBytes(bytes: number | null | undefined): string {
  return formatRatedVramGB(ratedVramGBFromBytes(bytes))
}

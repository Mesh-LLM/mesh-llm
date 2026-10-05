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

export type VramNodeInput = {
  vram_gb?: number | null
  my_vram_gb?: number | null
  gpus?: VramGpuInput[] | null
  /** Client-role nodes consume capacity but never serve, so they contribute nothing to mesh totals. */
  client?: boolean
}

export type VramMeshInput = VramNodeInput & {
  peers?: VramNodeInput[] | null
}

function sumGpuVramGB(
  gpus: VramGpuInput[] | null | undefined,
  pick: (gpu: VramGpuInput) => number | null
): number | null {
  if (!gpus?.length) return null
  const total = gpus.reduce((sum, gpu) => sum + (pick(gpu) ?? 0), 0)
  return total > 0 ? total : null
}

/**
 * Rated capacity class summed across a node's GPU inventory (marketing GB, e.g. 32 + 10 = 42).
 * Display-only; never use it for fit math or mesh totals.
 */
export function nodeRatedVramGB(node: VramNodeInput): number | null {
  return sumGpuVramGB(node.gpus, gpuRatedVramGB)
}

/**
 * Capacity the node advertises to the mesh, in decimal GB. This is the figure
 * `/api/status` reports as `my_vram_gb` / `vram_gb` and the one the scheduler and
 * `doctor split` sum, so aggregates built from it agree with the API and CLI.
 * Falls back to per-GPU allocatable bytes, then to the rated class, only for
 * legacy payloads that carry no announced value. Client-role nodes yield null:
 * the scheduler excludes them from aggregate capacity, so the console must too.
 */
export function nodeAdvertisedVramGB(node: VramNodeInput): number | null {
  if (node.client) return null
  return (
    finitePositive(node.vram_gb) ??
    finitePositive(node.my_vram_gb) ??
    sumGpuVramGB(node.gpus, gpuAllocatableVramGB) ??
    nodeRatedVramGB(node)
  )
}

/** Advertised capacity of the local node plus every peer, in decimal GB. */
export function meshAdvertisedVramGB(mesh: VramMeshInput): number {
  const local = nodeAdvertisedVramGB(mesh) ?? 0
  return (mesh.peers ?? []).reduce((sum, peer) => sum + (nodeAdvertisedVramGB(peer) ?? 0), local)
}

/** Minimal shape of `/api/status` needed to build mesh capacity totals. */
export type VramStatusLike = {
  my_vram_gb?: number | null
  gpus?: VramGpuInput[] | null
  is_client?: boolean
  node_state?: string
  peers?: Array<VramNodeInput & { node_state?: string; state?: string; role?: string }> | null
}

/**
 * Capacity rule for peers: any of the three fields can mark a client. This is
 * intentionally separate from the dashboard's `resolvePeerRole`, which is a
 * display rule that returns `host` before it looks at state. Keep the two apart;
 * rows and totals must both call this one so they cannot disagree. The live API
 * emits `state` and `role` on peers; `node_state` is honoured only because the
 * UI's `PeerInfo` type and the adapter's state resolution already accept it.
 */
export function isClientPeer(peer: { node_state?: string; state?: string; role?: string }): boolean {
  return peer.node_state === 'client' || peer.state === 'client' || peer.role?.toLowerCase() === 'client'
}

/**
 * Builds the capacity input for a status payload, marking the local node and
 * any peer in the client role so they are left out of mesh totals. Both the
 * dashboard and the chat header go through this so they agree by construction.
 */
export function meshCapacityInputFromStatus(status: VramStatusLike): VramMeshInput {
  return {
    vram_gb: status.my_vram_gb,
    gpus: status.gpus,
    client: status.is_client === true || status.node_state === 'client',
    peers: (status.peers ?? []).map((peer) => ({
      vram_gb: peer.vram_gb,
      my_vram_gb: peer.my_vram_gb,
      gpus: peer.gpus,
      client: isClientPeer(peer)
    }))
  }
}

/** The itemized capacity a node advertises, as `/api/status` reports it (bytes). */
export type MemoryBreakdownInput = {
  total_bytes: number
  reserved_bytes: number
  platform_reserve_bytes?: number
  configured_reserve_bytes: number
  usable_bytes: number
  system_ram_bytes?: number
  ram_offload_bytes: number
}

/** The same breakdown in decimal GB, ready to display. */
export type MemoryBreakdownGB = {
  totalGB: number
  reservedGB: number
  platformReserveGB: number
  configuredReserveGB: number
  usableGB: number
  systemRamGB: number | null
  ramOffloadGB: number
}

function finiteNonNegative(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

function decimalGBOrZero(bytes: number | null | undefined): number {
  return (finiteNonNegative(bytes) ?? 0) / DECIMAL_GB_BYTES
}

/**
 * The advertised breakdown in decimal GB, or null when the node sent none or
 * the block is unusable. Zero reserves are real values, not gaps; only the
 * optional system RAM stays null when it was not reported. Totals do not go
 * through here: they keep `nodeAdvertisedVramGB`.
 */
export function memoryBreakdownGB(memory: MemoryBreakdownInput | null | undefined): MemoryBreakdownGB | null {
  if (!memory) return null
  if (finiteNonNegative(memory.total_bytes) == null || finiteNonNegative(memory.usable_bytes) == null) return null
  const systemRam = finiteNonNegative(memory.system_ram_bytes)
  return {
    totalGB: decimalGBOrZero(memory.total_bytes),
    reservedGB: decimalGBOrZero(memory.reserved_bytes),
    platformReserveGB: decimalGBOrZero(memory.platform_reserve_bytes),
    configuredReserveGB: decimalGBOrZero(memory.configured_reserve_bytes),
    usableGB: decimalGBOrZero(memory.usable_bytes),
    systemRamGB: systemRam == null ? null : systemRam / DECIMAL_GB_BYTES,
    ramOffloadGB: decimalGBOrZero(memory.ram_offload_bytes)
  }
}

/** Exact decimal GB with one decimal, for itemized values that are not capacity classes. */
export function formatDecimalVramGB(valueGB: number | null | undefined): string {
  const value = finiteNonNegative(valueGB)
  if (value == null) return 'Unknown'
  return `${value.toFixed(1)} GB`
}

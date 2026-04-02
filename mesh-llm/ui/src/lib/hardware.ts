export type ReportedGpu = {
  name: string;
  vram_bytes: number;
  reserved_bytes?: number;
};

export function shortenHardwareModelName(name: string) {
  return name
    .replace(/^NVIDIA GeForce\s+/i, '')
    .replace(/^NVIDIA Quadro\s+/i, '')
    .replace(/^NVIDIA\s+/i, '')
    .replace(/^AMD Radeon\s+/i, '')
    .replace(/^AMD\s+/i, '')
    .replace(/^Intel Arc\s+/i, '')
    .replace(/^Intel\s+/i, '')
    .replace(/^Apple\s+/i, '')
    .trim();
}

export function formatHardwareNames(
  gpus: ReportedGpu[] | undefined,
  options?: { shorten?: boolean },
) {
  const shorten = options?.shorten ?? false;

  return (gpus ?? [])
    .map((gpu) => gpu.name?.trim())
    .filter((name): name is string => Boolean(name))
    .map((name) => (shorten ? shortenHardwareModelName(name) : name));
}

export function hardwareVramBytes(
  vramGb: number | null | undefined,
  gpus?: ReportedGpu[],
) {
  const gpuBytes = (gpus ?? []).reduce(
    (total, gpu) => total + Math.max(0, gpu.vram_bytes || 0),
    0,
  );
  if (gpuBytes > 0) return gpuBytes;
  return Math.round(Math.max(0, vramGb ?? 0) * 1e9);
}

export function hardwareVramGb(
  vramGb: number | null | undefined,
  gpus?: ReportedGpu[],
) {
  return hardwareVramBytes(vramGb, gpus) / 1e9;
}

export interface GpuTarget {
  index: number;
  name: string;
  vramBytes: number;
  reservedBytes: number;
  label: string;
}

export function gpuTargets(gpus: ReportedGpu[] | undefined): GpuTarget[] {
  return (gpus ?? []).map((gpu, index) => ({
    index,
    name: gpu.name,
    vramBytes: gpu.vram_bytes,
    reservedBytes: gpu.reserved_bytes ?? 0,
    label: `GPU ${index} · ${gpu.name} · ${(gpu.vram_bytes / 1e9).toFixed(1)} GB`,
  }));
}

export function isHomogenousGpuSet(gpus: ReportedGpu[] | undefined): boolean {
  const list = gpus ?? [];
  if (list.length <= 1) return true;
  const firstName = list[0].name;
  return list.every((gpu) => gpu.name === firstName);
}

export function fitsInPooled(modelBytes: number, gpus: ReportedGpu[] | undefined): boolean {
  const totalBytes = hardwareVramBytes(null, gpus ?? []);
  return totalBytes > 0 && modelBytes <= totalBytes;
}

export function fitsOnGpu(modelBytes: number, target: GpuTarget): boolean {
  return target.vramBytes > 0 && modelBytes <= target.vramBytes;
}

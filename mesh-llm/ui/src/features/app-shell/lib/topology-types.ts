export type TopologyNode = {
  id: string;
  vram: number;
  self: boolean;
  host: boolean;
  client: boolean;
  serving: string;
  servingModels: string[];
  statusLabel: string;
  latencyMs?: number | null;
  hostname?: string;
  isSoc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};

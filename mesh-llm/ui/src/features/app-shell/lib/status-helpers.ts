import type { MeshModel, Ownership, Peer, StatusPayload, ThemeMode } from "./status-types";
import type { TopologyNode } from "../../../topology/layouts/types";

export function modelDisplayName(model?: MeshModel | null) {
  if (!model) return "";
  return model.display_name || model.name;
}

export function shortName(name: string) {
  return (name || "").replace(/-Q\w+$/, "").replace(/-Instruct/, "");
}

export function peerAssignedModels(peer: Peer): string[] {
  return peer.serving_models?.filter(Boolean) ?? [];
}

export function peerRoutableModels(peer: Peer): string[] {
  const hosted = peer.hosted_models?.filter(Boolean) ?? [];
  if (peer.hosted_models_known === false) {
    return hosted.length ? hosted : peerAssignedModels(peer);
  }
  return hosted;
}

export function localRoutableModels(status: StatusPayload | null): string[] {
  if (!status || status.is_client) return [];
  const hosted = status.hosted_models?.filter(Boolean) ?? [];
  if (hosted.length > 0) return hosted;
  const serving = status.serving_models?.filter(Boolean) ?? [];
  if (serving.length > 0) return serving;
  return status.model_name ? [status.model_name] : [];
}

export function peerPrimaryModel(peer: Peer): string {
  return peerRoutableModels(peer)[0] ?? peerAssignedModels(peer)[0] ?? "";
}

export function overviewVramGb(isClient: boolean, vramGb?: number | null) {
  if (isClient) return 0;
  return Math.max(0, vramGb || 0);
}

export function peerStatusLabel(peer: Peer): string {
  if (peer.role === "Client") return "Client";
  if (peerRoutableModels(peer).some((model) => model !== "(idle)")) return "Serving";
  if (peerAssignedModels(peer).some((model) => model !== "(idle)")) return "Assigned";
  if (peer.role === "Host") return "Host";
  return "Idle";
}

export function meshGpuVram(status: StatusPayload | null) {
  if (!status) return 0;
  return (
    overviewVramGb(status.is_client, status.my_vram_gb) +
    (status.peers || []).reduce(
      (sum, peer) => sum + overviewVramGb(peer.role === "Client", peer.vram_gb),
      0,
    )
  );
}

export function ownershipTone(status?: string): "good" | "warn" | "bad" | "neutral" {
  switch (status) {
    case "verified":
      return "good";
    case "expired":
    case "untrusted_owner":
      return "warn";
    case "invalid_signature":
    case "mismatched_node_id":
    case "revoked_owner":
    case "revoked_cert":
    case "revoked_node_id":
      return "bad";
    default:
      return "neutral";
  }
}

export function ownershipStatusLabel(status?: string) {
  if (!status) return "Unknown";
  return status
    .split("_")
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(" ");
}

export function shortIdentity(value?: string | null, size = 12) {
  if (!value) return "n/a";
  return value.length <= size ? value : value.slice(0, size);
}

export function ownershipPrimaryLabel(owner?: Ownership | null) {
  if (!owner) return "Unsigned";
  if (owner.node_label) return owner.node_label;
  if (owner.owner_id) return shortIdentity(owner.owner_id, 16);
  return ownershipStatusLabel(owner.status);
}

export function formatLatency(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const ms = Math.round(Number(value));
  if (ms <= 0) return "<1 ms";
  return `${ms} ms`;
}

export function topologyStatusTone(
  status: string,
): "good" | "info" | "warn" | "bad" | "neutral" {
  if (status === "Serving" || status === "Serving (split)") return "good";
  if (status === "Client") return "info";
  if (status === "Host") return "info";
  if (status === "Idle" || status === "Standby") return "neutral";
  if (status === "Worker (split)") return "warn";
  return "neutral";
}

export function topologyStatusTooltip(status: string) {
  if (status === "Serving") {
    return "Actively serving a model.";
  }
  if (status === "Serving (split)") {
    return "Serving a split model with the mesh.";
  }
  if (status === "Worker (split)") {
    return "Contributing compute to a split model.";
  }
  if (status === "Host") {
    return "Coordinating requests for the mesh.";
  }
  if (status === "Client") {
    return "Sends requests, but does not contribute VRAM.";
  }
  if (status === "Idle" || status === "Standby") {
    return "Connected, but not serving a model.";
  }
  return "Current serving role.";
}

export function modelStatusTooltip(status?: string) {
  if (status === "warm") {
    return "Loaded and serving in the mesh.";
  }
  if (status === "cold") {
    return "Downloaded locally, but not currently serving.";
  }
  return "Current model availability in the mesh.";
}

export function uniqueModels(...groups: Array<string[] | undefined>): string[] {
  return [
    ...new Set(
      groups
        .flatMap((group) => group ?? [])
        .filter((model) => !!model && model !== "(idle)"),
    ),
  ];
}

export function formatGpuMemory(bytes?: number | null) {
  if (!bytes || !Number.isFinite(bytes)) return "Unknown";
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(0)} GB`;
}

export function trimGpuVendor(name: string) {
  return name
    .replace(/^NVIDIA GeForce\s+/i, "")
    .replace(/^NVIDIA Quadro\s+/i, "")
    .replace(/^NVIDIA\s+/i, "")
    .replace(/^AMD Radeon\s+/i, "")
    .replace(/^AMD\s+/i, "")
    .replace(/^Intel Arc\s+/i, "")
    .replace(/^Intel\s+/i, "")
    .replace(/^Apple\s+/i, "")
    .trim();
}

export function topologyNodeRole(node: Pick<TopologyNode, "client" | "host" | "serving">): string {
  if (node.client) return "Client";
  if (node.host) return "Host";
  if (node.serving && node.serving !== "(idle)") return "Worker";
  return "Idle";
}

export function readThemeMode(storageKey: string): ThemeMode {
  if (typeof window === "undefined") return "auto";
  const stored = window.localStorage.getItem(storageKey);
  return stored === "light" || stored === "dark" || stored === "auto"
    ? stored
    : "auto";
}

export function applyThemeMode(mode: ThemeMode) {
  if (typeof window === "undefined") return;
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const dark = mode === "dark" || (mode === "auto" && media.matches);
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.style.colorScheme =
    mode === "auto" ? "light dark" : dark ? "dark" : "light";
}

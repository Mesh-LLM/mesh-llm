import { useEffect, useMemo, useState } from "react";

export type MeshModel = {
  name: string;
  display_name?: string;
  status: "warm" | "cold" | string;
  node_count: number;
  mesh_vram_gb?: number;
  size_gb: number;
  architecture?: string;
  context_length?: number;
  quantization?: string;
  description?: string;
  multimodal?: boolean;
  multimodal_status?: "supported" | "none" | string;
  vision?: boolean;
  vision_status?: "supported" | "likely" | "none" | string;
  audio?: boolean;
  audio_status?: "supported" | "likely" | "none" | string;
  reasoning?: boolean;
  reasoning_status?: "supported" | "likely" | "none" | string;
  tool_use?: boolean;
  tool_use_status?: "supported" | "likely" | "none" | string;
  moe?: boolean;
  expert_count?: number;
  used_expert_count?: number;
  ranking_source?: string;
  ranking_origin?: string;
  ranking_prompt_count?: number;
  ranking_tokens?: number;
  ranking_layer_scope?: string;
  draft_model?: string;
  source_page_url?: string;
  fit_label?: string;
  fit_detail?: string;
  download_command?: string;
  run_command?: string;
  auto_command?: string;
  request_count?: number;
  last_active_secs_ago?: number;
  source_ref?: string;
  source_revision?: string;
  source_file?: string;
  active_nodes?: string[];
};

export type Ownership = {
  owner_id?: string;
  cert_id?: string;
  status: string;
  verified: boolean;
  expires_at_unix_ms?: number;
  node_label?: string;
  hostname_hint?: string;
};

export type Peer = {
  id: string;
  owner?: Ownership;
  role: string;
  models: string[];
  available_models?: string[];
  requested_models?: string[];
  vram_gb: number;
  serving_models?: string[];
  hosted_models?: string[];
  hosted_models_known?: boolean;
  rtt_ms?: number | null;
  hostname?: string;
  version?: string;
  is_soc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};

export type LocalInstance = {
  pid: number;
  api_port: number | null;
  version: string | null;
  started_at_unix: number;
  runtime_dir: string;
  is_self: boolean;
};

export type StatusPayload = {
  version?: string;
  latest_version?: string | null;
  node_id: string;
  owner?: Ownership;
  token: string;
  node_status: string;
  is_host: boolean;
  is_client: boolean;
  llama_ready: boolean;
  model_name: string;
  models?: string[];
  available_models?: string[];
  requested_models?: string[];
  serving_models?: string[];
  hosted_models?: string[];
  api_port: number;
  my_vram_gb: number;
  model_size_gb: number;
  mesh_name?: string | null;
  peers: Peer[];
  local_instances?: LocalInstance[];
  inflight_requests: number;
  launch_pi?: string | null;
  launch_goose?: string | null;
  nostr_discovery?: boolean;
  my_hostname?: string;
  my_is_soc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};

type ModelsPayload = {
  mesh_models: MeshModel[];
};

function modelCatalogKeyFromStatus(status: StatusPayload | null) {
  if (!status) return "";
  const local = [
    status.model_name,
    ...(status.models ?? []),
    ...(status.available_models ?? []),
    ...(status.requested_models ?? []),
    ...(status.serving_models ?? []),
    ...(status.hosted_models ?? []),
  ].join(",");
  const peers = [...(status.peers ?? [])]
    .map((peer) =>
      [
        peer.id,
        ...(peer.models ?? []),
        ...(peer.available_models ?? []),
        ...(peer.requested_models ?? []),
        ...(peer.serving_models ?? []),
        ...(peer.hosted_models ?? []),
      ].join(","),
    )
    .sort()
    .join("|");
  return `${status.node_id}::${local}::${peers}`;
}

export function useStatusStream() {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [modelsPayload, setModelsPayload] = useState<ModelsPayload | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    let statusEvents: EventSource | null = null;
    let reconnectTimer: number | null = null;
    let retryMs = 1000;
    const MAX_RETRY_MS = 15000;
    const reconnectStatusMessage =
      "Trying to reconnect automatically. Live updates will resume shortly.";

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const closeStatusEvents = () => {
      if (!statusEvents) return;
      statusEvents.onopen = null;
      statusEvents.onmessage = null;
      statusEvents.onerror = null;
      statusEvents.close();
      statusEvents = null;
    };

    const loadStatus = () => {
      fetch("/api/status")
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json() as Promise<StatusPayload>;
        })
        .then((data) => {
          if (stop) return;
          setStatus(data);
          setStatusError(null);
        })
        .catch((err: Error) => {
          if (stop) return;
          setStatusError(reconnectStatusMessage);
          console.warn("Failed to fetch /api/status:", err.message);
        });
    };

    const scheduleReconnect = () => {
      if (stop || reconnectTimer !== null) return;
      setStatusError(reconnectStatusMessage);
      console.warn("Connection lost. Reconnecting...");
      closeStatusEvents();
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        connectStatusEvents();
        retryMs = Math.min(retryMs * 2, MAX_RETRY_MS);
      }, retryMs);
    };

    const connectStatusEvents = () => {
      if (stop || statusEvents) return;

      let source: EventSource;
      try {
        source = new EventSource("/api/events");
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "failed to create EventSource";
        console.warn("Failed to connect status stream:", message);
        scheduleReconnect();
        return;
      }

      statusEvents = source;
      source.onopen = () => {
        if (stop) return;
        retryMs = 1000;
        setStatusError(null);
        loadStatus();
      };
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as StatusPayload;
          setStatus(payload);
          setStatusError(null);
        } catch {
          // ignore malformed status event
        }
      };
      source.onerror = () => {
        if (stop) return;
        scheduleReconnect();
      };
    };

    loadStatus();
    connectStatusEvents();

    return () => {
      stop = true;
      clearReconnectTimer();
      closeStatusEvents();
    };
  }, []);

  const modelCatalogKey = useMemo(() => modelCatalogKeyFromStatus(status), [status]);

  useEffect(() => {
    if (!modelCatalogKey) {
      setModelsPayload(null);
      setModelsLoading(false);
      return;
    }

    const controller = new AbortController();
    let cancelled = false;
    setModelsLoading(true);

    fetch("/api/models", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<ModelsPayload>;
      })
      .then((data) => {
        if (cancelled) return;
        setModelsPayload(data);
      })
      .catch((err: Error) => {
        if (cancelled || err.name === "AbortError") return;
        console.warn("Failed to fetch /api/models:", err.message);
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [modelCatalogKey]);

  return {
    status,
    statusError,
    meshModels: modelsPayload?.mesh_models ?? [],
    modelsLoading,
  };
}

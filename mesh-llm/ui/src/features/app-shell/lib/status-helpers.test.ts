import { describe, expect, it } from "vitest";

import {
  formatLiveNodeState,
  localRoutableModels,
  topologyStatusTone,
  topologyStatusTooltip,
} from "./status-helpers";
import type { LiveNodeState, StatusPayload } from "./status-types";

describe("live node state helpers", () => {
  it("formats all supported live node states", () => {
    const cases: Array<{
      state: LiveNodeState;
      label: string;
      tone: "good" | "info" | "warn" | "bad" | "neutral";
      tooltip: string;
    }> = [
      {
        state: "client",
        label: "Client",
        tone: "info",
        tooltip: "Sends requests, but does not contribute VRAM.",
      },
      {
        state: "standby",
        label: "Standby",
        tone: "neutral",
        tooltip: "Connected, but not currently serving a model.",
      },
      {
        state: "loading",
        label: "Loading",
        tone: "warn",
        tooltip: "Initializing model work before it can serve requests.",
      },
      {
        state: "serving",
        label: "Serving",
        tone: "good",
        tooltip: "Actively serving a model.",
      },
    ];

    for (const testCase of cases) {
      expect(formatLiveNodeState(testCase.state)).toBe(testCase.label);
      expect(topologyStatusTone(testCase.state)).toBe(testCase.tone);
      expect(topologyStatusTooltip(testCase.state)).toBe(testCase.tooltip);
    }
  });

  it("handles invalid states gracefully without throwing", () => {
    expect(formatLiveNodeState(undefined as any)).toBe("Unknown");
    expect(formatLiveNodeState(null as any)).toBe("Unknown");
    
    expect(topologyStatusTone(undefined as any)).toBe("neutral");
    expect(topologyStatusTone(null as any)).toBe("neutral");
    
    expect(topologyStatusTooltip(undefined as any)).toBe("Node state unavailable");
    expect(topologyStatusTooltip(null as any)).toBe("Node state unavailable");
  });

  it("uses node_state as the local routable-model source of truth", () => {
    const baseStatus: StatusPayload = {
      node_id: "local-node",
      node_status: "Serving",
      node_state: "serving",
      token: "token",
      is_host: false,
      is_client: true,
      llama_ready: true,
      peers: [],
      model_name: "fallback-model",
      requested_models: [],
      available_models: [],
      serving_models: ["serving-model"],
      hosted_models: ["hosted-model"],
      my_vram_gb: 24,
      api_port: 3131,
      model_size_gb: 0,
      inflight_requests: 0,
      version: "test",
      latest_version: null,
      wakeable_nodes: [],
    };

    expect(localRoutableModels(baseStatus)).toEqual(["hosted-model"]);
    expect(localRoutableModels({ ...baseStatus, node_state: "client", is_client: false })).toEqual(
      [],
    );
  });
});

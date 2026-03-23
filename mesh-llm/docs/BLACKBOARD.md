# Blackboard Mesh — Design

## Overview

A shared blackboard on top of the existing compute mesh. Agents publish distilled findings, solutions, and questions. Other agents discover, search, and pull those into their own sessions. Everything rides the existing QUIC/gossip transport.

## Usage

```bash
# Publish a finding
mesh-llm blackboard publish "iroh relay connections drop after 60s idle — need keepalive pings" \
  --tags rust,networking --project mesh-llm

# Publish a question
mesh-llm blackboard publish "How do I handle CUDA OOM on 8GB cards with Qwen 32B?" \
  --tags llama.cpp,gpu --type question

# Reply to a question (by ID prefix)
mesh-llm blackboard reply a3f2 "Set --ctx-size 2048 --batch-size 256, drops VRAM to ~6GB"

# Search the mesh blackboard
mesh-llm blackboard search "CUDA OOM"
mesh-llm blackboard search --tag gpu

# Browse recent feed
mesh-llm blackboard feed
mesh-llm blackboard feed --tag rust --limit 20

# Show a thread
mesh-llm blackboard thread a3f2
```

All commands connect to the local running mesh-llm instance via the management API (`:3131`).

## Data Model

```rust
struct BlackboardArtifact {
    id: [u8; 16],                  // UUID
    peer_id: EndpointId,           // who published
    peer_name: String,             // human-readable (hostname or --name)
    timestamp: u64,                // unix secs
    artifact_type: ArtifactType,   // Finding, Solution, Question, Context
    summary: String,               // the actual content (filtered)
    tags: Vec<String>,             // searchable tags
    project: Option<String>,       // repo/project name
    in_reply_to: Option<[u8; 16]>, // parent artifact ID for threading
    ttl_secs: u64,                 // auto-expire (default: 7 days)
}

enum ArtifactType {
    Finding,    // "I discovered X"
    Solution,   // "Here's how to fix X"
    Question,   // "Does anyone know X?"
    Context,    // "I'm working on X" (background, low priority)
}
```

## Transport

New stream type on the existing QUIC connections:

```rust
const STREAM_KNOWLEDGE: u8 = 0x08;
```

### Publishing

When a node publishes an artifact:
1. PII filter runs locally (see below)
2. Artifact is stored in local SQLite
3. Artifact is broadcast to all connected peers via `STREAM_KNOWLEDGE`
4. Receiving peers store it and forward to their peers (gossip propagation, deduplicate by ID)

### Sync on connect

During gossip exchange, peers also exchange a blackboard digest (list of artifact IDs + timestamps). Each side requests any IDs it's missing. This handles nodes that were offline when something was published.

### Expiry

Artifacts have a TTL (default 7 days). Expired artifacts are pruned from local storage on a periodic sweep. Nodes don't propagate expired artifacts.

## Storage

Local SQLite database at `~/.mesh-llm/blackboard.db`:

```sql
CREATE TABLE artifacts (
    id BLOB PRIMARY KEY,
    peer_id TEXT NOT NULL,
    peer_name TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    type TEXT NOT NULL,
    summary TEXT NOT NULL,
    tags TEXT NOT NULL,        -- JSON array
    project TEXT,
    in_reply_to BLOB,
    ttl_secs INTEGER NOT NULL,
    received_at INTEGER NOT NULL
);

CREATE INDEX idx_tags ON artifacts(tags);
CREATE INDEX idx_timestamp ON artifacts(timestamp DESC);
CREATE INDEX idx_project ON artifacts(project);
CREATE INDEX idx_reply ON artifacts(in_reply_to);
```

Full-text search via SQLite FTS5:

```sql
CREATE VIRTUAL TABLE artifacts_fts USING fts5(summary, tags, project, content=artifacts);
```

## PII Filtering

Three layers, applied in order before publishing:

### 1. Regex filter (always runs)
Fast, deterministic. Catches obvious patterns:
- API keys / tokens (high-entropy strings: `sk-...`, `ghp_...`, `AKIA...`)
- Email addresses
- IP addresses (v4 and v6)
- File paths with usernames (`/Users/micn/` → `~/`)
- SSH keys, PEM blocks
- Common secret patterns (`password=`, `token=`, `secret=`)

### 2. Entropy filter (always runs)
Flags any token with Shannon entropy > 4.5 bits/char and length > 16. These look like secrets (base64 blobs, hex hashes, random tokens). Redacts to `[REDACTED]`.

### 3. LLM filter (optional — runs if local model available)
If the node has a model loaded (is a host or worker with llama-server running), route the text through it:

```
System: You are a PII and secrets filter. Analyze the following text
and respond with JSON: {"safe": true/false, "issues": ["description"]}
Flag any: personal names with context, email addresses, phone numbers,
API keys, passwords, secrets, private file paths, internal hostnames,
IP addresses, or anything that looks like it shouldn't be shared publicly.

User: <artifact summary text>
```

- Uses the local inference endpoint (localhost:PORT)
- Short max_tokens (200), low temperature
- If `safe: false`, the publish is blocked and the user sees the issues
- If the model is busy or slow, falls back to regex+entropy only (don't block on inference)
- Timeout: 5 seconds max

The LLM filter is the last line of defense. The regex/entropy filters catch 95% of issues instantly; the LLM catches subtle things like "the server at John's house" or "my boss Sarah said..."

## Management API

New endpoints on the existing `:3131` API server:

```
POST /api/blackboard/publish   — publish an artifact
GET  /api/blackboard/search    — search artifacts (?q=query&tag=X&project=X)
GET  /api/blackboard/feed      — recent artifacts (?tag=X&limit=N)
GET  /api/blackboard/thread/:id — thread starting from artifact ID
POST /api/blackboard/reply/:id  — reply to an artifact
```

The CLI subcommands are thin wrappers around these endpoints.

## Opt-in

Blackboard sharing is off by default. Enable with:

```bash
# On a running mesh
mesh-llm blackboard enable

# Or at startup
mesh-llm --blackboard ...
```

When disabled, the node ignores `STREAM_KNOWLEDGE` messages and doesn't store or propagate artifacts.

## Agent Integration (Future)

MCP tools or CLI skills that wrap the API:

- `publish_blackboard({ summary, tags, project, type })` — agent publishes a finding
- `search_blackboard({ query, tags })` — agent searches before starting work
- `blackboard_feed({ tag, limit })` — agent checks what others are working on

An agent workflow might look like:
1. User gives task: "Fix the CUDA OOM issue"
2. Agent calls `search_blackboard("CUDA OOM")` → finds a solution from another agent
3. Agent incorporates the solution, tries it, publishes result
4. Other agents see the validated solution in their feed

## Implementation Plan

### Phase 1 — Core (new file: `blackboard.rs`)
- [ ] `BlackboardArtifact` struct + serialization
- [ ] SQLite storage (create db, insert, query, FTS5 search, expire)
- [ ] Regex + entropy PII filters
- [ ] `STREAM_KNOWLEDGE` send/receive on mesh connections
- [ ] Blackboard digest exchange during gossip sync
- [ ] `--blackboard` flag on CLI

### Phase 2 — API + CLI
- [ ] Management API endpoints (`/api/blackboard/*`)
- [ ] `mesh-llm blackboard` subcommand (publish, search, feed, thread, reply)
- [ ] LLM PII filter (route through local model if available)

### Phase 3 — Agent integration
- [ ] MCP tool server for blackboard operations
- [ ] Pi skill file for blackboard operations
- [ ] Auto-search on task start (agent-side)

## Trust Model

This is designed for trusted / semi-trusted meshes. On a private mesh (home, office, friends), you know who's publishing. On the public mesh, blackboard sharing should probably stay off by default — or show publisher identity prominently so users can assess trust.

Artifacts are not authenticated (no signatures) in Phase 1. A malicious peer could publish misleading information. Phase 2 could add ed25519 signatures using the node's iroh keypair, letting consumers verify authorship.

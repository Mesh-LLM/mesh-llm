use std::{
    collections::{BTreeMap, BTreeSet, VecDeque, hash_map::DefaultHasher},
    fs,
    hash::{Hash, Hasher},
    io::{self, BufRead, BufReader, IsTerminal, Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use mesh_client::models::gguf::{GgufCompactMeta, scan_gguf_compact_meta};
use openai_frontend::{ReasoningConfig, normalize_reasoning_template_options};
use rustyline::{DefaultEditor, error::ReadlineError};
use serde_json::Value;
use skippy_protocol::binary::{
    LLAMA_TOKEN_NULL, READY_MAGIC, StageReply, StageReplyStats, StageStateHeader, StageWireMessage,
    WireActivationDType, WireMessageKind, WireReplyKind, recv_reply, write_stage_message,
};
use skippy_protocol::{
    FlashAttentionType as StageFlashAttentionType, LoadMode, PeerConfig, StageConfig,
    StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
};
use skippy_runtime::{
    ChatTemplateMessage, ChatTemplateOptions, GGML_TYPE_F16, ModelInfo, RuntimeConfig,
    RuntimeLoadMode, StageModel, StageSession,
    package::{PackageStageRequest, inspect_layer_package, materialize_layer_package},
    restore_native_logs, suppress_native_logs,
};
use skippy_topology::{
    BoundaryDecision, NodeSpec, PlannerPolicy, TopologyPlanRequest, WireValidation,
    dense_attention_layers, infer_family_capability, plan_contiguous_with_splits,
};

const DEFAULT_MESH_CTX_SIZE: u32 = 4096;
const DEFAULT_MESH_PROMPT_MAX_NEW_TOKENS: usize = 0;
const PROMPT_EXACT_PREFIX_RESTORE_MIN_TOKENS: usize = 512;

include!("args.rs");
include!("command.rs");
include!("launch.rs");
include!("interrupt.rs");
include!("binary_repl.rs");
include!("logs.rs");
include!("prompt_format.rs");
include!("generation.rs");
include!("live_session.rs");
include!("speculative.rs");
include!("wire_messages.rs");
include!("draft.rs");
include!("history.rs");
include!("stage_config.rs");
include!("remote_sync.rs");
include!("formatting.rs");
include!("topology.rs");
include!("tests.rs");

use std::{
    collections::BTreeMap,
    fs,
    io::{self, BufRead, BufReader, IsTerminal, Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use openai_frontend::{ReasoningConfig, normalize_reasoning_template_options};
use rustyline::{DefaultEditor, error::ReadlineError};
use skippy_protocol::binary::{
    LLAMA_TOKEN_NULL, READY_MAGIC, StageReply, StageReplyStats, StageStateHeader, StageWireMessage,
    WireMessageKind, WireReplyKind, recv_reply, write_stage_message,
};
use skippy_runtime::{
    ChatTemplateMessage, ChatTemplateOptions, GGML_TYPE_F16, ModelInfo, MtpSource, RuntimeConfig,
    RuntimeLoadMode, StageModel, StageSession,
    package::{PackageStageRequest, inspect_layer_package, materialize_layer_package},
    plan_gguf_stage_runtime_plan_for_range, restore_native_logs, suppress_native_logs,
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

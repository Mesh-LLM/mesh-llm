//! Run or probe an MLX partial-layer engine over Skippy's binary stage wire.

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod real {
    use std::{
        io::Write,
        net::{SocketAddr, TcpStream},
        path::PathBuf,
        sync::Arc,
    };

    use anyhow::{Context, Result, ensure};
    use clap::{Parser, Subcommand, ValueEnum};
    use model_hf::safetensors_stage::{SafetensorsStageMaterializer, SafetensorsStageRequest};
    use skippy_engine_mlx::{
        MlxBoundaryBenchConfig, MlxComputeDtype, MlxDerivationControl, MlxDerivedStageCacheConfig,
        MlxDerivedStageConfig, MlxStageEngine, MlxStageEngineConfig, MlxTcpBoundaryBenchConfig,
        MlxTcpBoundarySinkConfig, MlxWeightQuantization, benchmark_mlx_boundary,
        benchmark_mlx_tcp_boundary, derive_quantized_stage, derive_quantized_stage_cached,
        mlx_derived_stage_cache_root, serve_mlx_tcp_boundary_sink,
        validate_nemotron_h_binary_wire_tokens, validate_nemotron_h_moe_stage,
        validate_nemotron_h_stage_engine,
    };
    use skippy_protocol::binary::{
        StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
        recv_ready, recv_reply, write_stage_message,
    };
    use skippy_server::engine_transport::{EngineStageServerOptions, serve_stage_engine};

    #[derive(Debug, Parser)]
    #[command(about = "Serve and prove partial SafeTensors MLX stages")]
    struct Cli {
        #[command(subcommand)]
        command: Command,
    }

    #[derive(Debug, Subcommand)]
    enum Command {
        /// Load one partial SafeTensors artifact and serve its layer range.
        Serve {
            #[arg(long)]
            model: PathBuf,
            #[arg(long, default_value = "mlx-stage-model")]
            model_id: String,
            #[arg(long)]
            stage_index: u32,
            #[arg(long)]
            layer_start: u32,
            #[arg(long)]
            layer_end: u32,
            #[arg(long)]
            bind: SocketAddr,
            #[arg(long)]
            downstream: Option<SocketAddr>,
            #[arg(long, value_enum, default_value_t = WireDtype::F16)]
            wire_dtype: WireDtype,
            #[arg(long, value_enum, default_value_t = ComputeDtype::Bf16)]
            compute_dtype: ComputeDtype,
            #[arg(long, value_enum)]
            weight_quantization: Option<WeightQuantization>,
        },
        /// Download exact stage tensors and write bounded MLX-quantized shards.
        Derive {
            #[arg(long)]
            repo: String,
            /// Immutable 40-character Hugging Face commit SHA.
            #[arg(long)]
            revision: String,
            #[arg(long)]
            layer_start: u32,
            #[arg(long)]
            layer_end: u32,
            #[arg(long = "include-prefix")]
            include_prefixes: Vec<String>,
            #[arg(long)]
            output: PathBuf,
            #[arg(long, value_enum, default_value_t = WeightQuantization::Affine4)]
            weight_quantization: WeightQuantization,
            /// Soft output shard target; one converted tensor bundle may exceed it.
            #[arg(long, default_value_t = 256)]
            shard_size_mib: usize,
        },
        /// Reuse or build an identity-bound quantized stage cache entry.
        DeriveCached {
            #[arg(long)]
            repo: String,
            /// Immutable 40-character Hugging Face commit SHA.
            #[arg(long)]
            revision: String,
            #[arg(long)]
            layer_start: u32,
            #[arg(long)]
            layer_end: u32,
            #[arg(long = "include-prefix")]
            include_prefixes: Vec<String>,
            /// Defaults to the mesh-llm cache directory.
            #[arg(long)]
            cache_root: Option<PathBuf>,
            #[arg(long, value_enum, default_value_t = WeightQuantization::Affine4)]
            weight_quantization: WeightQuantization,
            /// Soft output shard target; one converted tensor bundle may exceed it.
            #[arg(long, default_value_t = 256)]
            shard_size_mib: usize,
        },
        /// Strict-load and execute one derived Nemotron-H MoE layer.
        ValidateNemotronH {
            #[arg(long)]
            model: PathBuf,
            #[arg(long)]
            layer: usize,
        },
        /// Prove StageEngine output numerically matches direct execution.
        ValidateNemotronHStage {
            #[arg(long)]
            model: PathBuf,
            #[arg(long)]
            layer: usize,
        },
        /// Prove a Nemotron-H layer over the real binary stage wire.
        ValidateNemotronHWire {
            #[arg(long)]
            model: PathBuf,
            #[arg(long)]
            layer: usize,
            #[arg(long, default_value_t = 1)]
            tokens: usize,
            #[arg(long, value_enum, default_value_t = WireDtype::F16)]
            wire_dtype: WireDtype,
        },
        /// Measure the MLX completion, host-copy, and activation-codec boundary.
        BenchBoundary {
            #[arg(long)]
            width: usize,
            #[arg(long)]
            tokens: usize,
            #[arg(long, value_enum)]
            wire_dtype: WireDtype,
            #[arg(long, default_value_t = 3)]
            warmup_iterations: usize,
            #[arg(long, default_value_t = 20)]
            measured_iterations: usize,
            #[arg(long)]
            metrics_http: String,
            #[arg(long)]
            metrics_otlp_grpc: String,
            #[arg(long)]
            metrics_run_id: Option<String>,
            #[arg(long)]
            metrics_report: PathBuf,
            #[arg(long)]
            output: Option<PathBuf>,
        },
        /// Measure one activation through local or remote production Skippy TCP.
        BenchTcpBoundary {
            #[arg(long)]
            width: usize,
            #[arg(long)]
            tokens: usize,
            #[arg(long, value_enum)]
            wire_dtype: WireDtype,
            #[arg(long, default_value_t = 3)]
            warmup_iterations: usize,
            #[arg(long, default_value_t = 20)]
            measured_iterations: usize,
            #[arg(long)]
            metrics_http: String,
            #[arg(long)]
            metrics_otlp_grpc: String,
            #[arg(long)]
            metrics_run_id: Option<String>,
            #[arg(long)]
            metrics_report: PathBuf,
            /// Connect to a separately running validating sink instead of loopback.
            #[arg(long)]
            connect: Option<SocketAddr>,
            #[arg(long)]
            output: Option<PathBuf>,
        },
        /// Run a trusted-network-only validating sink for the TCP boundary benchmark.
        ServeTcpBoundarySink {
            #[arg(long)]
            bind: SocketAddr,
            #[arg(long)]
            width: usize,
            #[arg(long)]
            tokens: usize,
            #[arg(long, value_enum)]
            wire_dtype: WireDtype,
        },
        /// Drive a stage chain and assert its greedy token sequence.
        Prove {
            #[arg(long)]
            connect: SocketAddr,
            #[arg(long, default_value = "1,1531,314,260,3575,28")]
            tokens: String,
            #[arg(long, default_value = "260,2240,314,253,1379,282,25801,28")]
            expected: String,
            #[arg(long, value_enum, default_value_t = WireDtype::F16)]
            wire_dtype: WireDtype,
        },
    }

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum WireDtype {
        F16,
        F32,
    }

    impl From<WireDtype> for WireActivationDType {
        fn from(value: WireDtype) -> Self {
            match value {
                WireDtype::F16 => Self::F16,
                WireDtype::F32 => Self::F32,
            }
        }
    }

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum ComputeDtype {
        F16,
        Bf16,
        F32,
    }

    impl From<ComputeDtype> for MlxComputeDtype {
        fn from(value: ComputeDtype) -> Self {
            match value {
                ComputeDtype::F16 => Self::F16,
                ComputeDtype::Bf16 => Self::Bf16,
                ComputeDtype::F32 => Self::F32,
            }
        }
    }

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum WeightQuantization {
        Affine4,
        Affine8,
        Mxfp4,
    }

    impl From<WeightQuantization> for MlxWeightQuantization {
        fn from(value: WeightQuantization) -> Self {
            match value {
                WeightQuantization::Affine4 => Self::Affine {
                    group_size: 64,
                    bits: 4,
                },
                WeightQuantization::Affine8 => Self::Affine {
                    group_size: 64,
                    bits: 8,
                },
                WeightQuantization::Mxfp4 => Self::MxFp4,
            }
        }
    }

    pub fn main() -> Result<()> {
        match Cli::parse().command {
            Command::Serve {
                model,
                model_id,
                stage_index,
                layer_start,
                layer_end,
                bind,
                downstream,
                wire_dtype,
                compute_dtype,
                weight_quantization,
            } => serve(
                MlxStageEngineConfig {
                    model_dir: model,
                    model_id,
                    stage_index,
                    layer_start,
                    layer_end,
                    compute_dtype: compute_dtype.into(),
                    weight_quantization: weight_quantization.map(Into::into),
                    ctx_size: None,
                },
                EngineStageServerOptions {
                    bind_addr: bind,
                    downstream_addr: downstream,
                    wire_dtype: wire_dtype.into(),
                },
            ),
            Command::Derive {
                repo,
                revision,
                layer_start,
                layer_end,
                include_prefixes,
                output,
                weight_quantization,
                shard_size_mib,
            } => derive(
                SafetensorsStageRequest {
                    repo,
                    revision,
                    layer_start,
                    layer_end,
                    include_prefixes,
                },
                output,
                weight_quantization.into(),
                shard_size_mib,
            ),
            Command::DeriveCached {
                repo,
                revision,
                layer_start,
                layer_end,
                include_prefixes,
                cache_root,
                weight_quantization,
                shard_size_mib,
            } => derive_cached(
                SafetensorsStageRequest {
                    repo,
                    revision,
                    layer_start,
                    layer_end,
                    include_prefixes,
                },
                cache_root,
                weight_quantization.into(),
                shard_size_mib,
            ),
            Command::ValidateNemotronH { model, layer } => {
                let report = validate_nemotron_h_moe_stage(model, layer)?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                Ok(())
            }
            Command::ValidateNemotronHStage { model, layer } => {
                let report = validate_nemotron_h_stage_engine(model, layer)?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                Ok(())
            }
            Command::ValidateNemotronHWire {
                model,
                layer,
                tokens,
                wire_dtype,
            } => {
                let report = validate_nemotron_h_binary_wire_tokens(
                    model,
                    layer,
                    wire_dtype.into(),
                    tokens,
                )?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                Ok(())
            }
            Command::BenchBoundary {
                width,
                tokens,
                wire_dtype,
                warmup_iterations,
                measured_iterations,
                metrics_http,
                metrics_otlp_grpc,
                metrics_run_id,
                metrics_report,
                output,
            } => {
                let metrics_run_id = match metrics_run_id {
                    Some(metrics_run_id) => metrics_run_id,
                    None => default_boundary_run_id()?,
                };
                let report = benchmark_mlx_boundary(&MlxBoundaryBenchConfig {
                    width,
                    token_count: tokens,
                    wire_dtype: wire_dtype.into(),
                    warmup_iterations,
                    measured_iterations,
                    metrics_http,
                    metrics_otlp_grpc,
                    metrics_run_id,
                    metrics_report_path: metrics_report,
                })?;
                let json = serde_json::to_vec_pretty(&report)?;
                if let Some(output) = output {
                    if let Some(parent) = output
                        .parent()
                        .filter(|parent| !parent.as_os_str().is_empty())
                    {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(output, &json)?;
                }
                println!("{}", String::from_utf8(json)?);
                Ok(())
            }
            Command::BenchTcpBoundary {
                width,
                tokens,
                wire_dtype,
                warmup_iterations,
                measured_iterations,
                metrics_http,
                metrics_otlp_grpc,
                metrics_run_id,
                metrics_report,
                connect,
                output,
            } => {
                let metrics_run_id = match metrics_run_id {
                    Some(metrics_run_id) => metrics_run_id,
                    None => default_boundary_run_id()?,
                };
                let report = benchmark_mlx_tcp_boundary(&MlxTcpBoundaryBenchConfig {
                    width,
                    token_count: tokens,
                    wire_dtype: wire_dtype.into(),
                    warmup_iterations,
                    measured_iterations,
                    metrics_http,
                    metrics_otlp_grpc,
                    metrics_run_id,
                    metrics_report_path: metrics_report,
                    connect_addr: connect,
                })?;
                let json = serde_json::to_vec_pretty(&report)?;
                if let Some(output) = output {
                    if let Some(parent) = output
                        .parent()
                        .filter(|parent| !parent.as_os_str().is_empty())
                    {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(output, &json)?;
                }
                println!("{}", String::from_utf8(json)?);
                Ok(())
            }
            Command::ServeTcpBoundarySink {
                bind,
                width,
                tokens,
                wire_dtype,
            } => serve_mlx_tcp_boundary_sink(&MlxTcpBoundarySinkConfig {
                bind_addr: bind,
                width,
                token_count: tokens,
                wire_dtype: wire_dtype.into(),
            }),
            Command::Prove {
                connect,
                tokens,
                expected,
                wire_dtype,
            } => prove(connect, &tokens, &expected, wire_dtype.into()),
        }
    }

    fn serve(config: MlxStageEngineConfig, options: EngineStageServerOptions) -> Result<()> {
        let engine = Arc::new(MlxStageEngine::spawn(config)?);
        serve_stage_engine(engine, options)
    }

    fn derive(
        source: SafetensorsStageRequest,
        output_dir: PathBuf,
        quantization: MlxWeightQuantization,
        shard_size_mib: usize,
    ) -> Result<()> {
        let shard_size_bytes = shard_size_mib
            .checked_mul(1024 * 1024)
            .context("derived shard size overflow")?;
        let materializer = SafetensorsStageMaterializer::from_environment()?;
        let report = derive_quantized_stage(
            &materializer,
            &MlxDerivedStageConfig {
                source,
                output_dir,
                quantization,
                control: MlxDerivationControl::default(),
                shard_size_bytes,
            },
        )?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    fn derive_cached(
        source: SafetensorsStageRequest,
        cache_root: Option<PathBuf>,
        quantization: MlxWeightQuantization,
        shard_size_mib: usize,
    ) -> Result<()> {
        let shard_size_bytes = shard_size_mib
            .checked_mul(1024 * 1024)
            .context("derived shard size overflow")?;
        let cache_root = cache_root.unwrap_or_else(mlx_derived_stage_cache_root);
        let materializer = SafetensorsStageMaterializer::from_environment()?;
        let result = derive_quantized_stage_cached(
            &materializer,
            &MlxDerivedStageCacheConfig {
                source,
                cache_root,
                quantization,
                control: MlxDerivationControl::default(),
                shard_size_bytes,
            },
        )?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        Ok(())
    }

    fn prove(
        connect: SocketAddr,
        tokens: &str,
        expected: &str,
        wire_dtype: WireActivationDType,
    ) -> Result<()> {
        let prompt = parse_ids(tokens)?;
        let expected = parse_ids(expected)?;
        ensure!(
            !expected.is_empty(),
            "expected token sequence must not be empty"
        );
        let mut stream = TcpStream::connect(connect)
            .with_context(|| format!("connect first MLX stage at {connect}"))?;
        stream.set_nodelay(true).ok();
        recv_ready(&mut stream).context("first MLX stage did not become ready")?;

        let session_id = 1;
        let request_id = 1;
        let prefill = execution_message(
            WireMessageKind::PrefillFinalEmbd,
            &prompt,
            *prompt.last().context("prompt must not be empty")?,
            request_id,
            session_id,
            wire_dtype,
            0,
        );
        let mut generated = Vec::with_capacity(expected.len());
        generated.push(send_predicted(&mut stream, &prefill, wire_dtype)?);

        while generated.len() < expected.len() {
            let current = *generated.last().expect("generated has first token");
            let decode = execution_message(
                WireMessageKind::DecodeEmbd,
                &[],
                current,
                request_id,
                session_id,
                wire_dtype,
                i32::try_from(generated.len())?,
            );
            generated.push(send_predicted(&mut stream, &decode, wire_dtype)?);
        }

        let stop = StageWireMessage::stop_with_identity(wire_dtype, request_id, session_id);
        write_stage_message(&mut stream, &stop, wire_dtype)?;
        stream.flush().ok();
        let reply = recv_reply(&mut stream)?;
        ensure!(reply.kind == WireReplyKind::Ack, "stop did not return ACK");
        ensure!(
            generated == expected,
            "two-process stage tokens diverged: expected={expected:?} actual={generated:?}"
        );
        println!("PASS: two MLX stage processes matched the reference greedy tokens");
        println!("wire_dtype={wire_dtype:?}");
        println!("generated_tokens={generated:?}");
        Ok(())
    }

    fn execution_message(
        kind: WireMessageKind,
        tokens: &[i32],
        current_token: i32,
        request_id: u64,
        session_id: u64,
        wire_dtype: WireActivationDType,
        decode_step: i32,
    ) -> StageWireMessage {
        let mut state = StageStateHeader::new(kind, wire_dtype);
        state.current_token = current_token;
        state.prompt_token_count = i32::try_from(tokens.len()).unwrap_or_default();
        state.decode_step = decode_step;
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count: if kind == WireMessageKind::DecodeEmbd {
                1
            } else {
                i32::try_from(tokens.len()).unwrap_or_default()
            },
            state,
            request_id,
            session_id,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: tokens.to_vec(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn send_predicted(
        stream: &mut TcpStream,
        message: &StageWireMessage,
        wire_dtype: WireActivationDType,
    ) -> Result<i32> {
        write_stage_message(&mut *stream, message, wire_dtype)?;
        stream.flush().ok();
        let reply = recv_reply(&mut *stream)?;
        ensure!(
            matches!(
                reply.kind,
                WireReplyKind::PredictedToken | WireReplyKind::PredictedTokens
            ),
            "stage chain did not return a predicted token"
        );
        Ok(reply.predicted)
    }

    fn parse_ids(value: &str) -> Result<Vec<i32>> {
        value
            .split(',')
            .map(|token| token.trim().parse().context("parse token ID"))
            .collect()
    }

    fn default_boundary_run_id() -> Result<String> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("system clock is before Unix epoch")?
            .as_nanos();
        Ok(format!("mlx-boundary-{nanos}"))
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    real::main()
}

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
fn main() {
    eprintln!("mlx-stage requires macOS and `--features mlx`");
    std::process::exit(1);
}

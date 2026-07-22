use std::{net::SocketAddr, path::PathBuf};

use crate::telemetry::TelemetryLevel;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(about = "Llama staged-runtime server")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Serve(ServeArgs),
    ServeBinary(ServeBinaryArgs),
    #[command(name = "serve-openai")]
    ServeOpenAi(ServeOpenAiArgs),
    ExampleConfig,
}

#[derive(Parser)]
pub struct ServeArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub topology: Option<PathBuf>,
    #[arg(long)]
    pub bind_addr: Option<SocketAddr>,
    #[arg(long)]
    pub metrics_otlp_grpc: Option<String>,
    #[arg(long, default_value_t = 1024)]
    pub telemetry_queue_capacity: usize,
    #[arg(long, value_enum, default_value_t = TelemetryLevel::Summary)]
    pub telemetry_level: TelemetryLevel,
}

#[derive(Parser)]
pub struct ServeBinaryArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub topology: Option<PathBuf>,
    #[arg(long)]
    pub bind_addr: Option<SocketAddr>,
    #[arg(long)]
    pub activation_width: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long)]
    pub metrics_otlp_grpc: Option<String>,
    #[arg(long, default_value_t = 1024)]
    pub telemetry_queue_capacity: usize,
    #[arg(long, value_enum, default_value_t = TelemetryLevel::Summary)]
    pub telemetry_level: TelemetryLevel,
    #[arg(long, default_value_t = 4)]
    pub max_inflight: usize,
    #[arg(long)]
    pub reply_credit_limit: Option<usize>,
    #[arg(
        long,
        help = "Forward eligible non-final prefill activation frames on a bounded background writer. Enabled by default."
    )]
    pub async_prefill_forward: bool,
    #[arg(
        long,
        help = "Disable async forwarding for eligible non-final prefill activation frames."
    )]
    pub no_async_prefill_forward: bool,
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Artificial downstream write delay in milliseconds per binary stage message."
    )]
    pub downstream_wire_delay_ms: f64,
    #[arg(
        long,
        help = "Artificial downstream activation bandwidth cap in megabits per second."
    )]
    pub downstream_wire_mbps: Option<f64>,
    #[arg(long, default_value_t = 60)]
    pub downstream_connect_timeout_secs: u64,
    #[arg(
        long,
        help = "Also serve the OpenAI-compatible HTTP surface from this stage process. Intended for stage 0."
    )]
    pub openai_bind_addr: Option<SocketAddr>,
    #[arg(
        long,
        help = "Served OpenAI model id. Defaults to the stage config model_id."
    )]
    pub openai_model_id: Option<String>,
    #[arg(long, default_value_t = 16)]
    pub openai_default_max_tokens: u32,
    #[arg(
        long,
        default_value_t = 1,
        help = "Maximum number of concurrent OpenAI chat generation requests hosted by this stage."
    )]
    pub openai_generation_concurrency: usize,
    #[arg(long, default_value_t = 256)]
    pub openai_prefill_chunk_size: usize,
    #[arg(
        long,
        default_value = "adaptive-ramp",
        help = "OpenAI prefill chunk policy: fixed, schedule, or adaptive-ramp. Passing --openai-prefill-chunk-schedule keeps legacy schedule behavior."
    )]
    pub openai_prefill_chunk_policy: String,
    #[arg(
        long,
        help = "Comma-separated OpenAI prefill chunk schedule. Example: 128,256,512 sends the first chunk at 128 tokens, second at 256, and repeats 512 after that."
    )]
    pub openai_prefill_chunk_schedule: Option<String>,
    #[arg(long, default_value_t = 128)]
    pub openai_prefill_adaptive_start: usize,
    #[arg(long, default_value_t = 128)]
    pub openai_prefill_adaptive_step: usize,
    #[arg(long, default_value_t = 384)]
    pub openai_prefill_adaptive_max: usize,
    #[arg(
        long,
        help = "Draft GGUF to use for speculative decoding in the embedded stage-0 OpenAI surface."
    )]
    pub openai_draft_model_path: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    pub openai_speculative_window: usize,
    #[arg(long)]
    pub openai_adaptive_speculative_window: bool,
    #[arg(
        long,
        help = "Override n_gpu_layers for the embedded OpenAI draft model. Defaults to the stage config n_gpu_layers."
    )]
    pub openai_draft_n_gpu_layers: Option<i32>,
    #[arg(
        long,
        help = "JSON file containing the complete resolved speculative decode plan."
    )]
    pub openai_speculative_config: Option<PathBuf>,
}

#[derive(Parser)]
pub struct ServeOpenAiArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub topology: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:9337")]
    pub bind_addr: SocketAddr,
    #[arg(
        long,
        help = "Served model id to advertise and accept, for example org/repo:Q4_K_M. Defaults to config model_id."
    )]
    pub model_id: Option<String>,
    #[arg(
        long,
        help = "JSON file containing a complete resolved speculative decode plan."
    )]
    pub speculative_config: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    pub default_max_tokens: u32,
    #[arg(
        long,
        default_value_t = 1,
        help = "Maximum number of concurrent chat generation requests."
    )]
    pub generation_concurrency: usize,
    #[arg(
        long,
        help = "Deprecated and unsupported. Direct prediction return requires embedded stage-0 OpenAI serving via serve-binary --openai-bind-addr."
    )]
    pub first_stage_addr: Option<String>,
    #[arg(long, default_value_t = 256)]
    pub prefill_chunk_size: usize,
    #[arg(
        long,
        default_value = "adaptive-ramp",
        help = "Prefill chunk policy for split OpenAI serving: fixed, schedule, or adaptive-ramp. Passing --prefill-chunk-schedule keeps legacy schedule behavior."
    )]
    pub prefill_chunk_policy: String,
    #[arg(
        long,
        help = "Comma-separated prefill chunk schedule for split OpenAI serving. Example: 128,256,512 sends the first chunk at 128 tokens, second at 256, and repeats 512 after that."
    )]
    pub prefill_chunk_schedule: Option<String>,
    #[arg(long, default_value_t = 128)]
    pub prefill_adaptive_start: usize,
    #[arg(long, default_value_t = 128)]
    pub prefill_adaptive_step: usize,
    #[arg(long, default_value_t = 384)]
    pub prefill_adaptive_max: usize,
    #[arg(long, default_value = "f32")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
    #[arg(long)]
    pub metrics_otlp_grpc: Option<String>,
    #[arg(long, default_value_t = 1024)]
    pub telemetry_queue_capacity: usize,
    #[arg(long, value_enum, default_value_t = TelemetryLevel::Summary)]
    pub telemetry_level: TelemetryLevel,
    #[arg(
        long = "openai-guardrails",
        value_enum,
        default_value_t = OpenAiGuardrailsCliMode::Metrics,
        help = "OpenAI compatibility guardrail mode for standalone serving: disabled, metrics, or enforce."
    )]
    pub openai_guardrails: OpenAiGuardrailsCliMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OpenAiGuardrailsCliMode {
    Disabled,
    Metrics,
    Enforce,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_prefill_policy_defaults_to_adaptive_ramp() {
        let cli = Cli::try_parse_from([
            "skippy-server",
            "serve-binary",
            "--config",
            "stage.json",
            "--activation-width",
            "2048",
        ])
        .unwrap();

        let Command::ServeBinary(args) = cli.command else {
            panic!("expected serve-binary command");
        };
        assert_eq!(args.openai_prefill_chunk_policy, "adaptive-ramp");
        assert_eq!(args.openai_prefill_adaptive_start, 128);
        assert_eq!(args.openai_prefill_adaptive_step, 128);
        assert_eq!(args.openai_prefill_adaptive_max, 384);

        let cli = Cli::try_parse_from(["skippy-server", "serve-openai", "--config", "stage.json"])
            .unwrap();

        let Command::ServeOpenAi(args) = cli.command else {
            panic!("expected serve-openai command");
        };
        assert_eq!(args.prefill_chunk_policy, "adaptive-ramp");
        assert_eq!(args.prefill_adaptive_start, 128);
        assert_eq!(args.prefill_adaptive_step, 128);
        assert_eq!(args.prefill_adaptive_max, 384);
        assert_eq!(args.openai_guardrails, OpenAiGuardrailsCliMode::Metrics);
    }

    #[test]
    fn serve_openai_accepts_explicit_guardrail_mode() {
        let cli = Cli::try_parse_from([
            "skippy-server",
            "serve-openai",
            "--config",
            "stage.json",
            "--openai-guardrails",
            "enforce",
        ])
        .unwrap();

        let Command::ServeOpenAi(args) = cli.command else {
            panic!("expected serve-openai command");
        };
        assert_eq!(args.openai_guardrails, OpenAiGuardrailsCliMode::Enforce);
    }

    #[test]
    fn standalone_commands_accept_resolved_speculative_config_files() {
        let cli = Cli::try_parse_from([
            "skippy-server",
            "serve-binary",
            "--config",
            "stage.json",
            "--activation-width",
            "2048",
            "--openai-speculative-config",
            "decode-plan.json",
        ])
        .unwrap();
        let Command::ServeBinary(args) = cli.command else {
            panic!("expected serve-binary command");
        };
        assert_eq!(
            args.openai_speculative_config,
            Some(PathBuf::from("decode-plan.json"))
        );

        let cli = Cli::try_parse_from([
            "skippy-server",
            "serve-openai",
            "--config",
            "stage.json",
            "--speculative-config",
            "decode-plan.json",
        ])
        .unwrap();
        let Command::ServeOpenAi(args) = cli.command else {
            panic!("expected serve-openai command");
        };
        assert_eq!(
            args.speculative_config,
            Some(PathBuf::from("decode-plan.json"))
        );
    }
}

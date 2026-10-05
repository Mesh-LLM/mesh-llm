//! Complete standalone serving settings: CLI > environment > TOML > defaults.

mod catalog;
mod report;
mod resolution;
#[cfg(test)]
mod tests;

use std::{collections::BTreeMap, ffi::OsString, path::Path};

use anyhow::{Context, Result};
use clap::{Arg, ArgAction, ArgMatches, CommandFactory, FromArgMatches, parser::ValueSource};
use serde_json::Value;

use crate::cli::{Cli, Command};
use catalog::{Kind, OPTIONS};

#[derive(Clone, Debug, Default)]
pub struct ServeSettings {
    pub values: BTreeMap<String, Value>,
    pub sources: BTreeMap<String, String>,
}

pub fn command() -> clap::Command {
    let command = decorate(Cli::command());
    let global = command
        .get_arguments()
        .filter(|argument| argument.is_global_set())
        .cloned()
        .collect::<Vec<_>>();
    command.mut_subcommand("serve", |mut command| {
        command = decorate(command.args(global));
        for (index, spec) in OPTIONS.iter().enumerate() {
            let mut arg = Arg::new(spec.name)
                .long(spec.name)
                .help(spec.help)
                .help_heading(heading(spec.section))
                .display_order(section_order(spec.section) + index)
                .value_name("VALUE");
            if matches!(spec.kind, Kind::Bool) {
                arg = arg
                    .action(ArgAction::Set)
                    .num_args(0..=1)
                    .require_equals(true)
                    .default_missing_value("true")
                    .value_parser(clap::value_parser!(bool));
            } else {
                arg = arg.value_parser(clap::builder::StringValueParser::new());
                if matches!(spec.kind, Kind::Integer | Kind::Number) {
                    arg = arg.allow_hyphen_values(true);
                }
            }
            command = command.arg(arg);
        }
        order_help_groups(command)
    })
}

fn order_help_groups(command: clap::Command) -> clap::Command {
    // Clap discovers headings in insertion order, independently of option order.
    let mut arguments = command.get_arguments().cloned().collect::<Vec<_>>();
    arguments.sort_by_key(Arg::get_display_order);
    let mut arguments = arguments.into_iter();
    command.mut_args(|_| arguments.next().expect("one replacement per argument"))
}

fn decorate(mut command: clap::Command) -> clap::Command {
    let arguments = command.get_arguments().cloned().collect::<Vec<_>>();
    for (index, arg) in arguments.into_iter().enumerate() {
        let Some(name) = arg.get_long() else { continue };
        let heading = heading(section(name));
        let order = section_order(section(name)) + index;
        let description = description(name);
        let signed = matches!(name, "draft-n-gpu-layers" | "n-gpu-layers");
        command = command.mut_arg(arg.get_id(), |arg| {
            let mut arg = arg.help_heading(heading).display_order(order);
            if let Some(description) = description {
                arg = arg.help(description);
            }
            if matches!(arg.get_action(), ArgAction::SetTrue) {
                arg.action(ArgAction::Set)
                    .num_args(0..=1)
                    .require_equals(true)
                    .default_value("false")
                    .default_missing_value("true")
                    .value_parser(clap::value_parser!(bool))
            } else if signed {
                arg.allow_hyphen_values(true)
            } else {
                arg
            }
        });
    }
    command
}

pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Cli> {
    parse_with_environment(args, |name| std::env::var(name).ok())
}

fn parse_with_environment(
    args: impl IntoIterator<Item = OsString>,
    environment: impl Fn(&str) -> Option<String>,
) -> Result<Cli> {
    let command = command();
    let matches = command.clone().try_get_matches_from(args)?;
    let Some(("serve", serve)) = matches.subcommand() else {
        return Ok(Cli::from_arg_matches(&matches)?);
    };
    let mut settings = ServeSettings::default();
    if let Some(path) = serve.get_one::<std::path::PathBuf>("settings") {
        settings.load_file(path, &command)?;
    }
    settings.load_environment(&command, environment)?;
    settings.load_cli(&matches, &command)?;
    let mut argv = vec![OsString::from("skippy"), OsString::from("serve")];
    settings.append_arguments(&mut argv)?;
    let resolved = command.try_get_matches_from(argv)?;
    let mut cli = Cli::from_arg_matches(&resolved)?;
    if let Command::Serve(args) = &mut cli.command {
        args.public.settings = settings;
    }
    Ok(cli)
}

impl ServeSettings {
    fn insert(&mut self, name: &str, value: Value, source: String) -> Result<()> {
        let spec = OPTIONS.iter().find(|spec| spec.name == name);
        let value = spec
            .map_or(Ok(value.clone()), |spec| {
                if source.starts_with("file:") && matches!(spec.kind, Kind::Json | Kind::Content) {
                    Ok(value)
                } else {
                    parse_value(spec.kind, value)
                }
            })
            .with_context(|| format!("invalid setting {name} from {source}"))?;
        self.values.insert(name.to_owned(), value);
        self.sources.insert(name.to_owned(), source);
        Ok(())
    }

    fn load_file(&mut self, path: &Path, command: &clap::Command) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("read serving settings {}", path.display()))?;
        let value: toml::Value = toml::from_str(&content)
            .with_context(|| format!("parse serving settings {}", path.display()))?;
        let object = serde_json::to_value(value)?;
        let object = object
            .as_object()
            .context("serving settings must be a table")?;
        let known = known_arguments(command);
        for (group, entries) in object {
            if group == "version" {
                anyhow::ensure!(entries.as_u64() == Some(1), "settings version must be 1");
                continue;
            }
            anyhow::ensure!(
                [
                    "model",
                    "execution",
                    "kv",
                    "cache",
                    "scheduling",
                    "sampling",
                    "chat",
                    "speculative",
                    "media",
                    "api",
                    "distributed",
                    "runtime",
                    "diagnostics",
                    "network"
                ]
                .contains(&group.as_str()),
                "unknown settings section [{group}]"
            );
            let entries = entries
                .as_object()
                .with_context(|| format!("{group} must be a table"))?;
            for (key, value) in entries {
                let name = key.replace('_', "-");
                anyhow::ensure!(
                    known.contains_key(&name),
                    "unknown serving setting {group}.{key}"
                );
                anyhow::ensure!(
                    section(&name) == group,
                    "{key} belongs in [{}]",
                    section(&name)
                );
                anyhow::ensure!(
                    !matches!(
                        name.as_str(),
                        "settings" | "print-effective-config" | "help"
                    ),
                    "{name} is a CLI-only control"
                );
                let value = file_value(
                    &name,
                    value.clone(),
                    path.parent().unwrap_or(Path::new(".")),
                )?;
                self.insert(
                    &name,
                    value,
                    format!("file:{}:{group}.{key}", path.display()),
                )?;
            }
        }
        Ok(())
    }

    fn load_environment(
        &mut self,
        command: &clap::Command,
        environment: impl Fn(&str) -> Option<String>,
    ) -> Result<()> {
        for name in known_arguments(command).keys() {
            if matches!(
                name.as_str(),
                "settings" | "print-effective-config" | "help"
            ) {
                continue;
            }
            let variable = format!(
                "SKIPPY_SERVE_{}",
                name.replace('-', "_").to_ascii_uppercase()
            );
            if let Some(value) = environment(&variable) {
                self.insert(
                    name,
                    Value::String(value),
                    format!("environment:{variable}"),
                )?;
            }
        }
        Ok(())
    }

    fn load_cli(&mut self, matches: &ArgMatches, command: &clap::Command) -> Result<()> {
        let (_, serve) = matches.subcommand().context("expected serve command")?;
        for (name, id) in known_arguments(command) {
            let owner = if serve.try_get_raw(&id).is_ok() {
                serve
            } else {
                matches
            };
            if owner.value_source(&id) != Some(ValueSource::CommandLine) {
                continue;
            }
            let raw = owner
                .get_raw(&id)
                .context("setting has no value")?
                .map(|value| {
                    value
                        .to_str()
                        .context("serving options must be UTF-8")
                        .map(str::to_owned)
                })
                .collect::<Result<Vec<_>>>()?;
            let value = if name == "runtime-bundle" {
                serde_json::to_value(raw)?
            } else {
                Value::String(raw.first().cloned().unwrap_or_else(|| "true".into()))
            };
            self.insert(&name, value, "cli".into())?;
        }
        Ok(())
    }

    fn append_arguments(&self, argv: &mut Vec<OsString>) -> Result<()> {
        for (name, value) in &self.values {
            let spec = OPTIONS.iter().find(|spec| spec.name == name);
            let values = if name == "runtime-bundle" {
                value
                    .as_array()
                    .cloned()
                    .unwrap_or_else(|| vec![value.clone()])
            } else {
                vec![value.clone()]
            };
            for value in values {
                let encoded = if spec.is_some_and(|spec| matches!(spec.kind, Kind::Json)) {
                    serde_json::to_string(&value)?
                } else {
                    value
                        .as_str()
                        .map_or_else(|| value.to_string(), str::to_owned)
                };
                argv.push(OsString::from(format!("--{name}={encoded}")));
            }
        }
        Ok(())
    }
}

fn parse_value(kind: Kind, value: Value) -> Result<Value> {
    let Some(text) = value.as_str() else {
        return Ok(value);
    };
    match kind {
        Kind::Bool => Ok(Value::Bool(text.parse().context("expected true or false")?)),
        Kind::Unsigned => Ok(Value::from(
            text.parse::<u64>()
                .context("expected an unsigned integer")?,
        )),
        Kind::Integer => Ok(Value::from(
            text.parse::<i64>().context("expected an integer")?,
        )),
        Kind::Number => {
            let number = text.parse::<f64>().context("expected a number")?;
            anyhow::ensure!(number.is_finite(), "expected a finite number");
            Ok(Value::from(number))
        }
        Kind::Json => {
            let content = if let Some(path) = text.strip_prefix('@') {
                std::fs::read_to_string(path).with_context(|| format!("read JSON {path}"))?
            } else {
                text.to_owned()
            };
            serde_json::from_str(&content).context("expected JSON or @file")
        }
        Kind::Content => {
            if let Some(path) = text.strip_prefix('@') {
                Ok(Value::String(
                    std::fs::read_to_string(path).with_context(|| format!("read text {path}"))?,
                ))
            } else {
                Ok(value)
            }
        }
        Kind::Text | Kind::Bytes => Ok(value),
    }
}

fn file_value(name: &str, value: Value, directory: &Path) -> Result<Value> {
    if name == "runtime-bundle"
        && let Some(values) = value.as_array()
    {
        return values
            .iter()
            .map(|value| file_value(name, value.clone(), directory))
            .collect::<Result<Vec<_>>>()
            .map(Value::Array);
    }
    let Some(text) = value.as_str() else {
        return Ok(value);
    };
    if name == "model"
        && (text.starts_with("./")
            || text.starts_with("../")
            || text.ends_with(".gguf")
            || directory.join(text).exists())
    {
        return Ok(Value::String(
            directory.join(text).to_string_lossy().into_owned(),
        ));
    }
    if is_path(name) && !Path::new(text).is_absolute() {
        return Ok(Value::String(
            directory.join(text).to_string_lossy().into_owned(),
        ));
    }
    if let Some(path) = text.strip_prefix('@') {
        let path = directory.join(path);
        let kind = OPTIONS
            .iter()
            .find(|spec| spec.name == name)
            .map(|spec| spec.kind);
        if matches!(kind, Some(Kind::Json | Kind::Content)) {
            return parse_value(
                kind.context("missing content kind")?,
                Value::String(format!("@{}", path.display())),
            );
        }
    }
    Ok(value)
}

fn is_path(name: &str) -> bool {
    matches!(
        name,
        "config"
            | "model-path"
            | "checkpoint-imatrix"
            | "mmproj"
            | "hash-cache"
            | "topology"
            | "speculative-config"
            | "draft-model-path"
            | "native-mtp-draft-model-path"
            | "runtime-cache"
            | "runtime-bundle"
            | "kv-cache-disk-dir"
    )
}

fn known_arguments(command: &clap::Command) -> BTreeMap<String, String> {
    command
        .get_arguments()
        .chain(
            command
                .find_subcommand("serve")
                .into_iter()
                .flat_map(clap::Command::get_arguments),
        )
        .filter_map(|arg| {
            arg.get_long()
                .map(|name| (name.to_owned(), arg.get_id().to_string()))
        })
        .collect()
}

pub fn section(name: &str) -> &'static str {
    if let Some(spec) = OPTIONS.iter().find(|spec| spec.name == name) {
        return spec.section;
    }
    if name.starts_with("runtime-") {
        return "runtime";
    }
    if name.starts_with("generation-")
        || name.starts_with("adaptive-generation-")
        || name.starts_with("prefill-")
    {
        return "scheduling";
    }
    if name.starts_with("downstream-wire-") {
        return "network";
    }
    if name.contains("speculative") || name.starts_with("draft-") || name.starts_with("native-mtp-")
    {
        return "speculative";
    }
    if name.starts_with("kv-cache-") {
        return "cache";
    }
    match name {
        "ctx-size" => "kv",
        "n-gpu-layers" => "execution",
        "mmproj" => "media",
        "bind-addr" | "model-id" | "default-max-tokens" | "guardrails" | "prompt" => "api",
        "config"
        | "topology"
        | "stage-transport"
        | "worker-only"
        | "max-inflight"
        | "reply-credit-limit"
        | "async-prefill-forward"
        | "no-async-prefill-forward"
        | "downstream-connect-timeout-secs" => "distributed",
        "output"
        | "debug"
        | "metrics-otlp-grpc"
        | "telemetry-queue-capacity"
        | "telemetry-level"
        | "startup-timeout-secs" => "diagnostics",
        "settings" | "print-effective-config" | "help" => "settings",
        _ => "model",
    }
}

fn heading(section: &str) -> &'static str {
    match section {
        "model" => "Model loading",
        "execution" => "Devices and execution",
        "kv" => "Context and KV memory",
        "cache" => "Prompt caching",
        "scheduling" => "Scheduling and prefill",
        "sampling" => "Sampling defaults",
        "chat" => "Chat and reasoning",
        "speculative" => "Speculative decoding",
        "media" => "Multimodal",
        "api" => "API and compatibility",
        "distributed" => "Distributed stages",
        "runtime" => "Native runtime",
        "network" => "Network simulation (testing)",
        "settings" => "Settings",
        _ => "Diagnostics",
    }
}

fn section_order(section: &str) -> usize {
    [
        "settings",
        "model",
        "execution",
        "kv",
        "cache",
        "scheduling",
        "sampling",
        "chat",
        "speculative",
        "media",
        "api",
        "distributed",
        "runtime",
        "diagnostics",
        "network",
    ]
    .iter()
    .position(|candidate| *candidate == section)
    .unwrap_or(15)
        * 1000
}

fn description(name: &str) -> Option<&'static str> {
    match name {
        "topology" => Some("Admitted topology JSON matching the prepared stage configuration"),
        "default-max-tokens" => Some(
            "Default completion ceiling when a request omits its output limit; clamped to remaining context",
        ),
        "prefill-chunk-size" => Some("Tokens per fixed prefill chunk"),
        "prefill-adaptive-start" => Some("Initial adaptive prefill chunk size in tokens"),
        "prefill-adaptive-step" => Some("Token increment when adaptive prefill grows a chunk"),
        "prefill-adaptive-max" => Some("Maximum adaptive prefill chunk size in tokens"),
        "prefill-adaptive-target-ms" => {
            Some("Target processing time per adaptive prefill chunk in milliseconds")
        }
        "startup-timeout-secs" => Some("Maximum seconds to wait for listener readiness"),
        "metrics-otlp-grpc" => Some("OTLP/gRPC metrics collector endpoint"),
        "telemetry-queue-capacity" => Some("Bounded telemetry event queue capacity"),
        "telemetry-level" => Some("Serving telemetry detail: off, summary, debug"),
        "max-inflight" => {
            Some("Maximum outstanding binary-stage messages, bounded by admitted lane count")
        }
        "reply-credit-limit" => {
            Some("Maximum downstream replies outstanding before the sender waits")
        }
        "downstream-connect-timeout-secs" => {
            Some("Seconds allowed to establish a downstream stage connection")
        }
        "speculative-window" => Some("Maximum draft-model proposal length in tokens"),
        "adaptive-speculative-window" => {
            Some("Adapt draft-model proposal length according to acceptance")
        }
        _ => None,
    }
}

use std::str::FromStr;

use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use serde::Serialize;

use crate::output::{format_bytes, print_info, print_json_pretty, print_warn};
use crate::splits::SplitWindow;
use crate::types::ConvertOutputType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize)]
pub(crate) enum MemoryPolicy {
    Advisory,
    #[default]
    Hard,
}

impl MemoryPolicy {
    pub(crate) fn is_hard(self) -> bool {
        matches!(self, Self::Hard)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemorySize(u64);

impl MemorySize {
    pub(crate) fn bytes(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn from_bytes_for_tests(bytes: u64) -> Self {
        Self(bytes)
    }
}

impl FromStr for MemorySize {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("memory size is empty".to_string());
        }
        let suffix_start = trimmed
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(trimmed.len());
        let (digits, suffix) = trimmed.split_at(suffix_start);
        if digits.is_empty() {
            return Err(format!("memory size {raw:?} is missing a number"));
        }
        let value = digits
            .parse::<u64>()
            .map_err(|err| format!("invalid memory size {raw:?}: {err}"))?;
        let multiplier = match suffix.trim().to_ascii_lowercase().as_str() {
            "" | "b" => 1,
            "k" | "kb" | "kib" => 1024,
            "m" | "mb" | "mib" => 1024 * 1024,
            "g" | "gb" | "gib" => 1024 * 1024 * 1024,
            "t" | "tb" | "tib" => 1024_u64.pow(4),
            other => return Err(format!("unsupported memory size suffix {other:?}")),
        };
        value
            .checked_mul(multiplier)
            .map(Self)
            .ok_or_else(|| format!("memory size {raw:?} is too large"))
    }
}

#[derive(Debug, Serialize)]
struct MemoryBudgetPlan<'a> {
    kind: &'a str,
    backend: &'a str,
    max_memory_bytes: Option<u64>,
    memory_policy: MemoryPolicy,
    watchdog_seconds: Option<u64>,
    hard_limit: bool,
    first_split: u32,
    last_split: u32,
    window_shards: u32,
    stream_buffer_bytes: Option<usize>,
    estimated_stream_working_set_bytes: Option<u64>,
    llama_quantize_env_bytes: Option<u64>,
}

pub(crate) struct MemoryBudgetPlanInput<'a> {
    pub(crate) kind: &'a str,
    pub(crate) backend: &'a str,
    pub(crate) max_memory: Option<MemorySize>,
    pub(crate) memory_policy: MemoryPolicy,
    pub(crate) watchdog_seconds: Option<u64>,
    pub(crate) window: SplitWindow,
    pub(crate) stream_buffer_bytes: Option<usize>,
    pub(crate) estimated_stream_working_set_bytes: Option<u64>,
    pub(crate) llama_quantize_env_bytes: Option<u64>,
    pub(crate) json: bool,
}

pub(crate) fn print_memory_budget_plan(input: MemoryBudgetPlanInput<'_>) -> Result<()> {
    if input.max_memory.is_none() && input.watchdog_seconds.is_none() {
        return Ok(());
    }
    let plan = MemoryBudgetPlan {
        kind: input.kind,
        backend: input.backend,
        max_memory_bytes: input.max_memory.map(MemorySize::bytes),
        memory_policy: input.memory_policy,
        watchdog_seconds: input.watchdog_seconds,
        hard_limit: input.memory_policy.is_hard() && input.max_memory.is_some(),
        first_split: input.window.first_split,
        last_split: input.window.last_split,
        window_shards: input
            .window
            .last_split
            .saturating_sub(input.window.first_split)
            .saturating_add(1),
        stream_buffer_bytes: input.stream_buffer_bytes,
        estimated_stream_working_set_bytes: input.estimated_stream_working_set_bytes,
        llama_quantize_env_bytes: input.llama_quantize_env_bytes,
    };
    if input.json {
        print_json_pretty(&serde_json::json!({
            "event": format!("{}_memory_budget", input.kind),
            "plan": plan,
        }))?;
    } else if plan.hard_limit {
        print_warn(format!(
            "{} memory budget: hard cap {}",
            input.kind,
            format_bytes(input.max_memory.map(MemorySize::bytes).unwrap_or_default())
        ));
    } else {
        print_info(format!("{} memory budget configured", input.kind));
    }
    Ok(())
}

pub(crate) fn effective_stream_buffer_bytes(
    requested: usize,
    max_memory: Option<MemorySize>,
) -> Result<usize> {
    if let Some(max_memory) = max_memory {
        let budget_limited = bytes_to_usize(max_memory)? / 64;
        return Ok(requested.min(budget_limited.max(1)));
    }
    Ok(requested)
}

pub(crate) fn native_convert_stream_working_set_bytes(
    stream_buffer_bytes: usize,
    output_type: Option<ConvertOutputType>,
) -> Result<u64> {
    let multiplier = match output_type {
        None => 1,
        Some(ConvertOutputType::F16 | ConvertOutputType::Bf16) => 2,
        Some(ConvertOutputType::F32) => 3,
        Some(other) => {
            anyhow::bail!(
                "native conversion does not support output type {}",
                other.as_arg()
            );
        }
    };
    (stream_buffer_bytes as u64)
        .checked_mul(multiplier)
        .context("native conversion stream working-set estimate overflow")
}

pub(crate) fn enforce_memory_budget(
    label: &str,
    estimated_bytes: u64,
    max_memory: Option<MemorySize>,
    policy: MemoryPolicy,
) -> Result<()> {
    let Some(max_memory) = max_memory else {
        return Ok(());
    };
    if estimated_bytes <= max_memory.bytes() {
        return Ok(());
    }
    print_warn(format!(
        "{label} estimated working set {} exceeds budget {} ({policy:?})",
        format_bytes(estimated_bytes),
        format_bytes(max_memory.bytes())
    ));
    ensure!(
        !policy.is_hard(),
        "{label} estimated working set {} bytes exceeds --max-memory {} bytes",
        estimated_bytes,
        max_memory.bytes()
    );
    Ok(())
}

fn bytes_to_usize(value: MemorySize) -> Result<usize> {
    usize::try_from(value.bytes()).context("--max-memory does not fit usize on this platform")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_memory_sizes() {
        assert_eq!("1".parse::<MemorySize>().unwrap().bytes(), 1);
        assert_eq!("2K".parse::<MemorySize>().unwrap().bytes(), 2048);
        assert_eq!(
            "3MiB".parse::<MemorySize>().unwrap().bytes(),
            3 * 1024 * 1024
        );
        assert_eq!(
            "4G".parse::<MemorySize>().unwrap().bytes(),
            4 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn derives_effective_stream_buffer_from_budget() {
        assert_eq!(effective_stream_buffer_bytes(1024, None).unwrap(), 1024);
        assert_eq!(
            effective_stream_buffer_bytes(1024, Some(MemorySize::from_bytes_for_tests(128)))
                .unwrap(),
            2
        );
        assert_eq!(
            effective_stream_buffer_bytes(1024, Some(MemorySize::from_bytes_for_tests(1))).unwrap(),
            1
        );
    }

    #[test]
    fn estimates_native_convert_stream_working_set() {
        assert_eq!(
            native_convert_stream_working_set_bytes(1024, None).unwrap(),
            1024
        );
        assert_eq!(
            native_convert_stream_working_set_bytes(1024, Some(ConvertOutputType::Bf16)).unwrap(),
            2048
        );
        assert_eq!(
            native_convert_stream_working_set_bytes(1024, Some(ConvertOutputType::F32)).unwrap(),
            3072
        );
        assert!(
            native_convert_stream_working_set_bytes(1024, Some(ConvertOutputType::Q8_0)).is_err()
        );
        assert!(
            native_convert_stream_working_set_bytes(1024, Some(ConvertOutputType::Auto)).is_err()
        );
    }

    #[test]
    fn hard_memory_policy_rejects_over_budget_working_set() {
        let max_memory = Some(MemorySize::from_bytes_for_tests(100));
        assert!(enforce_memory_budget("test", 100, max_memory, MemoryPolicy::Hard).is_ok());
        assert!(enforce_memory_budget("test", 101, max_memory, MemoryPolicy::Hard).is_err());
        assert!(enforce_memory_budget("test", 101, max_memory, MemoryPolicy::Advisory).is_ok());
    }
}

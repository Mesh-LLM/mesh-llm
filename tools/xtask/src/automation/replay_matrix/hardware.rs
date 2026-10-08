//! Read the fixed macOS hardware fields used by replay cohort identity.
use crate::{
    command::DynResult,
    command_interrupt::Interrupt,
    process::{self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
    repository::{check_args::Grammar, check_report::CheckReport},
};
use serde::Serialize;
use std::{collections::BTreeMap, path::Path, time::Duration};

#[derive(Serialize)]
struct Hardware {
    machine_model: String,
    chip: String,
    gpu_cores: u64,
    unified_memory_bytes: u64,
    os_version: String,
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix hardware --output PATH --gpu-cores COUNT",
        values: &["--output", "--gpu-cores"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let output = Path::new(parsed.last("--output").ok_or("missing --output")?);
    let gpu = parsed
        .last("--gpu-cores")
        .ok_or("missing --gpu-cores")?
        .parse::<u64>()?;
    if gpu == 0 {
        return Err("GPU core count must be positive".into());
    }
    let interrupt = Interrupt::install()?;
    let result = read(gpu, &interrupt);
    interrupt.finish()?;
    let hardware = result?;
    let mut bytes = serde_json::to_vec_pretty(&hardware)?;
    bytes.push(b'\n');
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, bytes)?;
    Ok(())
}
fn read(gpu_cores: u64, interrupt: &Interrupt) -> DynResult<Hardware> {
    let machine_model = query("/usr/sbin/sysctl", &["-n", "hw.model"], interrupt)?;
    let chip = query(
        "/usr/sbin/sysctl",
        &["-n", "machdep.cpu.brand_string"],
        interrupt,
    )?;
    let memory = query("/usr/sbin/sysctl", &["-n", "hw.memsize"], interrupt)?;
    let os_version = query("/usr/bin/sw_vers", &["-productVersion"], interrupt)?;
    assemble(machine_model, chip, gpu_cores, &memory, os_version)
}
fn assemble(
    machine_model: String,
    chip: String,
    gpu_cores: u64,
    memory: &str,
    os_version: String,
) -> DynResult<Hardware> {
    let unified_memory_bytes = memory.parse::<u64>()?;
    if unified_memory_bytes == 0 {
        return Err("unified memory size must be positive".into());
    }
    Ok(Hardware {
        machine_model,
        chip,
        gpu_cores,
        unified_memory_bytes,
        os_version,
    })
}
fn query(executable: &str, args: &[&str], interrupt: &Interrupt) -> DynResult<String> {
    let spec = ProcessSpec {
        executable: executable.into(),
        arguments: args
            .iter()
            .map(|arg| Value::Public((*arg).into()))
            .collect(),
        cwd: std::env::current_dir()?,
        environment: BTreeMap::new(),
    };
    let report = process::supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 4096,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &interrupt.cancellation(),
        OutputFiles::default(),
    )?;
    if !report.success() || report.stdout.truncated {
        return Err(format!(
            "hardware query failed for {executable}: {:?}",
            report.outcome
        )
        .into());
    }
    text(&report.stdout.bytes_retained)
}
fn text(bytes: &[u8]) -> DynResult<String> {
    let value = std::str::from_utf8(bytes)?.trim();
    if value.is_empty() || value.contains(['\n', '\r']) {
        return Err("hardware query must return one nonempty field".into());
    }
    Ok(value.into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cohort_hardware_retains_exact_fields_and_integer_memory() {
        let hardware = assemble(
            "Mac16,9".into(),
            "Apple M3 Ultra".into(),
            80,
            "274877906944",
            "26.0".into(),
        )
        .unwrap();
        let value = serde_json::to_value(hardware).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 5);
        assert_eq!(value["gpu_cores"], 80);
        assert_eq!(value["unified_memory_bytes"], 274877906944_u64);
        assert_eq!(value["chip"], "Apple M3 Ultra");
    }
    #[test]
    fn hardware_queries_reject_empty_multiple_and_non_utf8_fields_and_invalid_memory() {
        assert_eq!(text(b"Mac16,9\n").unwrap(), "Mac16,9");
        for bytes in [b"\n".as_slice(), b"one\ntwo\n", &[255]] {
            assert!(text(bytes).is_err());
        }
        for memory in ["True", "0", "-1", "1.0", "18446744073709551616"] {
            assert!(assemble("fixture".into(), "chip".into(), 80, memory, "26.0".into()).is_err());
        }
    }
}

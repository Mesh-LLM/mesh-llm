use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use reqwest::StatusCode;
use serde::Serialize;
use serde_json::{Value, json};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use crate::cli::Cli;
use crate::cli::commands::{gpus, plugin};
use crate::cli::terminal_progress::start_indeterminate_progress;
use crate::runtime::instance;
use crate::system::backend::BinaryFlavor;
use crate::system::hardware::HardwareSurvey;

pub(crate) const DEFAULT_MAX_LOG_BYTES: u64 = 20 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum DoctorLogTarget {
    /// Prefer the newest running process; fall back to the newest previous process.
    Auto,
    /// Include logs from the newest currently running process.
    Current,
    /// Include logs from the newest known process, running or not.
    Last,
}

pub(crate) struct DoctorBundleOptions {
    pub(crate) output: Option<PathBuf>,
    pub(crate) target: DoctorLogTarget,
    pub(crate) pid: Option<u32>,
    pub(crate) port: Option<u16>,
    pub(crate) max_log_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct RuntimeInstanceInfo {
    pid: u32,
    api_port: Option<u16>,
    version: Option<String>,
    started_at_unix: Option<i64>,
    mesh_llm_binary: Option<String>,
    command: Option<String>,
    runtime_dir: PathBuf,
    runtime_metadata: bool,
    is_live: bool,
    sort_time_unix: i64,
}

#[derive(Clone, Debug, Serialize)]
struct IncludedFile {
    zip_path: String,
    source_path: Option<PathBuf>,
    bytes_written: u64,
    truncated: bool,
    original_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct DoctorManifest {
    generated_at: String,
    output_path: PathBuf,
    target: String,
    requested_pid: Option<u32>,
    requested_port: Option<u16>,
    max_log_bytes: u64,
    runtime_root: Option<PathBuf>,
    selected_instance: Option<RuntimeInstanceInfo>,
    included_files: Vec<IncludedFile>,
    warnings: Vec<String>,
}

pub(crate) async fn run_doctor_bundle(cli: &Cli, options: DoctorBundleOptions) -> Result<()> {
    let output_path = options.output.unwrap_or_else(default_output_path);
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }

    let mut progress = start_indeterminate_progress("Building doctor bundle");
    let hw = gpus::collect_gpus_survey();
    let runtime_root = instance::runtime_root().ok();
    let all_instances = runtime_root
        .as_deref()
        .map(load_runtime_instances)
        .transpose()?
        .unwrap_or_default();
    let selected = select_runtime_instance(&all_instances, options.target, options.pid)?;
    let api_port = options
        .port
        .or_else(|| selected.as_ref().and_then(|info| info.api_port))
        .unwrap_or(3131);

    let file = File::create(&output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut bundle = DoctorZip::new(file);
    let mut warnings = Vec::new();

    add_text(
        &mut bundle,
        "README.txt",
        doctor_readme(options.max_log_bytes),
    )?;
    add_json(
        &mut bundle,
        "system.json",
        &system_report(cli, &hw, api_port),
    )?;
    add_json(&mut bundle, "gpus.json", &gpus::gpus_json(&hw))?;
    add_plugin_inventory(&mut bundle, cli, &mut warnings)?;
    add_config_file(&mut bundle, cli, &mut warnings)?;
    add_json(
        &mut bundle,
        "runtime/instances.json",
        &json!({
            "runtime_root": runtime_root,
            "instances": all_instances,
        }),
    )?;

    if let Some(info) = selected.as_ref() {
        add_runtime_files(&mut bundle, info, options.max_log_bytes, &mut warnings)?;
    } else {
        warnings
            .push("no runtime instance was found; bundle contains system diagnostics only".into());
    }

    add_api_snapshots(&mut bundle, api_port, &mut warnings).await?;

    let mut manifest = DoctorManifest {
        generated_at: utc_timestamp(),
        output_path: output_path.clone(),
        target: format!("{:?}", options.target).to_ascii_lowercase(),
        requested_pid: options.pid,
        requested_port: options.port,
        max_log_bytes: options.max_log_bytes,
        runtime_root,
        selected_instance: selected,
        included_files: Vec::new(),
        warnings,
    };
    manifest.included_files.clone_from(&bundle.included_files);
    add_json(&mut bundle, "manifest.json", &manifest)?;
    bundle.finish()?;
    progress.finish();

    println!("✅ Created doctor bundle");
    println!();
    println!("Path: {}", output_path.display());
    println!(
        "Logs: {}",
        selected_runtime_label(manifest.selected_instance.as_ref(), &all_instances)
    );
    println!("Files: {}", manifest.included_files.len());
    if !manifest.warnings.is_empty() {
        println!();
        println!("⚠️  Warnings");
        for warning in &manifest.warnings {
            println!("  - {warning}");
        }
    }
    Ok(())
}

fn selected_runtime_label(
    info: Option<&RuntimeInstanceInfo>,
    instances: &[RuntimeInstanceInfo],
) -> String {
    let Some(info) = info else {
        return "none selected".to_string();
    };
    if info.is_live {
        return format!("running pid {}", info.pid);
    }
    if instances.iter().any(|instance| instance.is_live) {
        return format!("previous pid {}", info.pid);
    }
    format!(
        "no running mesh-llm process found; using previous pid {}",
        info.pid
    )
}

struct DoctorZip {
    writer: ZipWriter<File>,
    options: SimpleFileOptions,
    included_files: Vec<IncludedFile>,
}

impl DoctorZip {
    fn new(file: File) -> Self {
        Self {
            writer: ZipWriter::new(file),
            options: SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
            included_files: Vec::new(),
        }
    }

    fn add_bytes(
        &mut self,
        zip_path: &str,
        source_path: Option<PathBuf>,
        bytes: &[u8],
        truncated: bool,
        original_bytes: Option<u64>,
    ) -> Result<()> {
        self.writer
            .start_file(zip_path, self.options)
            .with_context(|| format!("failed to start zip entry {zip_path}"))?;
        self.writer
            .write_all(bytes)
            .with_context(|| format!("failed to write zip entry {zip_path}"))?;
        self.included_files.push(IncludedFile {
            zip_path: zip_path.to_string(),
            source_path,
            bytes_written: bytes.len() as u64,
            truncated,
            original_bytes,
        });
        Ok(())
    }

    fn finish(self) -> Result<()> {
        self.writer
            .finish()
            .context("failed to finish doctor zip")?;
        Ok(())
    }
}

fn add_text(zip: &mut DoctorZip, zip_path: &str, contents: impl AsRef<str>) -> Result<()> {
    zip.add_bytes(zip_path, None, contents.as_ref().as_bytes(), false, None)
}

fn add_json<T: Serialize>(zip: &mut DoctorZip, zip_path: &str, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    zip.add_bytes(zip_path, None, &bytes, false, None)
}

fn add_plugin_inventory(zip: &mut DoctorZip, cli: &Cli, warnings: &mut Vec<String>) -> Result<()> {
    match plugin::plugin_inventory_json(cli) {
        Ok(value) => add_json(zip, "plugins.json", &value),
        Err(err) => {
            warnings.push(format!("could not collect plugin inventory: {err:#}"));
            add_json(
                zip,
                "plugins.json",
                &json!({"ok": false, "error": err.to_string()}),
            )
        }
    }
}

fn add_config_file(zip: &mut DoctorZip, cli: &Cli, warnings: &mut Vec<String>) -> Result<()> {
    let config_path = match crate::plugin::config_path(cli.config.as_deref()) {
        Ok(path) => path,
        Err(err) => {
            warnings.push(format!("could not resolve config path: {err:#}"));
            return Ok(());
        }
    };
    if !config_path.exists() {
        return Ok(());
    }
    add_existing_file_full(zip, &config_path, "config/config.toml", warnings)
}

fn add_runtime_files(
    zip: &mut DoctorZip,
    info: &RuntimeInstanceInfo,
    max_log_bytes: u64,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if !info.runtime_metadata {
        warnings.push(format!(
            "selected process has no runtime metadata directory: {}",
            info.runtime_dir.display()
        ));
        return Ok(());
    }

    add_existing_file(
        zip,
        &info.runtime_dir.join("owner.json"),
        "runtime/owner.json",
        max_log_bytes,
        warnings,
    )?;

    let logs_dir = info.runtime_dir.join("logs");
    if !logs_dir.exists() {
        warnings.push(format!(
            "selected runtime has no logs directory: {}",
            logs_dir.display()
        ));
        return Ok(());
    }
    for path in collect_regular_files(&logs_dir)? {
        let relative = path
            .strip_prefix(&logs_dir)
            .unwrap_or(path.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        add_existing_file(
            zip,
            &path,
            &format!("runtime/logs/{relative}"),
            max_log_bytes,
            warnings,
        )?;
    }
    Ok(())
}

fn add_existing_file(
    zip: &mut DoctorZip,
    path: &Path,
    zip_path: &str,
    max_bytes: u64,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if !path.exists() {
        warnings.push(format!("missing diagnostic file: {}", path.display()));
        return Ok(());
    }

    let original_bytes = fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    let (bytes, truncated) = read_file_tail(path, max_bytes)?;
    if truncated {
        warnings.push(format!(
            "truncated {} to the last {} bytes",
            path.display(),
            bytes.len()
        ));
    }
    zip.add_bytes(
        zip_path,
        Some(path.to_path_buf()),
        &bytes,
        truncated,
        Some(original_bytes),
    )
}

fn add_existing_file_full(
    zip: &mut DoctorZip,
    path: &Path,
    zip_path: &str,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if !path.exists() {
        warnings.push(format!("missing diagnostic file: {}", path.display()));
        return Ok(());
    }

    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    zip.add_bytes(
        zip_path,
        Some(path.to_path_buf()),
        &bytes,
        false,
        Some(bytes.len() as u64),
    )
}

fn read_file_tail(path: &Path, max_bytes: u64) -> Result<(Vec<u8>, bool)> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let len = file.metadata()?.len();
    let truncated = max_bytes > 0 && len > max_bytes;
    if truncated {
        file.seek(SeekFrom::Start(len - max_bytes))?;
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok((bytes, truncated))
}

fn collect_regular_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_regular_files_inner(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_regular_files_inner(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_regular_files_inner(&path, files)?;
        } else if file_type.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

async fn add_api_snapshots(
    zip: &mut DoctorZip,
    port: u16,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    for (zip_path, endpoint) in [
        ("api/status.json", "/api/status"),
        ("api/runtime.json", "/api/runtime"),
        ("api/runtime-processes.json", "/api/runtime/processes"),
        ("api/models.json", "/api/models"),
        ("api/v1-models.json", "/v1/models"),
    ] {
        let value = fetch_api_snapshot(&client, port, endpoint).await;
        if value["ok"] == json!(false) {
            warnings.push(format!("could not fetch http://127.0.0.1:{port}{endpoint}"));
        }
        add_json(zip, zip_path, &value)?;
    }
    Ok(())
}

async fn fetch_api_snapshot(client: &reqwest::Client, port: u16, endpoint: &str) -> Value {
    let url = format!("http://127.0.0.1:{port}{endpoint}");
    match client.get(&url).send().await {
        Ok(response) => api_response_json(url, response).await,
        Err(err) => json!({"ok": false, "url": url, "error": err.to_string()}),
    }
}

async fn api_response_json(url: String, response: reqwest::Response) -> Value {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if status != StatusCode::OK {
        return json!({"ok": status.is_success(), "url": url, "status": status.as_u16(), "body": body});
    }
    if let Ok(json_body) = serde_json::from_str::<Value>(&body) {
        return json!({"ok": true, "url": url, "status": status.as_u16(), "body": json_body});
    }
    json!({"ok": status.is_success(), "url": url, "status": status.as_u16(), "body": body})
}

fn load_runtime_instances(root: &Path) -> Result<Vec<RuntimeInstanceInfo>> {
    if !root.exists() {
        return Ok(discover_running_mesh_processes(root, &[]));
    }

    let mut instances = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let Some(info) = read_runtime_instance(&path) else {
            continue;
        };
        instances.push(info);
    }
    instances.extend(discover_running_mesh_processes(root, &instances));
    instances.sort_by(|a, b| {
        b.sort_time_unix
            .cmp(&a.sort_time_unix)
            .then_with(|| b.pid.cmp(&a.pid))
    });
    Ok(instances)
}

fn read_runtime_instance(runtime_dir: &Path) -> Option<RuntimeInstanceInfo> {
    let owner_json = fs::read_to_string(runtime_dir.join("owner.json")).ok()?;
    let value = serde_json::from_str::<Value>(&owner_json).ok()?;
    let pid = value["pid"]
        .as_u64()
        .and_then(|pid| u32::try_from(pid).ok())?;
    let started_at_unix = value["started_at_unix"]
        .as_i64()
        .filter(|started_at| *started_at > 0);
    Some(RuntimeInstanceInfo {
        pid,
        api_port: value["api_port"]
            .as_u64()
            .and_then(|port| u16::try_from(port).ok()),
        version: value["version"].as_str().map(str::to_owned),
        started_at_unix,
        mesh_llm_binary: value["mesh_llm_binary"].as_str().map(str::to_owned),
        command: None,
        runtime_dir: runtime_dir.to_path_buf(),
        runtime_metadata: true,
        is_live: instance::validate::process_liveness(pid) != instance::validate::Liveness::Dead,
        sort_time_unix: started_at_unix.unwrap_or_else(|| path_modified_unix(runtime_dir)),
    })
}

fn discover_running_mesh_processes(
    runtime_root: &Path,
    known_instances: &[RuntimeInstanceInfo],
) -> Vec<RuntimeInstanceInfo> {
    let known_pids = known_instances
        .iter()
        .map(|info| info.pid)
        .collect::<std::collections::BTreeSet<_>>();
    parse_running_mesh_processes(
        &process_table_output().unwrap_or_default(),
        runtime_root,
        &known_pids,
        std::process::id(),
        chrono::Utc::now().timestamp(),
    )
}

#[cfg(unix)]
fn process_table_output() -> Option<String> {
    command_stdout("ps", &["-eo", "pid=,command="])
}

#[cfg(not(unix))]
fn process_table_output() -> Option<String> {
    None
}

fn parse_running_mesh_processes(
    output: &str,
    runtime_root: &Path,
    known_pids: &std::collections::BTreeSet<u32>,
    current_pid: u32,
    sort_time_unix: i64,
) -> Vec<RuntimeInstanceInfo> {
    output
        .lines()
        .filter_map(|line| parse_process_table_row(line))
        .filter(|(pid, command)| {
            *pid != current_pid && !known_pids.contains(pid) && is_mesh_llm_runtime_command(command)
        })
        .map(|(pid, command)| RuntimeInstanceInfo {
            pid,
            api_port: None,
            version: None,
            started_at_unix: None,
            mesh_llm_binary: command.split_whitespace().next().map(str::to_string),
            command: Some(command.to_string()),
            runtime_dir: runtime_root.join(pid.to_string()),
            runtime_metadata: false,
            is_live: true,
            sort_time_unix,
        })
        .collect()
}

fn parse_process_table_row(line: &str) -> Option<(u32, &str)> {
    let trimmed = line.trim_start();
    let split_at = trimmed.find(char::is_whitespace)?;
    let pid = trimmed[..split_at].parse().ok()?;
    let command = trimmed[split_at..].trim();
    (!command.is_empty()).then_some((pid, command))
}

fn is_mesh_llm_runtime_command(command: &str) -> bool {
    if !command.contains("mesh-llm") {
        return false;
    }
    let args = command.split_whitespace().collect::<Vec<_>>();
    let Some(binary_index) = args.iter().position(|arg| arg.contains("mesh-llm")) else {
        return false;
    };
    args[binary_index + 1..].iter().any(|arg| {
        matches!(
            *arg,
            "client" | "serve" | "--client" | "--auto" | "--model" | "--gguf"
        )
    })
}

fn select_runtime_instance(
    instances: &[RuntimeInstanceInfo],
    target: DoctorLogTarget,
    pid: Option<u32>,
) -> Result<Option<RuntimeInstanceInfo>> {
    if let Some(pid) = pid {
        let Some(info) = instances.iter().find(|info| info.pid == pid) else {
            bail!("no mesh-llm runtime instance found for pid {pid}");
        };
        return Ok(Some(info.clone()));
    }

    let selected = match target {
        DoctorLogTarget::Auto => instances
            .iter()
            .find(|info| info.is_live)
            .or_else(|| instances.first()),
        DoctorLogTarget::Current => instances.iter().find(|info| info.is_live),
        DoctorLogTarget::Last => instances.first(),
    };
    Ok(selected.cloned())
}

fn system_report(cli: &Cli, hw: &HardwareSurvey, api_port: u16) -> Value {
    json!({
        "generated_at": utc_timestamp(),
        "mesh_llm": {
            "version": crate::VERSION,
            "current_exe": std::env::current_exe().ok(),
            "current_dir": std::env::current_dir().ok(),
        },
        "platform": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "family": std::env::consts::FAMILY,
        },
        "flavor": {
            "requested_llama_flavor": cli.llama_flavor.map(BinaryFlavor::suffix),
            "detected_host_flavor": detect_host_flavor(hw).map(BinaryFlavor::suffix),
        },
        "system": {
            "memory": collect_memory_info(),
            "cpu": collect_cpu_info(),
        },
        "runtime": {
            "api_port": api_port,
            "runtime_root": instance::runtime_root().ok(),
        },
    })
}

fn detect_host_flavor(hw: &HardwareSurvey) -> Option<BinaryFlavor> {
    if cfg!(target_os = "macos") {
        return Some(BinaryFlavor::Metal);
    }

    let backend_devices = hw
        .gpus
        .iter()
        .filter_map(|gpu| gpu.backend_device.as_deref())
        .collect::<Vec<_>>();
    if backend_devices
        .iter()
        .any(|device| device.starts_with("CUDA"))
    {
        return Some(BinaryFlavor::Cuda);
    }
    if backend_devices
        .iter()
        .any(|device| device.starts_with("ROCm") || device.starts_with("HIP"))
    {
        return Some(BinaryFlavor::Rocm);
    }
    if backend_devices
        .iter()
        .any(|device| device.starts_with("Vulkan"))
    {
        return Some(BinaryFlavor::Vulkan);
    }
    Some(BinaryFlavor::Cpu)
}

fn collect_memory_info() -> Value {
    collect_memory_info_for_platform()
}

#[cfg(target_os = "linux")]
fn collect_memory_info_for_platform() -> Value {
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    json!({
        "total_bytes": parse_linux_meminfo_bytes(&meminfo, "MemTotal"),
        "available_bytes": parse_linux_meminfo_bytes(&meminfo, "MemAvailable"),
    })
}

#[cfg(target_os = "linux")]
fn parse_linux_meminfo_bytes(meminfo: &str, key: &str) -> Option<u64> {
    meminfo.lines().find_map(|line| {
        let (name, rest) = line.split_once(':')?;
        if name != key {
            return None;
        }
        Some(rest.split_whitespace().next()?.parse::<u64>().ok()? * 1024)
    })
}

#[cfg(target_os = "macos")]
fn collect_memory_info_for_platform() -> Value {
    json!({
        "total_bytes": command_u64("sysctl", &["-n", "hw.memsize"]),
        "available_bytes": macos_available_memory_bytes(),
    })
}

#[cfg(target_os = "macos")]
fn macos_available_memory_bytes() -> Option<u64> {
    let output = command_stdout("vm_stat", &[])?;
    parse_macos_vm_stat_available_bytes(&output)
}

#[cfg(target_os = "macos")]
fn parse_macos_vm_stat_available_bytes(output: &str) -> Option<u64> {
    let page_size = output.lines().next().and_then(parse_vm_stat_page_size)?;
    let pages = ["Pages free", "Pages inactive", "Pages speculative"]
        .into_iter()
        .filter_map(|key| parse_vm_stat_pages(output, key))
        .sum::<u64>();
    Some(pages * page_size)
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_page_size(line: &str) -> Option<u64> {
    let marker = "page size of ";
    let rest = &line[line.find(marker)? + marker.len()..];
    rest.chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_pages(output: &str, key: &str) -> Option<u64> {
    output.lines().find_map(|line| {
        let (name, rest) = line.split_once(':')?;
        if name.trim() != key {
            return None;
        }
        rest.chars()
            .filter(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .ok()
    })
}

#[cfg(target_os = "windows")]
fn collect_memory_info_for_platform() -> Value {
    let script = "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress";
    let Some(output) = command_stdout("powershell", &["-NoProfile", "-Command", script]) else {
        return json!({"total_bytes": Value::Null, "available_bytes": Value::Null});
    };
    let Ok(value) = serde_json::from_str::<Value>(&output) else {
        return json!({"total_bytes": Value::Null, "available_bytes": Value::Null});
    };
    json!({
        "total_bytes": value["TotalVisibleMemorySize"].as_u64().map(|kb| kb * 1024),
        "available_bytes": value["FreePhysicalMemory"].as_u64().map(|kb| kb * 1024),
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn collect_memory_info_for_platform() -> Value {
    json!({"total_bytes": Value::Null, "available_bytes": Value::Null})
}

fn collect_cpu_info() -> Value {
    json!({
        "logical_count": std::thread::available_parallelism().ok().map(usize::from),
        "physical_count": physical_cpu_count(),
        "brand": cpu_brand(),
    })
}

fn physical_cpu_count() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        return command_u64("sysctl", &["-n", "hw.physicalcpu"]);
    }
    #[cfg(target_os = "linux")]
    {
        return linux_physical_cpu_count();
    }
    #[cfg(target_os = "windows")]
    {
        let output = command_stdout(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).NumberOfProcessors",
            ],
        )?;
        return output.trim().parse().ok();
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(target_os = "linux")]
fn linux_physical_cpu_count() -> Option<u64> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut physical_ids = std::collections::BTreeSet::new();
    for block in cpuinfo.split("\n\n") {
        let physical_id = block.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == "physical id").then(|| value.trim().to_string())
        });
        if let Some(physical_id) = physical_id {
            physical_ids.insert(physical_id);
        }
    }
    (!physical_ids.is_empty()).then_some(physical_ids.len() as u64)
}

fn cpu_brand() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        return command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
    }
    #[cfg(target_os = "linux")]
    {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
        return cpuinfo.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == "model name").then(|| value.trim().to_string())
        });
    }
    #[cfg(target_os = "windows")]
    {
        return command_stdout(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ],
        )
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(target_os = "macos")]
fn command_u64(command: &str, args: &[&str]) -> Option<u64> {
    command_stdout(command, args)?.trim().parse().ok()
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(command)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())?
}

fn default_output_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(format!("mesh-llm-doctor-{}.zip", utc_filename_timestamp()))
}

fn doctor_readme(max_log_bytes: u64) -> String {
    format!(
        "mesh-llm doctor bundle\n\n\
         Contents:\n\
         - system.json: mesh-llm version, platform, flavor, CPU, and memory summary\n\
         - gpus.json: local GPU facts shown by `mesh-llm gpus --json`\n\
         - plugins.json: installed plugin metadata and resolved runtime plugins\n\
         - config/config.toml: resolved mesh-llm config file when available\n\
         - runtime/instances.json: local runtime process metadata\n\
         - runtime/owner.json and runtime/logs/: logs from the selected process when available\n\
         - api/*.json: local management/API snapshots when the console port is reachable\n\
         - manifest.json: source paths, truncation notes, and warnings\n\n\
         Environment variable values are intentionally not included. \
         Log files are capped at {max_log_bytes} bytes each and contain the tail when truncated.\n"
    )
}

fn utc_timestamp() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn utc_filename_timestamp() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

fn path_modified_unix(path: &Path) -> i64 {
    fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(system_time_unix)
        .unwrap_or(0)
}

fn system_time_unix(time: SystemTime) -> Option<i64> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_auto_prefers_newest_live_instance() {
        let selected = select_runtime_instance(
            &[
                instance_info(20, false, 20),
                instance_info(30, true, 30),
                instance_info(10, true, 10),
            ],
            DoctorLogTarget::Auto,
            None,
        )
        .unwrap();

        assert_eq!(selected.map(|info| info.pid), Some(30));
    }

    #[test]
    fn select_last_uses_newest_instance_even_when_dead() {
        let selected = select_runtime_instance(
            &[instance_info(20, false, 20), instance_info(10, true, 10)],
            DoctorLogTarget::Last,
            None,
        )
        .unwrap();

        assert_eq!(selected.map(|info| info.pid), Some(20));
    }

    #[test]
    fn select_pid_requires_matching_instance() {
        let err = select_runtime_instance(
            &[instance_info(10, true, 10)],
            DoctorLogTarget::Auto,
            Some(11),
        )
        .expect_err("missing pid should fail");

        assert!(err.to_string().contains("pid 11"));
    }

    #[test]
    fn runtime_label_explains_stale_fallback() {
        let previous = instance_info(10, false, 10);

        assert_eq!(
            selected_runtime_label(Some(&previous), std::slice::from_ref(&previous)),
            "no running mesh-llm process found; using previous pid 10"
        );
    }

    #[test]
    fn runtime_label_keeps_explicit_previous_when_live_exists() {
        let previous = instance_info(10, false, 10);
        let running = instance_info(20, true, 20);

        assert_eq!(
            selected_runtime_label(Some(&previous), &[previous.clone(), running]),
            "previous pid 10"
        );
    }

    #[test]
    fn read_runtime_instance_treats_zero_started_at_as_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime_dir = dir.path().join("1234");
        fs::create_dir(&runtime_dir).expect("runtime dir");
        fs::write(
            runtime_dir.join("owner.json"),
            json!({
                "pid": 1234,
                "api_port": 3131,
                "started_at_unix": 0,
                "mesh_llm_binary": "/tmp/mesh-llm",
                "version": "0.0.0-test"
            })
            .to_string(),
        )
        .expect("owner json");

        let info = read_runtime_instance(&runtime_dir).expect("runtime info");

        assert_eq!(info.started_at_unix, None);
        assert!(info.sort_time_unix > 0);
    }

    #[test]
    fn default_output_path_uses_current_directory() {
        let cwd = std::env::current_dir().expect("current directory should be available");
        let path = default_output_path();
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("default output path should have a UTF-8 filename");

        assert!(path.starts_with(cwd));
        assert!(file_name.starts_with("mesh-llm-doctor-"));
        assert!(file_name.ends_with(".zip"));
    }

    #[test]
    fn doctor_readme_documents_optional_config_file() {
        let readme = doctor_readme(1024);

        assert!(readme.contains("config/config.toml"));
        assert!(readme.contains("gpus.json"));
        assert!(!readme.contains("gpus.txt"));
        assert!(readme.contains("Environment variable values are intentionally not included"));
    }

    #[test]
    fn add_existing_file_full_includes_complete_config() {
        let temp = tempfile::tempdir().expect("tempdir");
        let config_path = temp.path().join("config.toml");
        fs::write(&config_path, "[gpu]\nmax_vram = 4\n").expect("write config");
        let zip_path = temp.path().join("doctor.zip");
        let file = File::create(&zip_path).expect("create zip");
        let mut zip = DoctorZip::new(file);
        let mut warnings = Vec::new();

        add_existing_file_full(&mut zip, &config_path, "config/config.toml", &mut warnings)
            .expect("add config");
        zip.finish().expect("finish zip");

        assert!(warnings.is_empty());
        let file = File::open(&zip_path).expect("open zip");
        let mut archive = zip::ZipArchive::new(file).expect("read zip");
        let mut entry = archive.by_name("config/config.toml").expect("config entry");
        let mut contents = String::new();
        entry
            .read_to_string(&mut contents)
            .expect("read config entry");
        assert_eq!(contents, "[gpu]\nmax_vram = 4\n");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parses_macos_available_memory_bytes() {
        let vm_stat = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                       Pages free:                               10.\n\
                       Pages inactive:                           20.\n\
                       Pages speculative:                        30.\n";

        assert_eq!(
            parse_macos_vm_stat_available_bytes(vm_stat),
            Some(60 * 16_384)
        );
    }

    #[test]
    fn parses_running_client_without_runtime_metadata() {
        let known_pids = std::collections::BTreeSet::new();
        let processes = parse_running_mesh_processes(
            "  42 ./target/debug/mesh-llm client --auto\n  43 ./target/debug/mesh-llm doctor\n",
            Path::new("/tmp/runtime"),
            &known_pids,
            99,
            123,
        );

        assert_eq!(processes.len(), 1);
        assert_eq!(processes[0].pid, 42);
        assert!(processes[0].is_live);
        assert!(!processes[0].runtime_metadata);
        assert_eq!(processes[0].runtime_dir, PathBuf::from("/tmp/runtime/42"));
    }

    #[test]
    fn process_discovery_skips_current_and_known_pids() {
        let known_pids = std::collections::BTreeSet::from([42]);
        let processes = parse_running_mesh_processes(
            "  42 ./target/debug/mesh-llm serve --model qwen\n  43 ./target/debug/mesh-llm client --auto\n",
            Path::new("/tmp/runtime"),
            &known_pids,
            43,
            123,
        );

        assert!(processes.is_empty());
    }

    fn instance_info(pid: u32, is_live: bool, sort_time_unix: i64) -> RuntimeInstanceInfo {
        RuntimeInstanceInfo {
            pid,
            api_port: Some(3131),
            version: Some("0.0.0-test".into()),
            started_at_unix: Some(sort_time_unix),
            mesh_llm_binary: Some("/tmp/mesh-llm".into()),
            command: None,
            runtime_dir: PathBuf::from(format!("/tmp/{pid}")),
            runtime_metadata: true,
            is_live,
            sort_time_unix,
        }
    }
}

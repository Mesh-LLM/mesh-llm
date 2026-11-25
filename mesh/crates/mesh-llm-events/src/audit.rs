#![forbid(unsafe_code)]

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::io::{self, Write};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, OnceLock, RwLock};
use tracing::debug;
use uuid::Uuid;

mod sanitization;

#[cfg(windows)]
mod windows_acl;
#[cfg(all(test, windows))]
mod windows_tests;

use sanitization::redact_secrets;
pub use sanitization::{SanitizedAuditDetailJson, SanitizedAuditScalar};

/// Audit log format
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum)]
pub enum AuditLogFormat {
    #[default]
    JsonLines,
}

/// Audit event severity level
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    PartialEq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum AuditLevel {
    #[default]
    Info,
    Warn,
    Error,
    Critical,
}

impl AuditLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            AuditLevel::Info => "info",
            AuditLevel::Warn => "warn",
            AuditLevel::Error => "error",
            AuditLevel::Critical => "critical",
        }
    }
}

/// Categories of audit events for filtering and routing
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditCategory {
    Authentication,
    Authorization,
    Configuration,
    MeshMembership,
    ModelAccess,
    AdminAction,
    System,
}

impl AuditCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            AuditCategory::Authentication => "authentication",
            AuditCategory::Authorization => "authorization",
            AuditCategory::Configuration => "configuration",
            AuditCategory::MeshMembership => "mesh_membership",
            AuditCategory::ModelAccess => "model_access",
            AuditCategory::AdminAction => "admin_action",
            AuditCategory::System => "system",
        }
    }
}

/// Outcome of an audited action
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    Success,
    Failure,
    Denied,
    Error,
}

/// Structured audit event for security-relevant actions
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuditEvent {
    /// Unique event identifier
    pub event_id: Uuid,
    /// Timestamp in RFC3339 format
    pub timestamp: DateTime<Utc>,
    /// Event category for filtering
    pub category: AuditCategory,
    /// Human-readable action description
    pub action: String,
    /// Resource being acted upon (model name, config path, peer ID, etc.)
    pub resource: Option<String>,
    /// Actor identity (owner ID, node ID, user, etc.)
    pub actor: Option<String>,
    /// Outcome of the action
    pub outcome: AuditOutcome,
    /// Severity level
    pub level: AuditLevel,
    /// Correlation ID for request tracing
    pub correlation_id: Option<Uuid>,
    /// Additional structured metadata
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
    /// Error details if outcome is Failure/Error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event with generated ID and current timestamp
    pub fn new(category: AuditCategory, action: impl Into<String>, outcome: AuditOutcome) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            category,
            action: action.into(),
            resource: None,
            actor: None,
            outcome,
            level: AuditLevel::Info,
            correlation_id: None,
            metadata: BTreeMap::new(),
            error: None,
        }
    }

    /// Set the resource being acted upon
    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }

    /// Set the actor identity
    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = Some(actor.into());
        self
    }

    /// Set the severity level
    pub fn with_level(mut self, level: AuditLevel) -> Self {
        self.level = level;
        self
    }

    /// Set correlation ID for request tracing
    pub fn with_correlation_id(mut self, correlation_id: Uuid) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Add metadata key-value pair
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set error details
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Serialize to JSON Lines format (single line JSON)
    pub fn to_json_line(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}

/// Trait for audit log sinks - separate from OutputSink for TUI/logging
pub trait AuditSink: Send + Sync {
    /// Emit an audit event
    fn emit_audit(&self, event: &AuditEvent) -> io::Result<()>;

    /// Flush any buffered events
    fn flush(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + '_>>;

    /// Get the configured log format
    fn format(&self) -> AuditLogFormat {
        AuditLogFormat::JsonLines
    }

    /// Get the minimum level to log
    fn min_level(&self) -> AuditLevel {
        AuditLevel::Info
    }
}

type AuditSinkFuture<'a> = Pin<Box<dyn Future<Output = io::Result<()>> + Send + 'a>>;

/// Global audit sink slot
static AUDIT_SINK: OnceLock<RwLock<Option<Arc<dyn AuditSink>>>> = OnceLock::new();

fn audit_sink_slot() -> &'static RwLock<Option<Arc<dyn AuditSink>>> {
    AUDIT_SINK.get_or_init(|| RwLock::new(None))
}

/// Set the global audit sink
pub fn set_audit_sink(sink: Arc<dyn AuditSink>) {
    if let Ok(mut slot) = audit_sink_slot().write() {
        *slot = Some(sink);
    }
}

/// Clear the global audit sink
pub fn clear_audit_sink() {
    if let Ok(mut slot) = audit_sink_slot().write() {
        *slot = None;
    }
}

/// Get the current audit sink
pub fn audit_sink() -> Option<Arc<dyn AuditSink>> {
    audit_sink_slot()
        .read()
        .ok()
        .and_then(|slot| slot.as_ref().cloned())
}

/// Emit an audit event to the global sink
pub fn emit_audit(event: AuditEvent) -> io::Result<()> {
    // Apply secret redaction before emission
    let redacted = redact_secrets(event);
    match audit_sink() {
        Some(sink) => {
            if redacted.level >= sink.min_level() {
                sink.emit_audit(&redacted)
            } else {
                Ok(())
            }
        }
        None => Ok(()),
    }
}

/// Flush the audit sink
pub async fn flush_audit() -> io::Result<()> {
    match audit_sink() {
        Some(sink) => sink.flush().await,
        None => Ok(()),
    }
}

/// Check if audit logging is enabled (sink configured)
pub fn audit_enabled() -> bool {
    audit_sink().is_some()
}

/// Configuration for file-based audit sink
#[derive(Clone, Debug)]
pub struct FileAuditSinkConfig {
    /// Path to audit log file
    pub path: PathBuf,
    /// Maximum file size before rotation (bytes)
    pub max_file_size: u64,
    /// Maximum number of rotated files to keep
    pub max_files: usize,
    /// Minimum audit level to log
    pub min_level: AuditLevel,
    /// Log format
    pub format: AuditLogFormat,
}

impl Default for FileAuditSinkConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("audit.log"),
            max_file_size: 100 * 1024 * 1024, // 100 MB
            max_files: 10,
            min_level: AuditLevel::Info,
            format: AuditLogFormat::JsonLines,
        }
    }
}

/// File-based audit sink with rotation
pub struct FileAuditSink {
    config: FileAuditSinkConfig,
    lock_path: PathBuf,
    // Keep the handle only for stable identity comparison; writes always reopen.
    private_file: std::sync::Mutex<same_file::Handle>,
    #[cfg(test)]
    fail_next_open: std::sync::atomic::AtomicBool,
}

impl FileAuditSink {
    /// Create a new file audit sink
    pub fn new(config: FileAuditSinkConfig) -> Result<Self> {
        // Ensure parent directory exists
        if config.max_files == 0 || config.max_file_size == 0 {
            return Err(anyhow::anyhow!("audit rotation limits must be non-zero"));
        }
        if let Some(parent) = config
            .path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            ensure_private_directory(parent)?;
        }

        reject_symlink(&config.path)?;
        for index in 1..config.max_files {
            reject_symlink(&rotated_path(&config.path, index))?;
        }
        let name = config.path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "audit path has no file name")
        })?;
        let mut lock_name = std::ffi::OsString::from(".");
        lock_name.push(name);
        lock_name.push(".lock");
        let lock_path = config.path.with_file_name(lock_name);
        let lock = open_private_append_file(&lock_path)?;
        lock.lock()?;
        let private_file = same_file::Handle::from_file(open_private_append_file(&config.path)?)?;
        Ok(Self {
            config,
            lock_path,
            private_file: std::sync::Mutex::new(private_file),
            #[cfg(test)]
            fail_next_open: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn lock_file(&self) -> io::Result<std::fs::File> {
        let lock = reopen_private_append_file(&self.lock_path)?;
        lock.lock()?;
        Ok(lock)
    }

    fn open_active_file(&self) -> io::Result<std::fs::File> {
        #[cfg(test)]
        if self
            .fail_next_open
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "simulated audit file reopen failure",
            ));
        }
        open_guarded_append_file(&self.config.path, true)
    }

    fn harden_replacement(
        &self,
        file: &std::fs::File,
        private_file: &mut same_file::Handle,
    ) -> io::Result<()> {
        let current = same_file::Handle::from_file(file.try_clone()?)?;
        if current != *private_file {
            apply_private_file_permissions(&self.config.path, file)?;
            *private_file = current;
        }
        Ok(())
    }

    fn rotate_if_needed(&self, private_file: &mut same_file::Handle) -> io::Result<std::fs::File> {
        let file = open_guarded_append_file(&self.config.path, true)?;
        self.harden_replacement(&file, private_file)?;
        if file.metadata()?.len() < self.config.max_file_size {
            return Ok(file);
        }

        drop(file);
        for index in 1..self.config.max_files {
            reject_symlink(&rotated_path(&self.config.path, index))?;
        }
        if self.config.max_files > 1 {
            let oldest = rotated_path(&self.config.path, self.config.max_files - 1);
            if oldest.try_exists()? {
                discard_rotated_file(&oldest)?;
            }
            for index in (1..self.config.max_files - 1).rev() {
                let from = rotated_path(&self.config.path, index);
                if from.try_exists()? {
                    std::fs::rename(&from, rotated_path(&self.config.path, index + 1))?;
                }
            }
            std::fs::rename(&self.config.path, rotated_path(&self.config.path, 1))?;
        } else {
            discard_rotated_file(&self.config.path)?;
        }
        let file = self.open_active_file()?;
        self.harden_replacement(&file, private_file)?;
        debug!("Rotated audit log file");
        Ok(file)
    }
}

impl AuditSink for FileAuditSink {
    fn emit_audit(&self, event: &AuditEvent) -> io::Result<()> {
        let line = event
            .to_json_line()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let _lock = self.lock_file()?;
        let mut private_file = self
            .private_file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut file = self.rotate_if_needed(&mut private_file)?;
        writeln!(file, "{}", line)?;
        file.flush()?;

        Ok(())
    }

    fn flush(&self) -> AuditSinkFuture<'_> {
        Box::pin(async {
            let _lock = self.lock_file()?;
            Ok(())
        })
    }

    fn format(&self) -> AuditLogFormat {
        self.config.format
    }

    fn min_level(&self) -> AuditLevel {
        self.config.min_level
    }
}

fn rotated_path(path: &std::path::Path, index: usize) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(format!(".{index}"));
    PathBuf::from(value)
}

#[cfg(windows)]
fn discard_rotated_file(path: &std::path::Path) -> io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || is_link_or_reparse_point(&metadata) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected a regular audit file for rotation",
        ));
    }
    // Other sinks can still hold this generation's identity handle. Give it a
    // unique name before requesting removal so classic Windows deletion
    // leaves the active and numbered names immediately reusable.
    let retired = path.with_file_name(format!(".mesh-audit-{}.tmp", Uuid::new_v4()));
    std::fs::rename(path, &retired)?;
    std::fs::remove_file(&retired).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to remove retired audit file {}: {error}",
                retired.display()
            ),
        )
    })
}

#[cfg(not(windows))]
fn discard_rotated_file(path: &std::path::Path) -> io::Result<()> {
    std::fs::remove_file(path)
}

fn reject_symlink(path: &std::path::Path) -> io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if is_link_or_reparse_point(&metadata) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "audit path must not be a symbolic link or reparse point",
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn is_link_or_reparse_point(metadata: &std::fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    false
}

fn ensure_private_directory(path: &std::path::Path) -> io::Result<()> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        if current.as_os_str().is_empty() {
            continue;
        }
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if is_link_or_reparse_point(&metadata) => {
                if is_trusted_platform_directory_link(&current) {
                    continue;
                }
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "audit directory path must not traverse a link or reparse point",
                ));
            }
            Ok(metadata) if !metadata.is_dir() => {
                return Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    "audit directory path contains a non-directory",
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                std::fs::create_dir(&current)?;
                apply_private_directory_permissions(&current)?;
            }
            Err(error) => return Err(error),
        }
    }
    reject_symlink(path)?;
    apply_private_directory_permissions(path)
}

fn is_trusted_platform_directory_link(_path: &std::path::Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        matches!(_path.to_str(), Some("/var" | "/tmp" | "/etc"))
    }
    #[cfg(not(target_os = "macos"))]
    false
}

fn apply_private_directory_permissions(path: &std::path::Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(windows)]
    windows_acl::apply_owner_only_acl(path, true)?;
    Ok(())
}

fn open_private_append_file(path: &std::path::Path) -> io::Result<std::fs::File> {
    let file = open_guarded_append_file(path, true)?;
    apply_private_file_permissions(path, &file)?;
    Ok(file)
}

fn apply_private_file_permissions(_path: &std::path::Path, file: &std::fs::File) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        const FILE_SHARE_READ_WRITE: u32 = 0x0000_0003;
        // Keep this exact path from being replaced while the path-based ACL
        // utility runs; ordinary write/identity handles still share deletion.
        let guard = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ_WRITE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(_path)?;
        if is_link_or_reparse_point(&guard.metadata()?)
            || same_file::Handle::from_file(guard.try_clone()?)?
                != same_file::Handle::from_file(file.try_clone()?)?
        {
            return Err(io::Error::other(
                "audit path changed before applying private permissions",
            ));
        }
        windows_acl::apply_owner_only_acl(_path, false)?;
    }
    Ok(())
}

fn reopen_private_append_file(path: &std::path::Path) -> io::Result<std::fs::File> {
    match open_guarded_append_file(path, false) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => open_private_append_file(path),
        result => result,
    }
}

fn open_guarded_append_file(path: &std::path::Path, create: bool) -> io::Result<std::fs::File> {
    reject_symlink(path)?;
    let mut options = std::fs::OpenOptions::new();
    // Read access also permits locking the sibling file on Windows, where an
    // append-only handle cannot be locked.
    options.create(create).read(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(path)?;
    if is_link_or_reparse_point(&file.metadata()?) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "audit file must not be a link or reparse point",
        ));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(index: usize) -> AuditEvent {
        AuditEvent::new(
            AuditCategory::System,
            format!("event-{index}"),
            AuditOutcome::Success,
        )
    }

    #[test]
    fn independent_sinks_rotate_the_current_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("audit.log");
        let config = FileAuditSinkConfig {
            path: path.clone(),
            max_file_size: 1,
            max_files: 2,
            ..FileAuditSinkConfig::default()
        };
        let serving = FileAuditSink::new(config.clone()).unwrap();
        let auth = FileAuditSink::new(config).unwrap();

        auth.emit_audit(&event(1)).unwrap();
        auth.emit_audit(&event(2)).unwrap();
        serving.emit_audit(&event(3)).unwrap();
        auth.emit_audit(&event(4)).unwrap();

        let active = std::fs::read_to_string(&path).unwrap();
        let previous = std::fs::read_to_string(rotated_path(&path, 1)).unwrap();
        assert!(active.contains("event-4"), "{active}");
        assert!(previous.contains("event-3"), "{previous}");
    }

    #[test]
    fn separate_processes_share_audit_rotation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("audit.log");
        let serving = FileAuditSink::new(FileAuditSinkConfig {
            path: path.clone(),
            max_file_size: 1,
            max_files: 32,
            ..FileAuditSinkConfig::default()
        })
        .unwrap();
        let children: Vec<_> = (1..=2)
            .map(|writer| {
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "audit::tests::audit_writer_process", "--ignored"])
                    .env("MESH_LLM_TEST_AUDIT_PATH", &path)
                    .env("MESH_LLM_TEST_AUDIT_WRITER", writer.to_string())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect();
        for child in children {
            let output = child.wait_with_output().unwrap();
            assert!(output.status.success(), "{output:?}");
        }
        serving.emit_audit(&event(99)).unwrap();

        let mut actions = std::collections::BTreeSet::new();
        for entry in std::fs::read_dir(directory.path()).unwrap() {
            let path = entry.unwrap().path();
            if path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("audit.log")
            {
                for line in std::fs::read_to_string(path).unwrap().lines() {
                    let record: AuditEvent = serde_json::from_str(line).unwrap();
                    assert!(actions.insert(record.action));
                }
            }
        }
        let expected: std::collections::BTreeSet<_> = (10..30)
            .chain(std::iter::once(99))
            .map(|index| format!("event-{index}"))
            .collect();
        assert_eq!(actions, expected);
    }

    #[test]
    #[ignore = "invoked with controlled inputs by separate_processes_share_audit_rotation"]
    fn audit_writer_process() {
        let path = std::env::var_os("MESH_LLM_TEST_AUDIT_PATH").unwrap().into();
        let writer: usize = std::env::var("MESH_LLM_TEST_AUDIT_WRITER")
            .unwrap()
            .parse()
            .unwrap();
        let sink = FileAuditSink::new(FileAuditSinkConfig {
            path,
            max_file_size: 1,
            max_files: 32,
            ..FileAuditSinkConfig::default()
        })
        .unwrap();
        for index in writer * 10..writer * 10 + 10 {
            sink.emit_audit(&event(index)).unwrap();
        }
    }

    #[test]
    fn concurrent_rotation_preserves_every_successful_write() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("audit.log");
        let sink = Arc::new(
            FileAuditSink::new(FileAuditSinkConfig {
                path: path.clone(),
                max_file_size: 512,
                max_files: 100,
                ..FileAuditSinkConfig::default()
            })
            .unwrap(),
        );
        let mut threads = Vec::new();
        for thread in 0..8 {
            let sink = Arc::clone(&sink);
            threads.push(std::thread::spawn(move || {
                for row in 0..20 {
                    sink.emit_audit(&event(thread * 20 + row)).unwrap();
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }

        let mut rows = 0;
        for entry in std::fs::read_dir(directory.path()).unwrap() {
            let path = entry.unwrap().path();
            if path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("audit.log")
            {
                rows += std::fs::read_to_string(path).unwrap().lines().count();
            }
        }
        assert_eq!(rows, 160);
    }

    #[test]
    fn rotation_reopen_failure_discards_stale_descriptor_and_recovers() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("audit.log");
        let sink = FileAuditSink::new(FileAuditSinkConfig {
            path: path.clone(),
            max_file_size: 1,
            max_files: 2,
            ..FileAuditSinkConfig::default()
        })
        .unwrap();

        sink.emit_audit(&event(1)).unwrap();
        sink.fail_next_open
            .store(true, std::sync::atomic::Ordering::SeqCst);

        assert!(sink.emit_audit(&event(2)).is_err());
        sink.emit_audit(&event(3)).unwrap();

        let active = std::fs::read_to_string(&path).unwrap();
        let rotated = std::fs::read_to_string(rotated_path(&path, 1)).unwrap();
        assert!(active.contains("event-3"));
        assert!(!rotated.contains("event-3"));
    }

    #[cfg(unix)]
    #[test]
    fn externally_replaced_audit_file_is_private_before_writing() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("audit.log");
        let old_path = directory.path().join("external-rotation.log");
        let sink = FileAuditSink::new(FileAuditSinkConfig {
            path: path.clone(),
            ..FileAuditSinkConfig::default()
        })
        .unwrap();
        sink.emit_audit(&event(1)).unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::rename(&path, &old_path).unwrap();
        std::fs::write(&path, "").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        sink.emit_audit(&event(2)).unwrap();

        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(std::fs::read_to_string(&path).unwrap().contains("event-2"));
        assert!(
            !std::fs::read_to_string(old_path)
                .unwrap()
                .contains("event-2")
        );
    }

    #[cfg(unix)]
    #[test]
    fn audit_files_are_private_and_symlinks_are_rejected() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir().unwrap();
        let private = directory.path().join("private");
        let path = private.join("audit.log");
        FileAuditSink::new(FileAuditSinkConfig {
            path: path.clone(),
            ..FileAuditSinkConfig::default()
        })
        .unwrap();
        assert_eq!(
            std::fs::metadata(&private).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            std::fs::metadata(private.join(".audit.log.lock"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );

        std::fs::remove_file(&path).unwrap();
        let target = directory.path().join("target");
        std::fs::write(&target, "do not overwrite").unwrap();
        symlink(&target, &path).unwrap();
        assert!(
            FileAuditSink::new(FileAuditSinkConfig {
                path,
                ..FileAuditSinkConfig::default()
            })
            .is_err()
        );

        let rotated_base = private.join("rotating.log");
        symlink(&target, rotated_path(&rotated_base, 1)).unwrap();
        assert!(
            FileAuditSink::new(FileAuditSinkConfig {
                path: rotated_base,
                max_files: 2,
                ..FileAuditSinkConfig::default()
            })
            .is_err()
        );

        let locked_path = private.join("locked.log");
        symlink(&target, private.join(".locked.log.lock")).unwrap();
        assert!(
            FileAuditSink::new(FileAuditSinkConfig {
                path: locked_path,
                ..FileAuditSinkConfig::default()
            })
            .is_err()
        );
        assert_eq!(std::fs::read_to_string(target).unwrap(), "do not overwrite");
    }
}

/// Helper to create audit events for common scenarios
pub mod audit_events {
    use super::*;

    /// Authentication attempt
    pub fn auth_attempt(actor: Option<String>, success: bool, method: &str) -> AuditEvent {
        AuditEvent::new(
            AuditCategory::Authentication,
            format!("auth_{}", if success { "success" } else { "failure" }),
            if success {
                AuditOutcome::Success
            } else {
                AuditOutcome::Failure
            },
        )
        .with_actor(actor.unwrap_or_else(|| "unknown".to_string()))
        .with_metadata("method".to_string(), Value::String(method.to_string()))
        .with_level(if success {
            AuditLevel::Info
        } else {
            AuditLevel::Warn
        })
    }

    /// Authorization check
    pub fn authz_check(
        actor: Option<String>,
        resource: &str,
        action: &str,
        allowed: bool,
    ) -> AuditEvent {
        AuditEvent::new(
            AuditCategory::Authorization,
            format!("authz_{}", if allowed { "allow" } else { "deny" }),
            if allowed {
                AuditOutcome::Success
            } else {
                AuditOutcome::Denied
            },
        )
        .with_actor(actor.unwrap_or_else(|| "unknown".to_string()))
        .with_resource(resource.to_string())
        .with_metadata("action".to_string(), Value::String(action.to_string()))
        .with_level(if allowed {
            AuditLevel::Info
        } else {
            AuditLevel::Warn
        })
    }

    /// Configuration change
    pub fn config_change(
        actor: Option<String>,
        config_path: &str,
        old_value: Option<Value>,
        new_value: Option<Value>,
    ) -> AuditEvent {
        let mut event = AuditEvent::new(
            AuditCategory::Configuration,
            "config_change".to_string(),
            AuditOutcome::Success,
        )
        .with_actor(actor.unwrap_or_else(|| "unknown".to_string()))
        .with_resource(config_path.to_string())
        .with_level(AuditLevel::Info);

        if let Some(v) = old_value {
            event = event.with_metadata("old_value".to_string(), v);
        }
        if let Some(v) = new_value {
            event = event.with_metadata("new_value".to_string(), v);
        }
        event
    }

    /// Mesh membership change (peer join/leave)
    pub fn mesh_membership(actor: Option<String>, peer_id: &str, joined: bool) -> AuditEvent {
        AuditEvent::new(
            AuditCategory::MeshMembership,
            if joined { "peer_joined" } else { "peer_left" },
            AuditOutcome::Success,
        )
        .with_actor(actor.unwrap_or_else(|| "system".to_string()))
        .with_resource(peer_id.to_string())
        .with_level(AuditLevel::Info)
    }

    /// Model access (load/unload/route)
    pub fn model_access(
        actor: Option<String>,
        model: &str,
        action: &str,
        success: bool,
    ) -> AuditEvent {
        AuditEvent::new(
            AuditCategory::ModelAccess,
            format!("model_{}", action),
            if success {
                AuditOutcome::Success
            } else {
                AuditOutcome::Failure
            },
        )
        .with_actor(actor.unwrap_or_else(|| "unknown".to_string()))
        .with_resource(model.to_string())
        .with_level(if success {
            AuditLevel::Info
        } else {
            AuditLevel::Warn
        })
    }

    /// Administrative action
    pub fn admin_action(
        actor: Option<String>,
        action: &str,
        resource: Option<&str>,
        success: bool,
        error: Option<String>,
    ) -> AuditEvent {
        let mut event = AuditEvent::new(
            AuditCategory::AdminAction,
            action.to_string(),
            if success {
                AuditOutcome::Success
            } else {
                AuditOutcome::Failure
            },
        )
        .with_actor(actor.unwrap_or_else(|| "unknown".to_string()))
        .with_level(if success {
            AuditLevel::Info
        } else {
            AuditLevel::Error
        });

        if let Some(r) = resource {
            event = event.with_resource(r.to_string());
        }
        if let Some(e) = error {
            event = event.with_error(e);
        }
        event
    }

    /// System event (startup, shutdown, error)
    pub fn system_event(action: &str, outcome: AuditOutcome, error: Option<String>) -> AuditEvent {
        let mut event = AuditEvent::new(AuditCategory::System, action.to_string(), outcome)
            .with_actor("system".to_string())
            .with_level(match outcome {
                AuditOutcome::Success => AuditLevel::Info,
                AuditOutcome::Failure | AuditOutcome::Error => AuditLevel::Error,
                AuditOutcome::Denied => AuditLevel::Warn,
            });

        if let Some(e) = error {
            event = event.with_error(e);
        }
        event
    }
}

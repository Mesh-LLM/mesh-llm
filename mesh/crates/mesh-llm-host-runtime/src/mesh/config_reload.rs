//! Settings a running node keeps current from its config file.
//!
//! `mesh-llm` commands, a hand edit or a config apply can change the file
//! while the node runs. The node notices the change and reads the setting
//! again, so no restart is needed. When the file is missing, cannot be read or
//! does not validate, a setting that only grants is empty until the file is
//! back, and one that can also take away keeps its last value (see
//! [`WhenUnloadable`]). The check stops once the node drops the setting.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, Weak};
use std::time::{Duration, SystemTime};

/// How often a running node checks its config file for a change.
const RELOAD_INTERVAL: Duration = Duration::from_secs(2);

/// What a setting is while its config file is missing or does not load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WhenUnloadable {
    /// Its empty value, for a setting that only grants: a missing or broken
    /// file grants nothing.
    Empty,
    /// Its last value that loaded, for a setting that can also take away: a
    /// missing or broken file (an editor saving it, say) must not lift a
    /// restriction. Removing one takes an edit of the file, or a restart.
    KeepLast,
}

/// Which setting to read and how it fails.
pub(crate) struct SettingSpec<T> {
    /// What log lines call the setting.
    pub(crate) name: &'static str,
    /// Reads the setting from a config that validated.
    pub(crate) read: fn(&mesh_llm_config::MeshConfig) -> T,
    pub(crate) when_unloadable: WhenUnloadable,
}

impl<T> Clone for SettingSpec<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SettingSpec<T> {}

/// One setting read from the config file, kept current while the node runs.
pub(crate) struct ReloadedSetting<T> {
    value: Arc<RwLock<Arc<T>>>,
    /// The config file the setting is kept current from, when there is one.
    file: Option<Arc<PathBuf>>,
    /// Held across each read of the file and the swap that follows, so an
    /// older read cannot replace a newer one.
    reload: Arc<std::sync::Mutex<()>>,
    spec: SettingSpec<T>,
}

impl<T> Clone for ReloadedSetting<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            file: self.file.clone(),
            reload: self.reload.clone(),
            spec: self.spec,
        }
    }
}

impl<T> std::fmt::Debug for ReloadedSetting<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReloadedSetting")
            .field("name", &self.spec.name)
            .field("file", &self.file)
            .finish_non_exhaustive()
    }
}

impl<T: Default + PartialEq + Send + Sync + 'static> ReloadedSetting<T> {
    /// A fixed `value` that no file change replaces.
    pub(crate) fn fixed(spec: SettingSpec<T>, value: T) -> Self {
        Self {
            value: Arc::new(RwLock::new(Arc::new(value))),
            file: None,
            reload: Arc::default(),
            spec,
        }
    }

    /// The setting in `config`, kept current from the config file it came
    /// from when `reload` (a node with its own config file, not a scoped
    /// shared-serving membership).
    pub(crate) fn kept_current(
        spec: SettingSpec<T>,
        config: &mesh_llm_config::MeshConfig,
        config_path: Option<&Path>,
        reload: bool,
    ) -> Self {
        let mut setting = Self::fixed(spec, (spec.read)(config));
        if reload && let Ok(path) = crate::plugin::config_path(config_path) {
            setting.file = Some(Arc::new(path.clone()));
            spawn_reload(setting.clone(), path);
        }
        setting
    }

    /// Read the setting from the config file now, rather than at the next
    /// check: a config apply does this before it reports the new value as
    /// live.
    pub(crate) fn reload_now(&self) {
        if let Some(path) = &self.file {
            self.reload_from(path, true);
        }
    }

    pub(crate) fn current(&self) -> Arc<T> {
        self.value
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Read the setting from `path` and install it. `warn` logs a file that
    /// does not load as a warning rather than at debug level.
    fn reload_from(&self, path: &Path, warn: bool) -> Reload {
        let _reload = self
            .reload
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let read = self.read(path, warn);
        Reload {
            changed: read.value.is_some_and(|value| self.replace(value)),
            loaded: read.loaded,
        }
    }

    /// The setting in the config file at `path`; when there is no file or it
    /// does not load, empty or `None` (keep the current value) as the
    /// setting's [`WhenUnloadable`] says.
    #[cfg(test)]
    fn load(&self, path: &Path) -> Option<T> {
        self.read(path, true).value
    }

    fn read(&self, path: &Path, warn: bool) -> Read<T> {
        let exists = match path.try_exists() {
            Ok(exists) => exists,
            // An unreadable directory is not a missing file.
            Err(error) => {
                return Read {
                    value: self.unloadable(
                        path,
                        &format!("config file cannot be checked: {error}"),
                        warn,
                    ),
                    loaded: false,
                };
            }
        };
        if !exists {
            // Running without a config file is normal: no warning.
            tracing::debug!(
                path = %path.display(),
                setting = self.spec.name,
                "no config file"
            );
            return Read {
                value: self.when_unloadable(),
                loaded: true,
            };
        }
        match mesh_llm_config::load_config(Some(path)) {
            Ok(config) => Read {
                value: Some((self.spec.read)(&config)),
                loaded: true,
            },
            Err(error) => Read {
                value: self.unloadable(path, &format!("config file does not load: {error}"), warn),
                loaded: false,
            },
        }
    }

    /// The setting while the file at `path` does not load, for `reason`.
    fn unloadable(&self, path: &Path, reason: &str, warn: bool) -> Option<T> {
        let state = if self.when_unloadable().is_none() {
            "unchanged"
        } else {
            "empty"
        };
        if warn {
            tracing::warn!(
                path = %path.display(),
                setting = self.spec.name,
                "{reason}; the setting is {state} until it is fixed",
            );
        } else {
            tracing::debug!(
                path = %path.display(),
                setting = self.spec.name,
                "{reason}; the setting is {state} until it is fixed",
            );
        }
        self.when_unloadable()
    }

    /// Empty, or `None` to keep the current value.
    fn when_unloadable(&self) -> Option<T> {
        (self.spec.when_unloadable == WhenUnloadable::Empty).then(T::default)
    }

    /// Replace the value; true when it changed.
    fn replace(&self, value: T) -> bool {
        let mut current = self
            .value
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if **current == value {
            return false;
        }
        *current = Arc::new(value);
        true
    }
}

/// One read of the config file for a setting.
struct Read<T> {
    /// The value to install, or `None` to keep the current one.
    value: Option<T>,
    /// Whether the file loaded (or is absent, which is as stable).
    loaded: bool,
}

/// What one reload did.
struct Reload {
    changed: bool,
    loaded: bool,
}

/// What identifies one version of the file: its modification time and size.
type FileVersion = Option<(SystemTime, u64)>;

fn file_version(path: &Path) -> FileVersion {
    let metadata = std::fs::metadata(path).ok()?;
    Some((metadata.modified().ok()?, metadata.len()))
}

/// What the background check has read from the config file.
#[derive(Default)]
struct FileWatch {
    /// The version that last loaded: an unchanged file is not read again.
    /// Nothing yet, so the first check reads the file, even a missing one.
    loaded: Option<FileVersion>,
    /// The version that last failed to load. It is read again at every
    /// check, since a fix such as a permission change can leave its
    /// modification time and size unchanged, but warned about only once.
    failed: Option<FileVersion>,
}

impl FileWatch {
    /// Read the setting again unless the file is the version that last
    /// loaded; true when the value changed.
    fn check<T: Default + PartialEq + Send + Sync + 'static>(
        &mut self,
        setting: &ReloadedSetting<T>,
        path: &Path,
    ) -> bool {
        let version = file_version(path);
        if self.loaded == Some(version) {
            return false;
        }
        let reload = setting.reload_from(path, self.failed != Some(version));
        if reload.loaded {
            (self.loaded, self.failed) = (Some(version), None);
        } else {
            (self.loaded, self.failed) = (None, Some(version));
        }
        reload.changed
    }
}

/// Read the setting again whenever the config file at `path` changes, until
/// every copy of `setting` is dropped.
fn spawn_reload<T: Default + PartialEq + Send + Sync + 'static>(
    setting: ReloadedSetting<T>,
    path: PathBuf,
) {
    let ReloadedSetting {
        value,
        reload,
        spec,
        ..
    } = setting;
    let value: Weak<RwLock<Arc<T>>> = Arc::downgrade(&value);
    tokio::spawn(async move {
        // An edit or removal made between the node's first read and this
        // task starting counts: the first check reads the file again.
        let mut watch = FileWatch::default();
        loop {
            tokio::time::sleep(RELOAD_INTERVAL).await;
            let Some(value) = value.upgrade() else {
                return;
            };
            let setting = ReloadedSetting {
                value,
                file: None,
                reload: reload.clone(),
                spec,
            };
            let reload_path = path.clone();
            let mut checking = std::mem::take(&mut watch);
            let Ok((checked, changed)) = tokio::task::spawn_blocking(move || {
                let changed = checking.check(&setting, &reload_path);
                (checking, changed)
            })
            .await
            else {
                continue;
            };
            watch = checked;
            if changed {
                tracing::info!(setting = spec.name, "setting reloaded from the config file");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_nodes(config: &mesh_llm_config::MeshConfig) -> Vec<String> {
        config.private.nodes.clone()
    }

    const NODES: SettingSpec<Vec<String>> = SettingSpec {
        name: "nodes",
        read: read_nodes,
        when_unloadable: WhenUnloadable::Empty,
    };

    fn node(seed: u8) -> String {
        hex::encode([seed; 32])
    }

    fn write_config(path: &Path, nodes: &[String]) {
        let nodes: Vec<String> = nodes.iter().map(|id| format!("\"{id}\"")).collect();
        std::fs::write(path, format!("[private]\nnodes = [{}]\n", nodes.join(", "))).unwrap();
    }

    fn watched(path: &Path) -> ReloadedSetting<Vec<String>> {
        let config = mesh_llm_config::load_config(Some(path)).unwrap();
        ReloadedSetting::kept_current(NODES, &config, Some(path), true)
    }

    async fn wait_for(setting: &ReloadedSetting<Vec<String>>, expected: &[String]) {
        for _ in 0..300 {
            if *setting.current() == expected {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert_eq!(*setting.current(), expected);
    }

    #[test]
    fn a_missing_or_broken_file_follows_the_setting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let granting = ReloadedSetting::fixed(NODES, vec![node(1)]);
        let restricting = ReloadedSetting::fixed(
            SettingSpec {
                when_unloadable: WhenUnloadable::KeepLast,
                ..NODES
            },
            vec![node(1)],
        );
        assert_eq!(granting.load(&path), Some(vec![]));
        assert_eq!(restricting.load(&path), None);

        std::fs::write(&path, "[private]\nnodes = [\"studio\"]\n").unwrap();
        assert_eq!(granting.load(&path), Some(vec![]));
        assert_eq!(restricting.load(&path), None);
        assert!(!restricting.reload_from(&path, true).changed);
        assert_eq!(*restricting.current(), [node(1)]);
    }

    #[tokio::test]
    async fn a_running_node_picks_up_an_edit_and_a_removal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_config(&path, &[node(1)]);
        let setting = watched(&path);

        write_config(&path, &[node(1), node(2), node(3)]);
        wait_for(&setting, &[node(1), node(2), node(3)]).await;

        std::fs::remove_file(&path).unwrap();
        wait_for(&setting, &[]).await;
    }

    #[tokio::test]
    async fn a_file_removed_before_the_first_check_empties_the_setting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_config(&path, &[node(1)]);
        let setting = watched(&path);
        std::fs::remove_file(&path).unwrap();
        wait_for(&setting, &[]).await;
    }

    #[tokio::test]
    async fn the_check_does_not_keep_a_dropped_setting_alive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_config(&path, &[node(1)]);
        let setting = watched(&path);
        let value = Arc::downgrade(&setting.value);
        drop(setting);
        assert!(value.upgrade().is_none());
    }

    #[tokio::test]
    async fn reload_now_reads_the_file_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_config(&path, &[node(1), node(2)]);
        let setting = watched(&path);
        assert_eq!(*setting.current(), [node(1), node(2)]);

        write_config(&path, &[node(2)]);
        setting.reload_now();
        assert_eq!(*setting.current(), [node(2)]);

        let config = mesh_llm_config::load_config(Some(&path)).unwrap();
        let unwatched = ReloadedSetting::kept_current(NODES, &config, Some(&path), false);
        write_config(&path, &[node(3)]);
        unwatched.reload_now();
        assert_eq!(*unwatched.current(), [node(2)]);
    }

    #[test]
    fn an_older_background_read_cannot_win_over_reload_now() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_config(&path, &[node(1), node(2)]);
        let mut setting = ReloadedSetting::fixed(NODES, vec![node(1), node(2)]);
        setting.file = Some(Arc::new(path.clone()));

        // The background check has read the old value and not yet swapped it in.
        let background = setting.reload.lock().unwrap();
        let old_read = setting.load(&path);

        // An apply removes node 1 and reloads before reporting it live.
        write_config(&path, &[node(2)]);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let applying = setting.clone();
        let apply = std::thread::spawn(move || {
            applying.reload_now();
            done_tx.send(()).unwrap();
        });
        assert!(
            done_rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "reload_now must wait for the read in progress"
        );

        setting.replace(old_read.unwrap());
        drop(background);
        apply.join().unwrap();
        assert_eq!(*setting.current(), [node(2)]);
    }

    /// A file that failed to load is read again even when the fix leaves
    /// its modification time and size unchanged.
    #[test]
    fn a_file_that_failed_to_load_is_read_again_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let setting = ReloadedSetting::fixed(
            SettingSpec {
                when_unloadable: WhenUnloadable::KeepLast,
                ..NODES
            },
            vec![],
        );
        let set_mtime = |seconds: u64| {
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(seconds))
                .unwrap();
        };
        let mut watch = FileWatch::default();
        write_config(&path, &[node(1)]);
        set_mtime(1_000);
        assert!(watch.check(&setting, &path));

        // The same length, but not a node id.
        let broken = std::fs::read_to_string(&path)
            .unwrap()
            .replacen('1', "g", 1);
        std::fs::write(&path, broken).unwrap();
        set_mtime(2_000);
        assert!(!watch.check(&setting, &path));
        assert_eq!(*setting.current(), [node(1)]);

        write_config(&path, &[node(2)]);
        set_mtime(2_000);
        assert!(watch.check(&setting, &path));
        assert_eq!(*setting.current(), [node(2)]);
        assert!(!watch.check(&setting, &path));
    }

    /// A config file that cannot be checked is a failed load, not an absent
    /// file: it keeps a setting that can take away, and is read again.
    /// (Windows reports a path through a file as not found.)
    #[cfg(unix)]
    #[test]
    fn an_uncheckable_file_is_not_a_missing_one() {
        let dir = tempfile::tempdir().unwrap();
        let not_a_directory = dir.path().join("file");
        std::fs::write(&not_a_directory, "").unwrap();
        let path = not_a_directory.join("config.toml");
        let restricting = ReloadedSetting::fixed(
            SettingSpec {
                when_unloadable: WhenUnloadable::KeepLast,
                ..NODES
            },
            vec![node(1)],
        );
        let read = restricting.read(&path, true);
        assert!(!read.loaded);
        assert_eq!(read.value, None);
        assert!(
            restricting
                .read(&dir.path().join("missing.toml"), true)
                .loaded
        );
    }
}

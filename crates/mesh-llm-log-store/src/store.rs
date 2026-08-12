//! LogStore owning the SQLite connection and lifecycle.

use crate::artifact_privacy::{
    ArtifactPrivacy, PlatformArtifactPrivacy, create_private_directory_tree,
};
use crate::error::LogStoreError;
use rusqlite::{Connection, Transaction};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Clock abstraction for deterministic timestamps in tests.
pub trait Clock: Send + Sync {
    fn now(&self) -> String;
}

#[derive(Debug, Clone)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> String {
        let dt = chrono::Utc::now();
        format!("{}", dt.format("%Y-%m-%dT%H:%M:%S%.3fZ"))
    }
}

pub struct LogStore {
    conn: Mutex<Connection>,
    clock: std::sync::Arc<dyn Clock>,
    #[cfg_attr(not(test), allow(unused))]
    db_path: PathBuf,
}

/// Outcome of the bounded SQLite maintenance run performed after retention
/// deletes. It deliberately distinguishes logical cleanup from physical file
/// reclamation: legacy databases with `auto_vacuum=NONE` remain correct but do
/// not promise to shrink without an explicit offline `VACUUM`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SQLiteSpaceMaintenance {
    pub optimized: bool,
    /// Actual reduction in SQLite's page count during the bounded incremental
    /// vacuum. Zero means that no pages were physically reclaimed.
    pub incremental_vacuum_pages: u32,
}

impl LogStore {
    pub fn open(
        root_path: impl AsRef<Path>,
        clock: std::sync::Arc<dyn Clock>,
    ) -> Result<Self, LogStoreError> {
        let root = prepare_private_store_root(root_path.as_ref())?;

        let db_path = root.join("log_store.db");
        reject_link_if_present(&db_path)?;
        let new_database = !db_path.exists();
        let conn = Connection::open(&db_path).map_err(|e| {
            LogStoreError::IoError(std::io::Error::other(format!("sqlite open: {}", e)))
        })?;
        prepare_private_database_files(&db_path)?;

        if new_database {
            // This only takes effect before schema allocation. Existing stores
            // retain their current mode; changing it would require a blocking
            // full VACUUM and is not safe in normal request handling.
            conn.execute_batch("PRAGMA auto_vacuum = INCREMENTAL;")
                .map_err(LogStoreError::Sqlite)?;
        }
        let pragmas = "
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 30000;
        ";
        conn.execute_batch(pragmas).map_err(LogStoreError::Sqlite)?;

        crate::migrations::apply_migrations(&conn)
            .map_err(|e| LogStoreError::MigrationFailed(e.to_string()))?;
        prepare_private_database_files(&db_path)?;

        Ok(Self {
            conn: Mutex::new(conn),
            clock,
            db_path,
        })
    }

    pub fn reopen_at(
        root_path: impl AsRef<Path>,
        clock: std::sync::Arc<dyn Clock>,
    ) -> Result<Self, LogStoreError> {
        Self::open(root_path, clock)
    }

    pub fn txn<T>(
        &self,
        f: impl FnOnce(&Transaction) -> Result<T, LogStoreError>,
    ) -> Result<T, LogStoreError> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|_| LogStoreError::Sqlite(rusqlite::Error::ExecuteReturnedResults))?;

        let tx = conn.transaction().map_err(LogStoreError::Sqlite)?;
        let result = f(&tx);
        if result.is_ok() {
            tx.commit().map_err(LogStoreError::Sqlite)?;
            self.prepare_private_database_files()?;
        }
        result
    }

    pub fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("connection mutex poisoned")
    }

    pub fn now(&self) -> String {
        self.clock.now()
    }

    /// Run safe, bounded SQLite maintenance after logical retention cleanup.
    ///
    /// `PRAGMA optimize` is advisory. Incremental vacuum is used only for new
    /// stores created in incremental-auto-vacuum mode and is capped at 64
    /// pages. This method never invokes full `VACUUM`.
    pub fn maintain_space_after_cleanup(&self) -> Result<SQLiteSpaceMaintenance, LogStoreError> {
        let conn = self.conn();
        conn.execute_batch("PRAGMA analysis_limit = 400; PRAGMA optimize;")
            .map_err(LogStoreError::Sqlite)?;
        let auto_vacuum: i64 = conn
            .query_row("PRAGMA auto_vacuum", [], |row| row.get(0))
            .map_err(LogStoreError::Sqlite)?;
        let incremental_vacuum_pages = if auto_vacuum == 2 {
            let page_count_before: i64 = conn
                .query_row("PRAGMA page_count", [], |row| row.get(0))
                .map_err(LogStoreError::Sqlite)?;
            conn.execute_batch("PRAGMA incremental_vacuum(64);")
                .map_err(LogStoreError::Sqlite)?;
            let page_count_after: i64 = conn
                .query_row("PRAGMA page_count", [], |row| row.get(0))
                .map_err(LogStoreError::Sqlite)?;
            u32::try_from(page_count_before.saturating_sub(page_count_after)).unwrap_or(u32::MAX)
        } else {
            0
        };
        self.prepare_private_database_files()?;
        Ok(SQLiteSpaceMaintenance {
            optimized: true,
            incremental_vacuum_pages,
        })
    }

    fn prepare_private_database_files(&self) -> Result<(), LogStoreError> {
        prepare_private_database_files(&self.db_path)
    }

    #[cfg(test)]
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn schema_version(&self) -> u32 {
        self.conn()
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap_or(0) as u32
    }

    #[cfg(test)]
    pub fn reopen(&self, clock: std::sync::Arc<dyn Clock>) -> Result<Self, LogStoreError> {
        let parent = self.db_path.parent().ok_or_else(|| {
            LogStoreError::IoError(std::io::Error::other("no parent dir for db path"))
        })?;

        Self::open(parent, clock)
    }
}

fn prepare_private_store_root(root: &Path) -> Result<PathBuf, LogStoreError> {
    let privacy = PlatformArtifactPrivacy;
    create_private_directory_tree(root, &privacy)?;
    let canonical = root.canonicalize().map_err(LogStoreError::IoError)?;
    privacy.prepare_directory(&canonical)?;
    Ok(canonical)
}

/// SQLite can materialize `-wal` and `-shm` lazily. Prepare the main database
/// and every sidecar that exists after opening/committing, rather than relying
/// on the process umask or inherited Windows ACL alone.
fn prepare_private_database_files(db_path: &Path) -> Result<(), LogStoreError> {
    let privacy = PlatformArtifactPrivacy;
    privacy.prepare_file(db_path)?;
    for suffix in ["-wal", "-shm"] {
        let path = PathBuf::from(format!("{}{}", db_path.display(), suffix));
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(LogStoreError::PathUnsafe {
                    segment: "symlink_not_allowed".to_string(),
                });
            }
            Ok(_) => privacy.prepare_file(&path)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(LogStoreError::IoError(error)),
        }
    }
    Ok(())
}

fn reject_link_if_present(path: &Path) -> Result<(), LogStoreError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(LogStoreError::PathUnsafe {
            segment: "symlink_not_allowed".to_string(),
        }),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(LogStoreError::IoError(error)),
    }
}

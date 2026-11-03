//! One unresolved Ethereum nonce is reserved by its immutable signed transaction hash.
use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use rusqlite::{Connection, OptionalExtension, params};
use std::{
    fs::{File, OpenOptions},
    path::Path,
    sync::Mutex,
};
use wallet_contract::{PayRequest, PaymentAttempt, PaymentState, WalletInfo};

pub struct Journal {
    db: Mutex<Connection>,
    _lock: File,
}

pub struct Stored {
    pub authorized: PayRequest,
    pub attempt: PaymentAttempt,
    pub signed: serde_json::Value,
    /// Local policy deadline only. An Ethereum signature has no native expiry.
    pub expires_at: u64,
}

pub fn private_directory(path: &Path) -> Result<()> {
    ensure!(path.is_absolute(), "data_dir must be absolute");
    let metadata = std::fs::symlink_metadata(path)
        .context("wallet directory missing; explicitly initialize it first")?;
    ensure!(
        metadata.file_type().is_dir(),
        "data_dir must be a real directory"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "data_dir must be owner-only (0700)"
        );
    }
    Ok(())
}

fn private_open(path: &Path) -> Result<File> {
    if let Ok(metadata) = std::fs::symlink_metadata(path) {
        ensure!(
            metadata.file_type().is_file(),
            "wallet state must be a regular file"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            ensure!(
                metadata.permissions().mode() & 0o077 == 0,
                "wallet state must be owner-only (0600)"
            );
        }
    }
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

impl Journal {
    pub fn open(directory: &Path, info: &WalletInfo) -> Result<Self> {
        private_directory(directory)?;
        let lock = private_open(&directory.join("wallet.lock"))?;
        lock.try_lock_exclusive()
            .context("another process owns this Ethereum wallet")?;
        let path = directory.join("payments.sqlite3");
        let _private = private_open(&path)?;
        let db = Connection::open(path)?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000;
          CREATE TABLE IF NOT EXISTS identity(id INTEGER PRIMARY KEY CHECK(id=1),body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS attempts(payment_id TEXT PRIMARY KEY,authorized TEXT NOT NULL,
            attempt TEXT NOT NULL,signed TEXT NOT NULL,expires_at TEXT NOT NULL,active INTEGER NOT NULL);
          CREATE UNIQUE INDEX IF NOT EXISTS one_active_transfer ON attempts(active) WHERE active=1;")?;
        let identity = serde_json::json!({
            "wallet_id":info.wallet_id,"address":info.address,"asset":info.asset,"method":info.method,
            "fee_asset":info.effective_fee_asset()
        });
        db.execute(
            "INSERT OR IGNORE INTO identity VALUES(1,?1)",
            [identity.to_string()],
        )?;
        let previous: String =
            db.query_row("SELECT body FROM identity WHERE id=1", [], |row| row.get(0))?;
        ensure!(
            serde_json::from_str::<serde_json::Value>(&previous)? == identity,
            "Ethereum journal belongs to another wallet, asset or method"
        );
        Ok(Self {
            db: Mutex::new(db),
            _lock: lock,
        })
    }

    pub fn get(&self, id: &str) -> Result<Option<Stored>> {
        let db = self.db.lock().unwrap();
        let row: Option<(String, String, String, String)> = db
            .query_row(
                "SELECT authorized,attempt,signed,expires_at FROM attempts WHERE payment_id=?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        row.map(|(authorized, attempt, signed, expires_at)| {
            Ok(Stored {
                authorized: serde_json::from_str(&authorized)?,
                attempt: serde_json::from_str(&attempt)?,
                signed: serde_json::from_str(&signed)?,
                expires_at: expires_at.parse()?,
            })
        })
        .transpose()
    }

    pub fn list(&self) -> Result<Vec<PaymentAttempt>> {
        let db = self.db.lock().unwrap();
        let mut query = db.prepare("SELECT attempt FROM attempts ORDER BY rowid")?;
        query
            .query_map([], |row| row.get::<_, String>(0))?
            .map(|row| Ok(serde_json::from_str(&row?)?))
            .collect()
    }

    pub fn unresolved(&self) -> Result<Vec<PaymentAttempt>> {
        Ok(self
            .list()?
            .into_iter()
            .filter(|attempt| !completed(&attempt.state))
            .collect())
    }

    pub fn reserved(&self) -> Result<u64> {
        self.unresolved()?.iter().try_fold(0u64, |sum, attempt| {
            sum.checked_add(attempt.request.obligation.amount)
                .context("USDT reservation overflow")
        })
    }

    pub fn insert(
        &self,
        request: &PayRequest,
        attempt: &PaymentAttempt,
        signed: &serde_json::Value,
        expires_at: u64,
    ) -> Result<()> {
        ensure!(
            attempt.state == PaymentState::Prepared,
            "must journal before broadcasting"
        );
        ensure!(
            attempt.tx_hashes.len() == 1,
            "Ethereum attempt needs exactly one transaction ID"
        );
        self.db
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO attempts VALUES(?1,?2,?3,?4,?5,1)",
                params![
                    request.payment_id,
                    serde_json::to_string(request)?,
                    serde_json::to_string(attempt)?,
                    serde_json::to_string(signed)?,
                    expires_at.to_string()
                ],
            )
            .context("another transfer is unresolved or this payment ID already exists")?;
        Ok(())
    }

    pub fn update(&self, attempt: &PaymentAttempt) -> Result<()> {
        let count = self.db.lock().unwrap().execute(
            "UPDATE attempts SET attempt=?1,active=?2 WHERE payment_id=?3",
            params![
                serde_json::to_string(attempt)?,
                !completed(&attempt.state),
                attempt.payment_id
            ],
        )?;
        ensure!(count == 1, "unknown payment attempt");
        Ok(())
    }
}

fn completed(state: &PaymentState) -> bool {
    // An Ethereum transaction has no native expiry. Absence or a local deadline
    // cannot release its nonce/payment reservation. Only finalized success or
    // finalized revert conclusively resolves this attempt.
    matches!(state, PaymentState::Finalized | PaymentState::Reverted)
}

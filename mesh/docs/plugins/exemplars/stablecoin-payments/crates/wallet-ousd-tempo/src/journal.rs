//! A write-ahead payment journal. Signed bytes commit before any broadcast.
use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use rusqlite::{Connection, OptionalExtension, params};
use std::{
    fs::{File, OpenOptions},
    path::Path,
    sync::Mutex,
};
use wallet_contract::{PayRequest, PaymentAttempt, PaymentState};

pub struct Journal {
    db: Mutex<Connection>,
    _lock: File,
}
pub struct Stored {
    pub request: PayRequest,
    pub attempt: PaymentAttempt,
    pub raw: Vec<u8>,
    pub nonce: u64,
    pub reserved: u64,
}

impl Journal {
    pub fn open(dir: &Path, wallet_id: &str, address: &str) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let lock = options.open(dir.join("wallet.lock"))?;
        lock.try_lock_exclusive()
            .context("another wallet process owns this state directory")?;
        let path = dir.join("payments.sqlite3");
        let _private_db = options.open(&path)?;
        let db = Connection::open(path)?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000;
          CREATE TABLE IF NOT EXISTS identity(singleton INTEGER PRIMARY KEY CHECK(singleton=1), wallet_id TEXT NOT NULL,address TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS attempts(payment_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, attempt_json TEXT NOT NULL,
            nonce INTEGER NOT NULL, raw_tx BLOB NOT NULL, reserved TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1);
          CREATE UNIQUE INDEX IF NOT EXISTS active_nonce ON attempts(nonce) WHERE active=1;")?;
        db.execute(
            "INSERT OR IGNORE INTO identity VALUES(1,?1,?2)",
            params![wallet_id, address],
        )?;
        let existing: (String, String) = db.query_row(
            "SELECT wallet_id,address FROM identity WHERE singleton=1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        ensure!(
            existing.0 == wallet_id && existing.1.eq_ignore_ascii_case(address),
            "wallet journal belongs to another account"
        );
        Ok(Self {
            db: Mutex::new(db),
            _lock: lock,
        })
    }
    pub fn get(&self, id: &str) -> Result<Option<Stored>> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("journal lock poisoned"))?;
        let result: Option<(String,String,Vec<u8>,u64,String)> = db.query_row(
          "SELECT request_json,attempt_json,raw_tx,nonce,reserved FROM attempts WHERE payment_id=?1",[id],
          |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?))).optional()?;
        result
            .map(|(request, attempt, raw, nonce, reserved)| {
                Ok(Stored {
                    request: serde_json::from_str(&request)?,
                    attempt: serde_json::from_str(&attempt)?,
                    raw,
                    nonce,
                    reserved: reserved.parse()?,
                })
            })
            .transpose()
    }
    pub fn list(&self) -> Result<Vec<PaymentAttempt>> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("journal lock poisoned"))?;
        let mut query = db.prepare("SELECT attempt_json FROM attempts ORDER BY nonce")?;
        let rows = query.query_map([], |r| r.get::<_, String>(0))?;
        rows.map(|row| Ok(serde_json::from_str(&row?)?)).collect()
    }
    pub fn unresolved(&self) -> Result<Vec<PaymentAttempt>> {
        Ok(self
            .list()?
            .into_iter()
            .filter(|a| {
                !matches!(
                    a.state,
                    PaymentState::Finalized | PaymentState::Reverted | PaymentState::Expired
                )
            })
            .collect())
    }
    pub fn reserved(&self) -> Result<u64> {
        self.unresolved()?.into_iter().try_fold(0u64, |sum, a| {
            sum.checked_add(
                self.get(&a.payment_id)?
                    .context("missing attempt")?
                    .reserved,
            )
            .context("reservation overflow")
        })
    }
    pub fn insert(
        &self,
        request: &PayRequest,
        attempt: &PaymentAttempt,
        nonce: u64,
        raw: &[u8],
        reserved: u64,
    ) -> Result<()> {
        ensure!(nonce <= i64::MAX as u64, "nonce exceeds journal range");
        ensure!(
            attempt.state == PaymentState::Prepared,
            "journal must precede broadcast"
        );
        self.db
            .lock()
            .map_err(|_| anyhow::anyhow!("journal lock poisoned"))?
            .execute(
                "INSERT INTO attempts(payment_id,request_json,attempt_json,nonce,raw_tx,reserved) VALUES(?1,?2,?3,?4,?5,?6)",
                params![
                    request.payment_id,
                    serde_json::to_string(request)?,
                    serde_json::to_string(attempt)?,
                    nonce,
                    raw,
                    reserved.to_string()
                ],
            )?;
        Ok(())
    }
    pub fn update(&self, attempt: &PaymentAttempt) -> Result<()> {
        let active = !matches!(
            attempt.state,
            PaymentState::Finalized | PaymentState::Reverted | PaymentState::Expired
        );
        let count = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("journal lock poisoned"))?
            .execute(
                "UPDATE attempts SET attempt_json=?1,active=?2 WHERE payment_id=?3",
                params![serde_json::to_string(attempt)?, active, attempt.payment_id],
            )?;
        ensure!(count == 1, "unknown payment attempt");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exclusive_process_lock_prevents_second_nonce_owner() {
        let dir = tempfile::tempdir().unwrap();
        let _first = Journal::open(
            dir.path(),
            "fixture",
            "0x1111111111111111111111111111111111111111",
        )
        .unwrap();
        assert!(
            Journal::open(
                dir.path(),
                "fixture",
                "0x1111111111111111111111111111111111111111"
            )
            .is_err()
        );
    }
}

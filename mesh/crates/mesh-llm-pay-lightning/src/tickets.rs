//! Durable channel admission to host-exchange correlation; never replay work.
use anyhow::{Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use std::{path::Path, sync::Mutex};

pub struct Tickets(Mutex<Connection>);
impl Tickets {
    pub fn open(directory: &Path) -> Result<Self> {
        let db = Connection::open(directory.join("plugin-tickets.sqlite"))?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS tickets(id TEXT PRIMARY KEY, peer TEXT NOT NULL, model TEXT NOT NULL, ready INTEGER NOT NULL DEFAULT 0, exchange TEXT UNIQUE, admission TEXT);")?;
        // PaymentService abandons all admitted work at restart. Never revive its tickets.
        db.execute("DELETE FROM tickets WHERE exchange IS NULL", [])?;
        Ok(Self(Mutex::new(db)))
    }
    pub fn insert(&self, id: &str, peer: &str, model: &str) -> Result<()> {
        self.0.lock().unwrap().execute(
            "INSERT INTO tickets(id,peer,model) VALUES (?1,?2,?3)",
            params![id, peer, model],
        )?;
        Ok(())
    }
    pub fn ready(&self, id: &str, peer: &str) -> Result<()> {
        let changed = self.0.lock().unwrap().execute(
            "UPDATE tickets SET ready=1 WHERE id=?1 AND peer=?2 AND exchange IS NULL",
            params![id, peer],
        )?;
        ensure!(changed == 1, "ticket consumed or unknown");
        Ok(())
    }
    pub fn consume(&self, peer: &str, model: &str, exchange: &str) -> Result<bool> {
        let mut db = self.0.lock().unwrap();
        let tx = db.transaction()?;
        let id: Option<String> = tx.query_row("SELECT id FROM tickets WHERE peer=?1 AND model=?2 AND ready=1 AND exchange IS NULL ORDER BY rowid LIMIT 1", params![peer, model], |r| r.get(0)).optional()?;
        let Some(id) = id else {
            return Ok(false);
        };
        tx.execute(
            "UPDATE tickets SET exchange=?1,ready=0 WHERE id=?2",
            params![exchange, id],
        )?;
        tx.commit()?;
        Ok(true)
    }
    pub fn exchange(&self, exchange: &str) -> Result<Option<(String, String)>> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .query_row(
                "SELECT peer,id FROM tickets WHERE exchange=?",
                [exchange],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?)
    }
    pub fn consumed(&self, id: &str, peer: &str) -> Result<bool> {
        Ok(self.0.lock().unwrap().query_row(
            "SELECT EXISTS(SELECT 1 FROM tickets WHERE id=?1 AND peer=?2 AND exchange IS NOT NULL)",
            params![id, peer],
            |r| r.get(0),
        )?)
    }
    pub fn save_admission(&self, id: &str, admission: &str) -> Result<()> {
        self.0.lock().unwrap().execute(
            "UPDATE tickets SET admission=?1 WHERE id=?2",
            params![admission, id],
        )?;
        Ok(())
    }
    pub fn admission(&self, peer: &str, model: &str) -> Result<Option<String>> {
        Ok(self.0.lock().unwrap().query_row("SELECT admission FROM tickets WHERE peer=?1 AND model=?2 AND exchange IS NULL AND admission IS NOT NULL ORDER BY rowid LIMIT 1", params![peer,model], |r|r.get(0)).optional()?)
    }
    pub fn outstanding(&self, peer: &str, model: &str) -> Result<bool> {
        Ok(self.0.lock().unwrap().query_row(
            "SELECT EXISTS(SELECT 1 FROM tickets WHERE peer=?1 AND model=?2 AND exchange IS NULL)",
            params![peer, model],
            |r| r.get(0),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn single_use_owner_bound_and_durable() {
        let dir = tempfile::tempdir().unwrap();
        let tickets = Tickets::open(dir.path()).unwrap();
        tickets.insert("ticket", "buyer", "model").unwrap();
        assert!(tickets.ready("ticket", "attacker").is_err());
        tickets.ready("ticket", "buyer").unwrap();
        assert!(!tickets.consume("attacker", "model", "x").unwrap());
        assert!(tickets.consume("buyer", "model", "x").unwrap());
        assert!(tickets.ready("ticket", "buyer").is_err());
        assert!(!tickets.consume("buyer", "model", "y").unwrap());
        drop(tickets);
        let reopened = Tickets::open(dir.path()).unwrap();
        assert_eq!(
            reopened.exchange("x").unwrap(),
            Some(("buyer".into(), "ticket".into()))
        );
    }
}

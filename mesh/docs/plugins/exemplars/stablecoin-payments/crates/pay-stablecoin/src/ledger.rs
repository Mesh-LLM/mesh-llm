use crate::{config::now, protocol::Exchange};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use std::{path::Path, sync::Mutex};
use wallet_contract::{Receipt, WalletInfo};

pub struct Ledger(Mutex<Connection>);
impl Ledger {
    pub fn open(directory: &Path) -> Result<Self> {
        std::fs::create_dir_all(directory)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
        }
        let db = Connection::open(directory.join("payments.sqlite"))?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS exchanges(id TEXT PRIMARY KEY,peer TEXT NOT NULL,seller INTEGER NOT NULL,active INTEGER NOT NULL,body TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS one_peer_exchange ON exchanges(peer) WHERE active=1;
            CREATE TABLE IF NOT EXISTS standing(peer TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS receipts(receipt_key TEXT PRIMARY KEY,obligation TEXT UNIQUE NOT NULL);
            CREATE TABLE IF NOT EXISTS budgets(id TEXT PRIMARY KEY,day INTEGER NOT NULL,amount TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS identity(id INTEGER PRIMARY KEY CHECK(id=1),body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS node_identity(id INTEGER PRIMARY KEY CHECK(id=1),node TEXT NOT NULL);")?;
        Ok(Self(Mutex::new(db)))
    }
    pub fn pin_node(&self, node: &str) -> Result<()> {
        crate::config::validate_peer(node)?;
        ensure!(
            node == node.to_ascii_lowercase(),
            "Mesh node identity must be canonical lowercase hexadecimal"
        );
        let mut db = self.0.lock().unwrap();
        let tx = db.transaction()?;
        let pinned: Option<String> = tx
            .query_row("SELECT node FROM node_identity WHERE id=1", [], |r| {
                r.get(0)
            })
            .optional()?;
        if let Some(pinned) = pinned {
            ensure!(
                pinned == node,
                "stable Mesh identity required: node ID changed; restore the pinned node key and run serve with runtime.mode='on_demand', never an ephemeral client identity"
            );
        } else {
            let history: bool =
                tx.query_row("SELECT EXISTS(SELECT 1 FROM exchanges)", [], |r| r.get(0))?;
            ensure!(
                !history,
                "existing payment ledger has no node identity pin; operator review is required before adopting a Mesh identity"
            );
            tx.execute("INSERT INTO node_identity(id,node) VALUES(1,?1)", [node])?;
        }
        tx.commit()?;
        Ok(())
    }
    pub fn pin(&self, info: &WalletInfo) -> Result<()> {
        let db = self.0.lock().unwrap();
        let identity = serde_json::to_string(info)?;
        let pinned: Option<String> = db
            .query_row("SELECT body FROM identity WHERE id=1", [], |r| r.get(0))
            .optional()?;
        if let Some(pinned) = pinned {
            let old: WalletInfo = serde_json::from_str(&pinned)?;
            ensure!(
                old.wallet_id == info.wallet_id
                    && old.address == info.address
                    && old.asset == info.asset
                    && old.effective_fee_asset() == info.effective_fee_asset()
                    && old.method == info.method,
                "wallet identity changed; refusing to adopt another account"
            );
        } else {
            db.execute("INSERT INTO identity(id,body) VALUES(1,?1)", [identity])?;
        }
        Ok(())
    }
    /// A restarted host cannot still own an unstarted admission's HTTP observation.
    /// Running or billed work is deliberately retained and may not be inferred again.
    pub fn detach_buyer_admissions(&self) -> Result<()> {
        for exchange in self.active()? {
            if !exchange.seller
                && matches!(exchange.state.as_str(), "negotiating" | "ready")
                && exchange.exchange_id.is_some()
            {
                self.update(&exchange.id, |current| {
                    if !current.seller && matches!(current.state.as_str(), "negotiating" | "ready")
                    {
                        current.exchange_id = None;
                    }
                    Ok(())
                })?;
            }
        }
        Ok(())
    }
    pub fn insert(&self, exchange: &Exchange) -> Result<()> {
        let db = self.0.lock().unwrap();
        db.execute(
            "INSERT INTO exchanges(id,peer,seller,active,body) VALUES(?1,?2,?3,1,?4)",
            params![
                exchange.id,
                exchange.peer,
                exchange.seller,
                serde_json::to_string(exchange)?
            ],
        )
        .context("one outstanding exchange per peer; settle or cancel existing work first")?;
        Ok(())
    }
    pub fn get(&self, id: &str) -> Result<Exchange> {
        let db = self.0.lock().unwrap();
        let body: String =
            db.query_row("SELECT body FROM exchanges WHERE id=?1", [id], |r| r.get(0))?;
        Ok(serde_json::from_str(&body)?)
    }
    pub fn update<T>(&self, id: &str, f: impl FnOnce(&mut Exchange) -> Result<T>) -> Result<T> {
        let mut db = self.0.lock().unwrap();
        let tx = db.transaction()?;
        let body: String =
            tx.query_row("SELECT body FROM exchanges WHERE id=?1", [id], |r| r.get(0))?;
        let mut exchange: Exchange = serde_json::from_str(&body)?;
        let result = f(&mut exchange)?;
        let active = !matches!(exchange.state.as_str(), "settled" | "cancelled");
        tx.execute(
            "UPDATE exchanges SET active=?1,body=?2 WHERE id=?3",
            params![active, serde_json::to_string(&exchange)?, id],
        )?;
        tx.commit()?;
        Ok(result)
    }
    pub fn active(&self) -> Result<Vec<Exchange>> {
        let db = self.0.lock().unwrap();
        let mut statement =
            db.prepare("SELECT body FROM exchanges WHERE active=1 ORDER BY rowid")?;
        let bodies = statement
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        bodies
            .into_iter()
            .map(|body| Ok(serde_json::from_str(&body)?))
            .collect()
    }
    pub fn earned(&self, peer: &str) -> Result<bool> {
        Ok(self.0.lock().unwrap().query_row(
            "SELECT EXISTS(SELECT 1 FROM standing WHERE peer=?1)",
            [peer],
            |r| r.get(0),
        )?)
    }
    pub fn receipt(
        &self,
        obligation: &str,
        peer: &str,
        receipt: &Receipt,
        admission: bool,
    ) -> Result<()> {
        let mut db = self.0.lock().unwrap();
        let tx = db.transaction()?;
        let identity: String =
            tx.query_row("SELECT body FROM identity WHERE id=1", [], |r| r.get(0))?;
        let wallet: WalletInfo = serde_json::from_str(&identity)?;
        let key = format!(
            "{}:{}:{}:{}",
            wallet.asset.chain_id, wallet.asset.token, receipt.tx_hash, receipt.transfer_log_index
        );
        let old: Option<String> = tx
            .query_row(
                "SELECT obligation FROM receipts WHERE receipt_key=?1",
                [&key],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(old) = old {
            ensure!(
                old == obligation,
                "receipt was consumed by another obligation"
            );
        } else {
            tx.execute(
                "INSERT INTO receipts(receipt_key,obligation) VALUES(?1,?2)",
                params![key, obligation],
            )?;
        }
        if admission {
            tx.execute("INSERT OR IGNORE INTO standing(peer) VALUES(?1)", [peer])?;
        }
        tx.commit()?;
        Ok(())
    }
    pub fn reserve_budget(&self, id: &str, amount: u64, maximum: u64) -> Result<()> {
        let mut db = self.0.lock().unwrap();
        let tx = db.transaction()?;
        let old: Option<String> = tx
            .query_row("SELECT amount FROM budgets WHERE id=?1", [id], |r| r.get(0))
            .optional()?;
        if let Some(old) = old {
            ensure!(old.parse::<u64>()? == amount, "payment amount changed");
            return Ok(());
        }
        let day = now() / 86400;
        let mut query = tx.prepare("SELECT amount FROM budgets WHERE day=?1")?;
        let amounts = query
            .query_map([day], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut total = amount;
        for value in amounts {
            total = total
                .checked_add(value.parse()?)
                .context("daily budget overflow")?;
        }
        ensure!(total <= maximum, "daily budget exceeded");
        drop(query);
        tx.execute(
            "INSERT INTO budgets(id,day,amount) VALUES(?1,?2,?3)",
            params![id, day, amount.to_string()],
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn budget_reserved(&self, id: &str) -> Result<bool> {
        Ok(self.0.lock().unwrap().query_row(
            "SELECT EXISTS(SELECT 1 FROM budgets WHERE id=?1)",
            [id],
            |r| r.get(0),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn exchange(id: &str) -> Exchange {
        Exchange {
            id: id.into(),
            peer: "peer".into(),
            model: "model".into(),
            payer: "payer".into(),
            seller: true,
            exchange_id: None,
            state: "ready".into(),
            quote: None,
            usage: None,
            usage_payment: None,
            admission_receipt: None,
            usage_receipt: None,
            error: None,
        }
    }
    #[test]
    fn durable_budget_is_reserved_once_and_never_wraps() {
        let temp = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(temp.path()).unwrap();
        ledger.reserve_budget("a", 6, 10).unwrap();
        ledger.reserve_budget("a", 6, 10).unwrap();
        assert!(ledger.reserve_budget("b", 5, 10).is_err());
        assert!(ledger.reserve_budget("a", 5, 10).is_err());
        drop(ledger);
        assert!(
            Ledger::open(temp.path())
                .unwrap()
                .reserve_budget("b", 5, 10)
                .is_err()
        );
    }
    #[test]
    fn stable_node_identity_survives_restart_and_rejects_an_ephemeral_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let info = WalletInfo {
            wallet_id: "same-wallet".into(),
            address: "fixture-account".into(),
            asset: wallet_contract::Asset::default(),
            method: "fixture".into(),
            can_send: true,
            fee_asset: None,
        };
        let ledger = Ledger::open(temp.path()).unwrap();
        ledger.pin_node(&"aa".repeat(32)).unwrap();
        ledger.pin(&info).unwrap();
        drop(ledger);
        let reopened = Ledger::open(temp.path()).unwrap();
        reopened.pin_node(&"aa".repeat(32)).unwrap();
        reopened.pin(&info).unwrap();
        let error = reopened.pin_node(&"bb".repeat(32)).unwrap_err();
        assert!(error.to_string().contains("stable Mesh identity required"));
        assert!(reopened.pin_node(&"AA".repeat(32)).is_err());
        assert!(reopened.pin_node("wrong").is_err());
    }
    #[test]
    fn outstanding_work_blocks_new_model_and_survives_restart() {
        let temp = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(temp.path()).unwrap();
        ledger.insert(&exchange("first")).unwrap();
        let mut second = exchange("second");
        second.model = "another-model".into();
        assert!(ledger.insert(&second).is_err());
        drop(ledger);
        let ledger = Ledger::open(temp.path()).unwrap();
        assert!(ledger.insert(&second).is_err());
        ledger
            .update("first", |x| {
                x.state = "settled".into();
                Ok(())
            })
            .unwrap();
        ledger.insert(&second).unwrap();
    }
    #[test]
    fn each_finalized_transfer_can_settle_only_one_obligation() {
        let temp = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(temp.path()).unwrap();
        ledger
            .pin(&WalletInfo {
                wallet_id: "test".into(),
                address: "fixture-account".into(),
                asset: wallet_contract::Asset::default(),
                method: "fixture".into(),
                can_send: true,
                fee_asset: None,
            })
            .unwrap();
        let receipt = Receipt {
            tx_hash: "0xabc".into(),
            block_hash: "0xdef".into(),
            block_number: 1,
            transfer_log_index: 4,
            payer: "p".into(),
            recipient: "r".into(),
            amount: 1,
            memo: "memo".into(),
            finalized: true,
        };
        ledger.receipt("a", "peer", &receipt, true).unwrap();
        ledger.receipt("a", "peer", &receipt, true).unwrap();
        assert!(ledger.earned("peer").unwrap());
        assert!(ledger.receipt("b", "peer", &receipt, false).is_err());
    }
    #[test]
    fn fee_currency_is_pinned_but_explicit_payment_asset_keeps_legacy_identity() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(directory.path()).unwrap();
        let mut info = WalletInfo {
            wallet_id: "legacy".into(),
            address: "account".into(),
            asset: wallet_contract::Asset::default(),
            method: "fixture".into(),
            can_send: true,
            fee_asset: None,
        };
        ledger.pin(&info).unwrap();
        info.fee_asset = Some(info.asset.clone());
        ledger.pin(&info).unwrap();
        info.fee_asset = Some(wallet_contract::Asset {
            token: "native:fee-token".into(),
            ..info.asset.clone()
        });
        assert!(ledger.pin(&info).is_err());
        let separate = tempfile::tempdir().unwrap();
        let separate = Ledger::open(separate.path()).unwrap();
        separate.pin(&info).unwrap();
        info.fee_asset = None;
        assert!(separate.pin(&info).is_err());
    }
    #[test]
    fn restart_detaches_unstarted_buyer_hooks_without_replaying_running_work() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(directory.path()).unwrap();
        for (id, peer, seller, state) in [
            ("pending", "buyer-pending", false, "negotiating"),
            ("ready", "buyer-ready", false, "ready"),
            ("running", "buyer-running", false, "running"),
            ("seller", "serving-peer", true, "ready"),
        ] {
            let mut row = exchange(id);
            row.peer = peer.into();
            row.seller = seller;
            row.state = state.into();
            row.exchange_id = Some(format!("{id}-observation"));
            ledger.insert(&row).unwrap();
        }
        drop(ledger);
        let ledger = Ledger::open(directory.path()).unwrap();
        ledger.detach_buyer_admissions().unwrap();
        assert!(ledger.get("pending").unwrap().exchange_id.is_none());
        assert!(ledger.get("ready").unwrap().exchange_id.is_none());
        assert_eq!(
            ledger.get("running").unwrap().exchange_id.as_deref(),
            Some("running-observation")
        );
        assert_eq!(ledger.get("running").unwrap().state, "running");
        assert_eq!(
            ledger.get("seller").unwrap().exchange_id.as_deref(),
            Some("seller-observation")
        );
    }
}

use crate::{
    bridge::{self, Bridge, Worker},
    journal::Journal,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Mutex;
use wallet_contract::{
    Asset, Balance, FeeEstimate, PayRequest, PaymentAttempt, PaymentRequest, PaymentState,
    VerifyRequest, VerifyResult, WalletInfo,
};

pub const CHAIN_ID: u64 = 1;
pub const USDT: &str = "0xdac17f958d2ee523a2206206994597c13d831ec7";
pub const METHOD: &str = "ethereum.erc20-memo.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub data_dir: PathBuf,
    pub rpc_url: String,
    pub password_file: Option<PathBuf>,
    pub node_binary: PathBuf,
    pub worker_path: Option<PathBuf>,
    #[serde(with = "wallet_contract::decimal")]
    pub max_payment: u64,
    #[serde(with = "wallet_contract::decimal")]
    pub max_fee: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("wallet-usdt-ethereum-data"),
            rpc_url: "https://ethereum-rpc.publicnode.com".into(),
            password_file: None,
            node_binary: PathBuf::from("node"),
            worker_path: None,
            max_payment: 1_000_000,
            max_fee: 0,
        }
    }
}

impl Settings {
    pub fn worker_settings(&self) -> Value {
        let mut value = json!({"data_dir":self.data_dir,"rpc_url":self.rpc_url});
        if let Some(path) = &self.password_file {
            value["password_file"] = json!(path);
        }
        value
    }
}

#[derive(Deserialize)]
pub struct Prepared {
    pub signed_transaction: String,
    pub tx_hash: String,
    /// Local policy deadline in milliseconds, not an Ethereum transaction expiry.
    pub expires_at: u64,
    /// The nonce is also committed inside the signed raw transaction.
    pub nonce: String,
    #[serde(with = "wallet_contract::decimal")]
    pub fee: u64,
}

pub struct Wallet {
    settings: Settings,
    pub info: WalletInfo,
    worker: Arc<dyn Worker>,
    pub journal: Journal,
    serial: Mutex<()>,
    checked: Mutex<std::collections::BTreeMap<String, tokio::time::Instant>>,
}

impl Wallet {
    pub async fn load(settings: Settings) -> Result<Arc<Self>> {
        crate::journal::private_directory(&settings.data_dir)?;
        let worker = Bridge::spawn(&settings).await?;
        let info: WalletInfo = serde_json::from_value(
            bridge::invoke(worker.as_ref(), "open", settings.worker_settings()).await?,
        )?;
        Self::with_worker(settings, info, worker)
    }

    pub fn with_worker(
        settings: Settings,
        info: WalletInfo,
        worker: Arc<dyn Worker>,
    ) -> Result<Arc<Self>> {
        ensure!(
            info.asset == asset() && info.method == METHOD,
            "wallet is not mainnet USDT on Ethereum"
        );
        ensure!(
            info.effective_fee_asset() == &fee_asset(),
            "Ethereum wallet fees must be denominated in native ETH"
        );
        ensure!(
            canonical_address(&info.address),
            "wallet address must be canonical lowercase hex"
        );
        let journal = Journal::open(&settings.data_dir, &info)?;
        Ok(Arc::new(Self {
            settings,
            info,
            worker,
            journal,
            serial: Mutex::new(()),
            checked: Mutex::new(Default::default()),
        }))
    }

    pub async fn info(&self) -> Result<WalletInfo> {
        let current: WalletInfo = self.call("info", json!({})).await?;
        ensure!(
            current.wallet_id == self.info.wallet_id
                && current.address == self.info.address
                && current.asset == self.info.asset
                && current.method == self.info.method
                && current.effective_fee_asset() == self.info.effective_fee_asset(),
            "WDK wallet identity changed"
        );
        Ok(current)
    }

    async fn call<T: serde::de::DeserializeOwned>(
        &self,
        operation: &str,
        input: Value,
    ) -> Result<T> {
        Ok(serde_json::from_value(
            bridge::invoke(self.worker.as_ref(), operation, input).await?,
        )?)
    }

    pub async fn balance(&self) -> Result<Balance> {
        let mut balance: Balance = self.call("balance", json!({})).await?;
        balance.reserved = self.journal.reserved()?;
        balance.spendable = balance.balance.saturating_sub(balance.reserved);
        Ok(balance)
    }

    fn validate_request(&self, request: &PaymentRequest) -> Result<()> {
        request.validate()?;
        ensure!(
            request.obligation.asset == asset(),
            "only canonical mainnet USDT is supported"
        );
        ensure!(
            request.obligation.payer == self.info.address,
            "payer is not this Ethereum wallet"
        );
        ensure!(
            canonical_address(&request.obligation.recipient),
            "recipient must be canonical lowercase hex"
        );
        ensure!(
            request.obligation.amount <= self.settings.max_payment,
            "payment exceeds local USDT limit"
        );
        Ok(())
    }

    pub async fn estimate(&self, request: PaymentRequest) -> Result<FeeEstimate> {
        self.validate_request(&request)?;
        let estimate: FeeEstimate = self
            .call("estimate", serde_json::to_value(request)?)
            .await?;
        ensure!(
            estimate.fee_asset == fee_asset(),
            "unexpected fee currency from Ethereum worker"
        );
        Ok(estimate)
    }

    pub async fn pay(&self, request: PayRequest) -> Result<PaymentAttempt> {
        let _serial = self.serial.lock().await;
        let id = request.payment_id.clone();
        match tokio::time::timeout(Duration::from_secs(25), self.pay_locked(request)).await {
            Ok(result) => result,
            Err(_) => {
                let mut attempt = self.journal.get(&id)?.context(
                    "preparation timed out before a signed transaction was journaled; inspect before resubmitting")?.attempt;
                if !matches!(
                    attempt.state,
                    PaymentState::Finalized | PaymentState::Reverted | PaymentState::Held
                ) {
                    attempt.state = PaymentState::Uncertain;
                    attempt.error = Some("wallet deadline reached; restart provider if needed and reconcile the original transaction ID".into());
                    self.journal.update(&attempt)?;
                }
                Ok(attempt)
            }
        }
    }

    async fn pay_locked(&self, request: PayRequest) -> Result<PaymentAttempt> {
        self.validate_request(&request.request)?;
        ensure!(
            !request.payment_id.is_empty() && request.payment_id.len() <= 128,
            "invalid payment ID"
        );
        ensure!(
            request.payment_id == request.request.obligation.id,
            "payment ID must equal the immutable obligation ID"
        );
        if let Some(stored) = self.journal.get(&request.payment_id)? {
            ensure!(
                stored.authorized.request == request.request,
                "payment ID reused with different terms"
            );
            return self.resume(&request.payment_id, true).await;
        }
        ensure!(
            self.info.can_send,
            "wallet is locked; configure its password_file"
        );
        ensure!(
            request.request.obligation.expires_at > seconds()?,
            "payment obligation expired"
        );
        ensure!(
            request.max_total >= request.request.obligation.amount,
            "authorized USDT total is below payment amount"
        );
        ensure!(
            self.journal.unresolved()?.is_empty(),
            "an earlier Ethereum transfer is unresolved; reconcile it first"
        );
        let prepared: Prepared = self
            .call("prepare", serde_json::to_value(&request)?)
            .await?;
        ensure!(
            canonical_hash(&prepared.tx_hash),
            "invalid prepared Ethereum transaction ID"
        );
        ensure!(
            prepared.signed_transaction.starts_with("0x")
                && prepared.signed_transaction.len() > 2
                && prepared.signed_transaction.len().is_multiple_of(2)
                && prepared.signed_transaction[2..]
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()),
            "invalid signed Ethereum transaction encoding"
        );
        ensure!(
            prepared
                .nonce
                .parse::<u64>()
                .is_ok_and(|nonce| nonce.to_string() == prepared.nonce),
            "invalid prepared Ethereum nonce"
        );
        let obligation_deadline = request
            .request
            .obligation
            .expires_at
            .checked_mul(1000)
            .context("obligation expiration overflow")?;
        ensure!(
            prepared.expires_at == obligation_deadline,
            "prepared local deadline does not match immutable obligation"
        );
        for cap in [request.max_fee, self.settings.max_fee] {
            ensure!(
                cap == 0 || prepared.fee <= cap,
                "estimated ETH fee exceeds configured limit"
            );
        }
        let attempt = PaymentAttempt {
            payment_id: request.payment_id.clone(),
            request: request.request.clone(),
            state: PaymentState::Prepared,
            tx_hashes: vec![prepared.tx_hash],
            receipt: None,
            error: None,
        };
        // This FULL-synchronous commit precedes every possible broadcast path.
        self.journal.insert(
            &request,
            &attempt,
            &Value::String(prepared.signed_transaction),
            prepared.expires_at,
        )?;
        self.resume(&request.payment_id, true).await
    }

    async fn resume(&self, id: &str, broadcast: bool) -> Result<PaymentAttempt> {
        let stored = self.journal.get(id)?.context("unknown payment ID")?;
        let mut attempt = stored.attempt;
        if matches!(
            attempt.state,
            PaymentState::Finalized | PaymentState::Reverted | PaymentState::Held
        ) {
            self.checked.lock().await.remove(id);
            return Ok(attempt);
        }
        let hash = attempt
            .tx_hashes
            .first()
            .context("missing original transaction ID")?
            .clone();
        let due = {
            let mut checked = self.checked.lock().await;
            let due = checked
                .get(id)
                .is_none_or(|last| last.elapsed() >= Duration::from_secs(3));
            if due {
                checked.insert(id.into(), tokio::time::Instant::now());
            }
            due
        };
        if due {
            match self
                .verify(VerifyRequest {
                    request: attempt.request.clone(),
                    tx_hash: hash.clone(),
                })
                .await
            {
                Ok(result) => {
                    attempt.state = result.state;
                    attempt.receipt = result.receipt;
                    attempt.error = result.reason;
                }
                Err(_) => {
                    attempt.state = PaymentState::Uncertain;
                    attempt.error = Some(
                        "Ethereum lookup failed; original transaction remains reserved".into(),
                    );
                }
            }
            self.journal.update(&attempt)?;
        }
        if matches!(
            attempt.state,
            PaymentState::Finalized | PaymentState::Reverted | PaymentState::Held
        ) {
            self.checked.lock().await.remove(id);
            return Ok(attempt);
        }
        if broadcast {
            if seconds()? * 1000 >= stored.expires_at {
                attempt.state = PaymentState::Uncertain;
                attempt.error = Some("local obligation deadline passed; signed Ethereum transaction remains valid and reserved until finalized evidence, never automatically replacing it".into());
            } else {
                let result = self
                    .call::<Value>(
                        "broadcast",
                        json!({"request":attempt.request,
                    "signed_transaction":stored.signed,"tx_hash":hash}),
                    )
                    .await;
                match result {
                    Ok(value) if value["accepted"] == true => {
                        attempt.state = PaymentState::Submitted;
                        attempt.error = None;
                    }
                    _ => {
                        attempt.state = PaymentState::Uncertain;
                        attempt.error = Some(
                            "broadcast outcome uncertain; reconcile the journaled transaction ID"
                                .into(),
                        );
                    }
                }
            }
            self.journal.update(&attempt)?;
        }
        Ok(attempt)
    }

    pub async fn lookup(&self, id: &str) -> Result<PaymentAttempt> {
        let _serial = self.serial.lock().await;
        self.resume(id, false).await
    }

    pub async fn verify(&self, request: VerifyRequest) -> Result<VerifyResult> {
        request.request.validate()?;
        ensure!(
            request.request.obligation.asset == asset(),
            "wrong verification asset"
        );
        ensure!(
            canonical_hash(&request.tx_hash),
            "invalid Ethereum transaction hash"
        );
        ensure!(
            canonical_address(&request.request.obligation.payer)
                && canonical_address(&request.request.obligation.recipient),
            "verification addresses must be canonical lowercase hex"
        );
        let result: VerifyResult = self.call("verify", serde_json::to_value(&request)?).await?;
        if result.state == PaymentState::Finalized {
            let receipt = result
                .receipt
                .as_ref()
                .context("finalized verification omitted receipt")?;
            let obligation = &request.request.obligation;
            ensure!(
                receipt.finalized
                    && receipt.tx_hash == request.tx_hash
                    && receipt.payer == obligation.payer
                    && receipt.recipient == obligation.recipient
                    && receipt.amount == obligation.amount
                    && receipt.memo == request.request.memo,
                "WDK receipt does not match immutable obligation"
            );
        }
        Ok(result)
    }
}

pub fn asset() -> Asset {
    Asset {
        chain_id: CHAIN_ID,
        token: USDT.into(),
        decimals: 6,
    }
}

pub fn fee_asset() -> Asset {
    Asset {
        chain_id: CHAIN_ID,
        token: "native:ETH".into(),
        decimals: 18,
    }
}

fn seconds() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

fn canonical_hash(value: &str) -> bool {
    canonical_hex(value, 64)
}

fn canonical_address(value: &str) -> bool {
    canonical_hex(value, 40)
}

fn canonical_hex(value: &str, digits: usize) -> bool {
    value.len() == digits + 2
        && value.starts_with("0x")
        && value[2..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

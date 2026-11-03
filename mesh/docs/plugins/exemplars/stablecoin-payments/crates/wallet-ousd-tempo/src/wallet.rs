use crate::{
    journal::Journal,
    keys::{self, Profile},
    receipt,
    rpc::{Rpc, address, ceil_fee, hex_u128, quantity},
};
use alloy::{
    eips::eip2718::Encodable2718,
    network::{EthereumWallet, NetworkTransactionBuilder, TransactionBuilder},
    primitives::{B256, U256, keccak256},
    signers::local::PrivateKeySigner,
    sol_types::SolCall,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    num::NonZeroU64,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tempo_alloy::{contracts::precompiles::ITIP20, rpc::TempoTransactionRequest};
use wallet_contract::{
    Asset, Balance, CHAIN_ID, FeeEstimate, METHOD, OUSD, PayRequest, PaymentAttempt,
    PaymentRequest, PaymentState, VerifyRequest, VerifyResult, WalletInfo,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub data_dir: PathBuf,
    pub rpc_url: String,
    pub password_file: Option<PathBuf>,
    #[serde(with = "wallet_contract::decimal")]
    pub max_payment: u64,
    #[serde(with = "wallet_contract::decimal")]
    pub max_fee: u64,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("wallet-ousd-tempo-data"),
            rpc_url: "https://rpc.tempo.xyz".into(),
            password_file: None,
            max_payment: 1_000_000,
            max_fee: 10_000,
        }
    }
}

pub struct Wallet {
    pub settings: Settings,
    pub profile: Profile,
    signer: Option<PrivateKeySigner>,
    pub rpc: Rpc,
    pub journal: Journal,
    send_lock: tokio::sync::Mutex<()>,
}
impl Wallet {
    pub fn load(mut settings: Settings) -> Result<Arc<Self>> {
        settings.data_dir = keys::canonical_state_dir(&settings.data_dir)?;
        let profile = keys::load_profile(&settings.data_dir)?;
        let signer = keys::unlock(
            &settings.data_dir,
            settings.password_file.as_deref(),
            &profile,
        )?;
        let journal = Journal::open(&settings.data_dir, &profile.wallet_id, &profile.address)?;
        let rpc = Rpc::new(settings.rpc_url.clone())?;
        Ok(Arc::new(Self {
            settings,
            profile,
            signer,
            rpc,
            journal,
            send_lock: tokio::sync::Mutex::new(()),
        }))
    }
    pub async fn info(&self) -> Result<WalletInfo> {
        self.rpc.check_network().await?;
        Ok(WalletInfo {
            wallet_id: self.profile.wallet_id.clone(),
            address: self.profile.address.clone(),
            asset: Asset::default(),
            method: METHOD.into(),
            can_send: self.signer.is_some(),
            fee_asset: None,
        })
    }
    pub async fn balance(&self) -> Result<Balance> {
        self.rpc.check_network().await?;
        let (balance, block_number) = self.rpc.balance(address(&self.profile.address)?).await?;
        let reserved = self.journal.reserved()?;
        Ok(Balance {
            balance,
            reserved,
            spendable: balance.saturating_sub(reserved),
            block_number,
        })
    }
    fn validate_request(&self, request: &PaymentRequest) -> Result<()> {
        request.validate()?;
        request.obligation.asset.validate_mainnet_ousd()?;
        wallet_contract::validate_address(&request.obligation.payer)?;
        wallet_contract::validate_address(&request.obligation.recipient)?;
        ensure!(
            address(&request.obligation.payer)? != address(&request.obligation.recipient)?,
            "self-payment refused"
        );
        ensure!(
            request
                .obligation
                .payer
                .eq_ignore_ascii_case(&self.profile.address),
            "obligation payer is not this wallet"
        );
        ensure!(
            request.obligation.amount <= self.settings.max_payment,
            "amount exceeds the wallet's local per-payment limit"
        );
        Ok(())
    }
    async fn transaction(
        &self,
        request: &PaymentRequest,
        nonce: u64,
    ) -> Result<(TempoTransactionRequest, FeeEstimate)> {
        self.validate_request(request)?;
        self.rpc.check_network().await?;
        ensure!(
            crate::rpc::hex_u64(
                &self
                    .rpc
                    .token_call("0x5c975abb".into(), "finalized")
                    .await?
            )? == 0,
            "OUSD transfers are paused"
        );
        let memo: B256 = request.memo.parse()?;
        let data = ITIP20::transferWithMemoCall {
            to: address(&request.obligation.recipient)?,
            amount: U256::from(request.obligation.amount),
            memo,
        }
        .abi_encode();
        let mut tx = TempoTransactionRequest::default()
            .with_chain_id(CHAIN_ID)
            .with_from(address(&self.profile.address)?)
            .with_to(address(OUSD)?)
            .with_value(U256::ZERO)
            .with_input(data)
            .with_nonce(nonce)
            .with_nonce_key(U256::ZERO)
            .with_fee_token(address(OUSD)?)
            .with_valid_before(
                NonZeroU64::new(request.obligation.expires_at).context("zero expiry")?,
            );
        let estimated = quantity(
            &self
                .rpc
                .call("eth_estimateGas", json!([serde_json::to_value(&tx)?]))
                .await?,
        )?;
        let gas_limit = estimated
            .checked_mul(120)
            .context("gas limit overflow")?
            .div_ceil(100);
        ensure!(
            gas_limit > 0 && gas_limit <= 2_000_000,
            "unreasonable transfer gas limit"
        );
        let price = hex_u128(&self.rpc.call("eth_gasPrice", json!([])).await?)?;
        ensure!(price > 0, "RPC reported zero gas price");
        let max_price = price.checked_mul(2).context("gas price overflow")?;
        tx = tx
            .with_gas_limit(gas_limit)
            .with_max_fee_per_gas(max_price)
            .with_max_priority_fee_per_gas(0);
        let fee = ceil_fee(gas_limit, max_price)?;
        let block = quantity(&self.rpc.finalized().await?["number"])?;
        Ok((
            tx,
            FeeEstimate {
                gas_limit,
                fee,
                fee_asset: Asset::default(),
                block_number: block,
            },
        ))
    }
    pub async fn estimate(&self, request: PaymentRequest) -> Result<FeeEstimate> {
        self.validate_request(&request)?;
        let nonce = quantity(
            &self
                .rpc
                .call(
                    "eth_getTransactionCount",
                    json!([self.profile.address, "latest"]),
                )
                .await?,
        )?;
        Ok(self.transaction(&request, nonce).await?.1)
    }
    pub async fn pay(&self, request: PayRequest) -> Result<PaymentAttempt> {
        let _send = self.send_lock.lock().await;
        let id = request.payment_id.clone();
        match tokio::time::timeout(Duration::from_secs(22), self.pay_locked(request)).await {
            Ok(result) => result,
            Err(_) => self.timed_out(&id),
        }
    }
    fn timed_out(&self, id: &str) -> Result<PaymentAttempt> {
        let mut attempt = self
            .journal
            .get(id)?
            .context(
                "payment preparation timed out before submission; no signed transfer was journaled",
            )?
            .attempt;
        if !matches!(
            attempt.state,
            PaymentState::Finalized
                | PaymentState::Reverted
                | PaymentState::Expired
                | PaymentState::Held
        ) {
            attempt.state = PaymentState::Uncertain;
            attempt.error = Some(
                "wallet deadline reached; reconcile this existing payment ID before any retry"
                    .into(),
            );
            self.journal.update(&attempt)?;
        }
        Ok(attempt)
    }
    async fn pay_locked(&self, request: PayRequest) -> Result<PaymentAttempt> {
        self.validate_request(&request.request)?;
        ensure!(
            !request.payment_id.is_empty() && request.payment_id.len() <= 128,
            "invalid payment ID"
        );
        if let Some(previous) = self.journal.get(&request.payment_id)? {
            ensure!(
                previous.request.request == request.request,
                "payment ID reused with different immutable economic fields"
            );
            return self.resume(&request.payment_id, true).await;
        }
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        ensure!(
            now < request.request.obligation.expires_at,
            "obligation expired before payment preparation"
        );
        let signer = self
            .signer
            .as_ref()
            .context("wallet is locked; configure its local password_file")?;
        // An uncertain nonce is never skipped to make a second transfer.
        for attempt in self.journal.unresolved()? {
            let _ = self.resume(&attempt.payment_id, false).await?;
        }
        ensure!(
            self.journal.unresolved()?.is_empty(),
            "an earlier outgoing payment is unresolved; reconcile it first"
        );
        let nonce = quantity(
            &self
                .rpc
                .call(
                    "eth_getTransactionCount",
                    json!([self.profile.address, "latest"]),
                )
                .await?,
        )?;
        let (tx, fee) = self.transaction(&request.request, nonce).await?;
        ensure!(
            fee.fee <= request.max_fee && fee.fee <= self.settings.max_fee,
            "estimated maximum fee exceeds an authorized fee limit"
        );
        let reserved = request
            .request
            .obligation
            .amount
            .checked_add(fee.fee)
            .context("total amount overflow")?;
        ensure!(
            reserved <= request.max_total,
            "amount plus maximum fee exceeds authorized total"
        );
        let (balance, _) = self.rpc.balance(address(&self.profile.address)?).await?;
        ensure!(
            balance >= reserved,
            "insufficient OUSD for transfer plus its maximum fee"
        );
        let envelope = tx.build(&EthereumWallet::from(signer.clone())).await?;
        let raw = envelope.encoded_2718();
        ensure!(
            raw.first() == Some(&0x76),
            "refusing to submit a non-Tempo transaction"
        );
        let hash = format!("{}", keccak256(&raw));
        let attempt = PaymentAttempt {
            payment_id: request.payment_id.clone(),
            request: request.request.clone(),
            state: PaymentState::Prepared,
            tx_hashes: vec![hash],
            receipt: None,
            error: None,
        };
        self.journal
            .insert(&request, &attempt, nonce, &raw, reserved)?;
        self.resume(&request.payment_id, true).await
    }
    /// A retry may rebroadcast only the original byte-for-byte signed payment.
    async fn resume(&self, id: &str, broadcast: bool) -> Result<PaymentAttempt> {
        let stored = self.journal.get(id)?.context("unknown payment ID")?;
        let mut attempt = stored.attempt;
        if matches!(
            attempt.state,
            PaymentState::Finalized | PaymentState::Reverted | PaymentState::Expired
        ) {
            return Ok(attempt);
        }
        let hash = attempt
            .tx_hashes
            .first()
            .context("journaled payment has no transaction hash")?
            .clone();
        let result = receipt::verify(&self.rpc, &attempt.request, &hash).await;
        if let Ok(result) = result {
            attempt.state = result.state;
            attempt.receipt = result.receipt;
            attempt.error = result.reason;
            self.journal.update(&attempt)?;
            if matches!(
                attempt.state,
                PaymentState::Finalized | PaymentState::Reverted | PaymentState::Held
            ) {
                return Ok(attempt);
            }
        }
        // A native validity deadline plus finalized nonce proves an absent
        // transaction can never execute. Local wall time alone cannot.
        let finalized = self.rpc.finalized().await?;
        let expiry = attempt.request.obligation.expires_at;
        if quantity(&finalized["timestamp"])? >= expiry {
            let tag = finalized["number"]
                .as_str()
                .context("finalized block lacks number")?;
            let nonce = quantity(
                &self
                    .rpc
                    .call(
                        "eth_getTransactionCount",
                        json!([self.profile.address, tag]),
                    )
                    .await?,
            )?;
            if nonce == stored.nonce {
                attempt.state = PaymentState::Expired;
                attempt.error = Some(
                    "native validity window expired without consuming the nonce at finalized state"
                        .into(),
                );
                self.journal.update(&attempt)?;
                return Ok(attempt);
            }
        }
        if broadcast {
            if SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() >= expiry {
                attempt.state = PaymentState::Uncertain;
                attempt.error=Some("not rebroadcasting after local expiry; awaiting finalized nonce reconciliation".into());
                self.journal.update(&attempt)?;
                return Ok(attempt);
            }
            // If an unknown transaction consumed this nonce, never allocate a new one.
            let current = quantity(
                &self
                    .rpc
                    .call(
                        "eth_getTransactionCount",
                        json!([self.profile.address, "latest"]),
                    )
                    .await?,
            )?;
            if current > stored.nonce {
                attempt.state = PaymentState::Uncertain;
                attempt.error=Some("nonce already consumed; waiting for the original canonical receipt (never automatically repaying)".into());
                self.journal.update(&attempt)?;
                return Ok(attempt);
            }
            match self
                .rpc
                .call(
                    "eth_sendRawTransaction",
                    json!([format!("0x{}", hex::encode(&stored.raw))]),
                )
                .await
            {
                Ok(value)
                    if value
                        .as_str()
                        .is_some_and(|v| v.eq_ignore_ascii_case(&hash)) =>
                {
                    attempt.state = PaymentState::Submitted;
                    attempt.error = None;
                }
                Ok(_) => {
                    attempt.state = PaymentState::Uncertain;
                    attempt.error = Some(
                        "RPC returned an unexpected transaction hash; reconcile the local hash"
                            .into(),
                    );
                }
                Err(error) => {
                    attempt.state = PaymentState::Uncertain;
                    attempt.error = Some(error.to_string());
                }
            }
            self.journal.update(&attempt)?;
            // Confirmation is bounded; later lookup resumes from the durable hash.
            for _ in 0..4 {
                tokio::time::sleep(Duration::from_millis(500)).await;
                match tokio::time::timeout(
                    Duration::from_secs(2),
                    receipt::verify(&self.rpc, &attempt.request, &hash),
                )
                .await
                {
                    Ok(Ok(result)) => {
                        attempt.state = result.state;
                        attempt.receipt = result.receipt;
                        attempt.error = result.reason;
                        self.journal.update(&attempt)?;
                        if matches!(
                            attempt.state,
                            PaymentState::Finalized | PaymentState::Reverted | PaymentState::Held
                        ) {
                            break;
                        }
                    }
                    Ok(Err(error)) => {
                        attempt.state = PaymentState::Uncertain;
                        attempt.error = Some(error.to_string());
                        self.journal.update(&attempt)?;
                        break;
                    }
                    Err(_) => {
                        attempt.state = PaymentState::Uncertain;
                        attempt.error = Some(
                            "confirmation check timed out; query the existing payment ID".into(),
                        );
                        self.journal.update(&attempt)?;
                        break;
                    }
                }
            }
        }
        Ok(attempt)
    }
    pub async fn lookup(&self, id: &str) -> Result<PaymentAttempt> {
        let _send = self.send_lock.lock().await;
        match tokio::time::timeout(Duration::from_secs(22), self.resume(id, false)).await {
            Ok(result) => result,
            Err(_) => self.timed_out(id),
        }
    }
    pub async fn verify(&self, request: VerifyRequest) -> Result<VerifyResult> {
        receipt::verify(&self.rpc, &request.request, &request.tx_hash).await
    }
}

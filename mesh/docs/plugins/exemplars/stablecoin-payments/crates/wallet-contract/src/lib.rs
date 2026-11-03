//! Stablecoin operations are opaque to the Mesh host.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CAPABILITY: &str = "wallet.stablecoin.v1";
pub const CHAIN_ID: u64 = 4217;
pub const OUSD: &str = "0x20c0000000000000000000006a37da5c996874be";
pub const METHOD: &str = "tempo.tip20-memo.v1";
pub const SCALE: u64 = 1_000_000;

pub mod ops {
    pub const OPEN: &str = "wallet_open";
    pub const BALANCE: &str = "wallet_balance";
    pub const ESTIMATE: &str = "wallet_estimate_payment";
    pub const PAY: &str = "wallet_pay";
    pub const LOOKUP: &str = "wallet_lookup";
    pub const VERIFY: &str = "wallet_verify_payment";
    pub const ATTEMPTS: &str = "wallet_list_attempts";
}

/// Never serialize native amounts as JSON floating-point numbers.
pub mod decimal {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};
    pub fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(D::Error::custom)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Asset {
    pub chain_id: u64,
    pub token: String,
    pub decimals: u8,
}
impl Default for Asset {
    fn default() -> Self {
        Self {
            chain_id: CHAIN_ID,
            token: OUSD.into(),
            decimals: 6,
        }
    }
}
impl Asset {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.chain_id != 0, "missing network identity");
        ensure!(
            !self.token.is_empty() && self.token.len() <= 256,
            "invalid asset identity"
        );
        ensure!(self.decimals <= 36, "unsupported token precision");
        Ok(())
    }
    /// The first concrete provider accepts only this explicitly funded rail.
    pub fn validate_mainnet_ousd(&self) -> Result<()> {
        self.validate()?;
        ensure!(self.chain_id == CHAIN_ID, "only Tempo mainnet is supported");
        ensure!(self.token.eq_ignore_ascii_case(OUSD), "unsupported token");
        ensure!(self.decimals == 6, "unexpected token precision");
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    Admission,
    Usage,
}

/// Fixed field order and a versioned domain define the memo encoding.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Obligation {
    pub version: u16,
    pub id: String,
    pub purpose: Purpose,
    pub buyer_peer: String,
    pub seller_peer: String,
    pub request_id: String,
    pub model: String,
    pub offer_id: String,
    pub asset: Asset,
    pub payer: String,
    pub recipient: String,
    #[serde(with = "decimal")]
    pub amount: u64,
    pub expires_at: u64,
}
impl Obligation {
    pub fn memo(&self) -> Result<String> {
        let mut h = Sha256::new();
        h.update(b"mesh-llm/stablecoin/obligation/v1\0");
        h.update(serde_json::to_vec(self)?);
        Ok(format!("0x{}", hex::encode(h.finalize())))
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.version == 1, "unsupported obligation version");
        self.asset.validate()?;
        ensure!(
            !self.id.is_empty() && self.id.len() <= 128,
            "invalid obligation ID"
        );
        ensure!(!self.request_id.is_empty(), "missing request ID");
        ensure!(self.amount > 0, "zero obligation requires no payment");
        ensure!(
            !self.payer.is_empty() && self.payer.len() <= 256,
            "invalid payer identity"
        );
        ensure!(
            !self.recipient.is_empty() && self.recipient.len() <= 256,
            "invalid recipient identity"
        );
        ensure!(self.payer != self.recipient, "self-payment refused");
        Ok(())
    }
}

pub fn validate_address(value: &str) -> Result<()> {
    let bytes = hex::decode(value.strip_prefix("0x").unwrap_or(value))?;
    ensure!(
        bytes.len() == 20 && bytes.iter().any(|b| *b != 0),
        "invalid account address"
    );
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PaymentRequest {
    pub obligation: Obligation,
    pub memo: String,
}
impl PaymentRequest {
    pub fn new(obligation: Obligation) -> Result<Self> {
        obligation.validate()?;
        Ok(Self {
            memo: obligation.memo()?,
            obligation,
        })
    }
    pub fn validate(&self) -> Result<()> {
        self.obligation.validate()?;
        ensure!(
            self.memo == self.obligation.memo()?,
            "memo does not match obligation"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletInfo {
    pub wallet_id: String,
    pub address: String,
    pub asset: Asset,
    pub method: String,
    pub can_send: bool,
    /// Absent means fees are paid in `asset`, preserving the original contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fee_asset: Option<Asset>,
}
impl WalletInfo {
    pub fn effective_fee_asset(&self) -> &Asset {
        self.fee_asset.as_ref().unwrap_or(&self.asset)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Balance {
    #[serde(with = "decimal")]
    pub balance: u64,
    #[serde(with = "decimal")]
    pub reserved: u64,
    #[serde(with = "decimal")]
    pub spendable: u64,
    pub block_number: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeeEstimate {
    pub gas_limit: u64,
    #[serde(with = "decimal")]
    pub fee: u64,
    pub fee_asset: Asset,
    pub block_number: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayRequest {
    pub payment_id: String,
    pub request: PaymentRequest,
    /// Maximum fee in the wallet's declared fee asset; zero-cap semantics are provider-owned.
    #[serde(with = "decimal")]
    pub max_fee: u64,
    /// Maximum payment-asset debit, including fees only when paid in that asset.
    #[serde(with = "decimal")]
    pub max_total: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LookupRequest {
    pub payment_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyRequest {
    pub request: PaymentRequest,
    pub tx_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PaymentState {
    Prepared,
    Submitted,
    Uncertain,
    Finalized,
    Reverted,
    Expired,
    Held,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub tx_hash: String,
    pub block_hash: String,
    pub block_number: u64,
    pub transfer_log_index: u64,
    pub payer: String,
    pub recipient: String,
    #[serde(with = "decimal")]
    pub amount: u64,
    pub memo: String,
    pub finalized: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaymentAttempt {
    pub payment_id: String,
    pub request: PaymentRequest,
    pub state: PaymentState,
    pub tx_hashes: Vec<String>,
    pub receipt: Option<Receipt>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyResult {
    pub state: PaymentState,
    pub receipt: Option<Receipt>,
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obligation() -> Obligation {
        Obligation {
            version: 1,
            id: "server-generated-unique-id".into(),
            purpose: Purpose::Usage,
            buyer_peer: "buyer".into(),
            seller_peer: "seller".into(),
            request_id: "request-1".into(),
            model: "demo-model".into(),
            offer_id: "offer-1".into(),
            asset: Asset::default(),
            payer: "0x1111111111111111111111111111111111111111".into(),
            recipient: "0x2222222222222222222222222222222222222222".into(),
            amount: 123,
            expires_at: 2_000_000_000,
        }
    }

    #[test]
    fn memo_commits_amount_peer_purpose_and_request() {
        let original = obligation();
        let expected = original.memo().unwrap();
        for changed in [
            Obligation {
                amount: 124,
                ..original.clone()
            },
            Obligation {
                buyer_peer: "another-buyer".into(),
                ..original.clone()
            },
            Obligation {
                purpose: Purpose::Admission,
                ..original.clone()
            },
            Obligation {
                request_id: "request-2".into(),
                ..original.clone()
            },
        ] {
            assert_ne!(changed.memo().unwrap(), expected);
        }
        let mut request = PaymentRequest::new(original).unwrap();
        request.obligation.amount += 1;
        assert!(request.validate().is_err());
    }

    #[test]
    fn wire_amounts_are_exact_decimal_strings() {
        let value = Balance {
            balance: u64::MAX,
            reserved: 1,
            spendable: 2,
            block_number: 3,
        };
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(encoded.contains("\"18446744073709551615\""));
        assert_eq!(
            serde_json::from_str::<Balance>(&encoded).unwrap().balance,
            u64::MAX
        );
        assert!(serde_json::from_str::<Balance>("{\"balance\":1.5}").is_err());
    }
    #[test]
    fn asset_identity_rejects_lookalike_tokens_and_testnet() {
        let mut asset = Asset::default();
        assert!(asset.validate_mainnet_ousd().is_ok());
        asset.chain_id = 42431;
        assert!(asset.validate_mainnet_ousd().is_err());
        assert!(validate_address("0x0000000000000000000000000000000000000000").is_err());
    }
    #[test]
    fn legacy_wallet_info_keeps_payment_asset_as_fee_currency() {
        let legacy = serde_json::json!({
            "wallet_id":"legacy","address":"account","asset":Asset::default(),
            "method":METHOD,"can_send":true
        });
        let info: WalletInfo = serde_json::from_value(legacy.clone()).unwrap();
        assert!(info.fee_asset.is_none());
        assert_eq!(info.effective_fee_asset(), &info.asset);
        assert_eq!(serde_json::to_value(&info).unwrap(), legacy);
        let explicit = WalletInfo {
            fee_asset: Some(info.asset.clone()),
            ..info
        };
        assert_eq!(explicit.effective_fee_asset(), &explicit.asset);
    }
}

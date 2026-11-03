use crate::config::Price;
use serde::{Deserialize, Serialize};
use wallet_contract::{PaymentRequest, Receipt, WalletInfo};

pub const CHANNEL: &str = "pay.stablecoin.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quote {
    pub request_id: String,
    pub offer_id: String,
    pub model: String,
    pub price: Price,
    pub wallet: WalletInfo,
    pub admission: Option<PaymentRequest>,
    #[serde(with = "wallet_contract::decimal")]
    pub first_credit: u64,
    pub expires_at: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Message {
    Hello {
        id: String,
        model: String,
        payer: String,
    },
    Quote {
        quote: Quote,
    },
    AdmissionProof {
        id: String,
        tx_hash: String,
    },
    Ready {
        id: String,
    },
    Bill {
        id: String,
        input: u64,
        output: u64,
        payment: Option<PaymentRequest>,
    },
    UsageProof {
        id: String,
        tx_hash: String,
    },
    Settled {
        id: String,
    },
    Cancel {
        id: String,
    },
    Error {
        id: String,
        reason: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Exchange {
    pub id: String,
    pub peer: String,
    pub model: String,
    pub payer: String,
    pub seller: bool,
    pub exchange_id: Option<String>,
    pub state: String,
    pub quote: Option<Quote>,
    pub usage: Option<(u64, u64)>,
    pub usage_payment: Option<PaymentRequest>,
    pub admission_receipt: Option<Receipt>,
    pub usage_receipt: Option<Receipt>,
    pub error: Option<String>,
}

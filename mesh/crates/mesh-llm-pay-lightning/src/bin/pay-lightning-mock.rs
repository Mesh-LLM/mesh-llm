//! Explicitly offline test wallet. Files model payments, never real funds.
use anyhow::{Context, Result};
use async_trait::async_trait;
use bitcoin::{
    hashes::{Hash, sha256},
    secp256k1::{Secp256k1, SecretKey},
};
use lightning_invoice::{Currency, InvoiceBuilder, PaymentHash, PaymentSecret};
use mesh_llm_wallet::{invoice::Invoice, provider::*, provisioning::WalletFactory};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

struct Offline(PathBuf, PathBuf);
#[async_trait]
impl WalletFactory for Offline {
    fn is_provisioned(&self, _: &Path) -> bool {
        true
    }
    async fn open(&self, directory: &Path) -> Result<Arc<dyn WalletProvider>> {
        std::fs::create_dir_all(&self.0)?;
        let owned = directory.join("offline-invoices");
        std::fs::create_dir_all(&owned)?;
        Ok(Arc::new(Self(self.0.clone(), owned)))
    }
}
#[async_trait]
impl WalletProvider for Offline {
    async fn balance(&self) -> Result<Balance> {
        Ok(Balance {
            spendable_msat: 1_000_000_000,
        })
    }
    async fn transactions(&self, _: usize) -> Result<Vec<Transaction>> {
        Ok(vec![])
    }
    async fn create_invoice(&self, amount: Option<u64>, expiry: u32) -> Result<Invoice> {
        let hash = sha256::Hash::hash(uuid::Uuid::new_v4().as_bytes()).to_byte_array();
        let secret = SecretKey::from_slice(&[7; 32])?;
        let builder = InvoiceBuilder::new(Currency::Bitcoin)
            .description("OFFLINE TEST ONLY".into())
            .payment_hash(PaymentHash(hash))
            .payment_secret(PaymentSecret([42; 32]))
            .current_timestamp()
            .expiry_time(std::time::Duration::from_secs(expiry.into()))
            .min_final_cltv_expiry_delta(144);
        let builder = if let Some(amount) = amount {
            builder.amount_milli_satoshis(amount)
        } else {
            builder
        };
        std::fs::write(self.1.join(hex_hash(&hash)), [])?;
        Invoice::parse(
            &builder
                .build_signed(|hash| Secp256k1::new().sign_ecdsa_recoverable(hash, &secret))?
                .to_string(),
        )
    }
    async fn pay(
        &self,
        invoice: &Invoice,
        amount: u64,
        cap: u64,
    ) -> std::result::Result<Transaction, PayError> {
        if amount > cap {
            return Err(PayError::NotSubmitted(anyhow::anyhow!("offline fee cap")));
        }
        let payment = Transaction {
            id: invoice.payment_hash.clone(),
            payment_hash: Some(invoice.payment_hash.clone()),
            inbound: false,
            amount_msat: amount,
            fee_msat: 0,
            status: PaymentStatus::Succeeded,
            claiming: false,
            status_msg: None,
            created_at_ms: mesh_llm_wallet::now_ms(),
            settled_at_ms: Some(mesh_llm_wallet::now_ms()),
        };
        let temporary = self.0.join(format!(".{}", uuid::Uuid::new_v4()));
        std::fs::write(
            &temporary,
            serde_json::to_vec(&payment).map_err(anyhow::Error::from)?,
        )
        .map_err(anyhow::Error::from)?;
        std::fs::rename(temporary, self.0.join(&invoice.payment_hash))
            .map_err(anyhow::Error::from)?;
        Ok(payment)
    }
    async fn lookup(&self, hash: &str) -> Result<Option<Transaction>> {
        anyhow::ensure!(
            hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid hash"
        );
        match std::fs::read(self.0.join(hash)) {
            Ok(bytes) => {
                let mut tx: Transaction = serde_json::from_slice(&bytes)?;
                tx.inbound = self.1.join(hash).is_file();
                Ok(Some(tx))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    async fn wait_for_payment(&self, hash: &str) -> Result<Transaction> {
        self.lookup(hash).await?.context("offline payment absent")
    }
}
fn hex_hash(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}
#[tokio::main]
async fn main() -> Result<()> {
    if std::env::args().any(|arg| arg == "--manifest") {
        writeln!(
            mesh_llm_events::machine_out(),
            "{}",
            mesh_llm_pay_lightning::manifest_json("pay-lightning")?
        )?;
        return Ok(());
    }
    let name = std::env::var("MESH_LLM_PLUGIN_NAME").context("launch via plugin host")?;
    let config = mesh_llm_pay_lightning::Config::from_launch_env()?;
    let network = PathBuf::from(
        std::env::var_os("MESH_LLM_OFFLINE_NETWORK_DIR")
            .context("explicit offline network directory required")?,
    );
    anyhow::ensure!(
        network.is_absolute(),
        "offline network directory must be absolute"
    );
    mesh_llm_pay_lightning::run(
        name,
        env!("CARGO_PKG_VERSION"),
        Offline(network, PathBuf::new()),
        config,
    )
    .await
}

use anyhow::{Context, Result, bail, ensure};
use mesh_llm_plugin::{
    PluginContext,
    operation_binding::{BoundOperationRequest, BoundOperationResponse, METHOD},
};
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;
use tokio::sync::Mutex;
use wallet_contract::{WalletInfo, ops};

pub trait WalletTransport: Send + Sync {
    fn execute<'a>(
        &'a self,
        operation: &'a str,
        input: serde_json::Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send + 'a>>;
}

#[derive(Clone)]
pub struct Wallet {
    pub info: WalletInfo,
    transport: Arc<dyn WalletTransport>,
}
struct HostWallet {
    binding: String,
    context: PluginContext<'static>,
    handle: Arc<Mutex<String>>,
}
impl Wallet {
    pub async fn open(binding: String, context: PluginContext<'static>) -> Result<Self> {
        let (handle, output) = setup(&context, &binding).await?;
        let info: WalletInfo = serde_json::from_str(&output)?;
        info.asset.validate()?;
        info.effective_fee_asset().validate()?;
        validate_identifier(&info.wallet_id)?;
        validate_identifier(&info.address)?;
        ensure!(
            !info.method.is_empty() && info.method.len() <= 128,
            "invalid wallet payment method"
        );
        Ok(Self {
            info,
            transport: Arc::new(HostWallet {
                binding,
                context,
                handle: Arc::new(Mutex::new(handle)),
            }),
        })
    }
    pub async fn call<I: Serialize, O: DeserializeOwned>(
        &self,
        operation: &str,
        input: I,
    ) -> Result<O> {
        serde_json::from_value(
            self.transport
                .execute(operation, serde_json::to_value(input)?)
                .await?,
        )
        .context("wallet returned an invalid response")
    }
    pub async fn balance(&self) -> Result<wallet_contract::Balance> {
        self.call(ops::BALANCE, serde_json::json!({})).await
    }
    #[cfg(test)]
    pub(crate) fn fixture(info: WalletInfo, transport: Arc<dyn WalletTransport>) -> Self {
        Self { info, transport }
    }
}
pub fn validate_identifier(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control),
        "invalid wallet account identifier"
    );
    Ok(())
}
impl WalletTransport for HostWallet {
    fn execute<'a>(
        &'a self,
        operation: &'a str,
        input: serde_json::Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send + 'a>>
    {
        Box::pin(async move {
            let handle = self.handle.lock().await.clone();
            let (_, output) = invoke(
                &self.context,
                BoundOperationRequest {
                    binding: self.binding.clone(),
                    handle: Some(handle),
                    operation: operation.into(),
                    input_json: serde_json::to_string(&input)?,
                },
            )
            .await?;
            Ok(serde_json::from_str(&output)?)
        })
    }
}
async fn invoke(
    context: &PluginContext<'_>,
    request: BoundOperationRequest,
) -> Result<(String, String)> {
    interpret(response(context, request).await?)
}
async fn response(
    context: &PluginContext<'_>,
    request: BoundOperationRequest,
) -> Result<BoundOperationResponse> {
    tokio::time::timeout(
        std::time::Duration::from_secs(31),
        context.request_host::<_, BoundOperationResponse>(METHOD, request),
    )
    .await
    .context("wallet host call timed out; inspect wallet attempts before retry")?
}
fn interpret(response: BoundOperationResponse) -> Result<(String, String)> {
    match response {
        BoundOperationResponse::Responded {
            handle,
            output_json,
            is_error: false,
        } => Ok((handle, output_json)),
        BoundOperationResponse::Responded { output_json, .. } => {
            bail!("wallet refused operation: {output_json}")
        }
        BoundOperationResponse::NotDispatched { reason } => {
            bail!("wallet operation not dispatched: {reason}")
        }
        BoundOperationResponse::Uncertain { reason } => {
            bail!("wallet operation uncertain; reconcile before retry: {reason}")
        }
    }
}
async fn setup(context: &PluginContext<'_>, binding: &str) -> Result<(String, String)> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        let result = response(
            context,
            BoundOperationRequest {
                binding: binding.into(),
                handle: None,
                operation: String::new(),
                input_json: String::new(),
            },
        )
        .await?;
        match result {
            BoundOperationResponse::NotDispatched { reason }
                if tokio::time::Instant::now() < deadline =>
            {
                tracing::debug!("waiting for stablecoin wallet provider: {reason}");
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
            outcome => return interpret(outcome),
        }
    }
}

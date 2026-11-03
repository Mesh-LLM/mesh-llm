use alloy::primitives::{Address, U256};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::time::Duration;
use wallet_contract::{CHAIN_ID, OUSD};

#[derive(Clone)]
pub struct Rpc {
    client: reqwest::Client,
    url: String,
}

impl Rpc {
    pub fn new(url: String) -> Result<Self> {
        let parsed = reqwest::Url::parse(&url)?;
        ensure!(
            parsed.scheme() == "https"
                || (parsed.scheme() == "http"
                    && matches!(parsed.host_str(), Some("127.0.0.1" | "localhost" | "::1"))),
            "RPC must use HTTPS (loopback HTTP is allowed for local tests)"
        );
        ensure!(
            parsed.username().is_empty() && parsed.password().is_none(),
            "RPC credentials must not be embedded in a URL"
        );
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(8))
                .connect_timeout(Duration::from_secs(3))
                .build()?,
            url,
        })
    }
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let response = self
            .client
            .post(&self.url)
            .json(&json!({"jsonrpc":"2.0","id":1,"method":method,"params":params}))
            .send()
            .await
            .context("RPC transport failure")?;
        ensure!(
            response.status().is_success(),
            "RPC HTTP status {}",
            response.status()
        );
        let body: Value = response.json().await.context("invalid RPC response")?;
        if let Some(error) = body.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("RPC rejected request");
            bail!(
                "{}: {}",
                method,
                message.chars().take(240).collect::<String>()
            );
        }
        body.get("result")
            .cloned()
            .context("RPC response has no result")
    }
    pub async fn finalized(&self) -> Result<Value> {
        let block = self
            .call("eth_getBlockByNumber", json!(["finalized", false]))
            .await?;
        ensure!(block.is_object(), "RPC did not provide finalized state");
        Ok(block)
    }
    pub async fn token_call(&self, data: String, tag: &str) -> Result<String> {
        self.call("eth_call", json!([{"to":OUSD,"data":data},tag]))
            .await?
            .as_str()
            .map(str::to_owned)
            .context("invalid token call response")
    }
    pub async fn check_network(&self) -> Result<()> {
        ensure!(
            quantity(&self.call("eth_chainId", json!([])).await?)? == CHAIN_ID,
            "RPC is not Tempo mainnet (chain 4217)"
        );
        let code = self.call("eth_getCode", json!([OUSD, "finalized"])).await?;
        ensure!(
            code.as_str().is_some_and(|x| x != "0x" && x != "0x0"),
            "OUSD token is absent on RPC network"
        );
        ensure!(
            hex_u64(&self.token_call("0x313ce567".into(), "finalized").await?)? == 6,
            "OUSD precision is not six decimals"
        );
        Ok(())
    }
    pub async fn balance(&self, address: Address) -> Result<(u64, u64)> {
        let block = self.finalized().await?;
        let height = quantity(&block["number"])?;
        let data = format!("0x70a08231{:0>64}", hex::encode(address));
        let result = self.token_call(data, &format!("0x{height:x}")).await?;
        let balance = U256::from_str_radix(result.trim_start_matches("0x"), 16)?;
        Ok((
            u64::try_from(balance).context("wallet balance exceeds supported u64 native units")?,
            height,
        ))
    }
}

pub fn quantity(value: &Value) -> Result<u64> {
    hex_u64(value.as_str().context("expected RPC hex quantity")?)
}
pub fn hex_u64(value: &str) -> Result<u64> {
    u64::from_str_radix(
        value
            .strip_prefix("0x")
            .context("quantity lacks 0x prefix")?,
        16,
    )
    .context("invalid/oversized RPC quantity")
}
pub fn hex_u128(value: &Value) -> Result<u128> {
    u128::from_str_radix(
        value
            .as_str()
            .context("expected RPC quantity")?
            .strip_prefix("0x")
            .context("quantity lacks 0x prefix")?,
        16,
    )
    .context("invalid/oversized RPC quantity")
}
pub fn address(value: &str) -> Result<Address> {
    wallet_contract::validate_address(value)?;
    Ok(value.parse()?)
}
pub fn ceil_fee(gas: u64, price: u128) -> Result<u64> {
    let raw = u128::from(gas)
        .checked_mul(price)
        .context("gas fee overflow")?;
    u64::try_from(raw.div_ceil(1_000_000_000_000)).context("fee exceeds supported native units")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attodollars_round_up_to_micro_ousd() {
        assert_eq!(ceil_fee(47_002, 600_000_000).unwrap(), 29);
        assert_eq!(ceil_fee(0, 600_000_000).unwrap(), 0);
        assert_eq!(ceil_fee(1, 1).unwrap(), 1);
        assert!(quantity(&json!("0xffffffffffffffffffff")).is_err());
    }
}

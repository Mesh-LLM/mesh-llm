use crate::rpc::{Rpc, quantity};
use alloy::primitives::keccak256;
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use wallet_contract::{PaymentRequest, PaymentState, Receipt, VerifyResult};

fn waiting(reason: &str) -> VerifyResult {
    VerifyResult {
        state: PaymentState::Submitted,
        receipt: None,
        reason: Some(reason.into()),
    }
}
fn equal(a: &Value, b: &str) -> bool {
    a.as_str().is_some_and(|a| a.eq_ignore_ascii_case(b))
}
fn topic_address(value: &Value, address: &str) -> bool {
    let Some(encoded) = value.as_str().and_then(|v| v.strip_prefix("0x")) else {
        return false;
    };
    let Ok(word) = hex::decode(encoded) else {
        return false;
    };
    let Ok(expected) = hex::decode(address.strip_prefix("0x").unwrap_or(address)) else {
        return false;
    };
    word.len() == 32
        && expected.len() == 20
        && word[..12].iter().all(|b| *b == 0)
        && word[12..] == expected
}

pub async fn verify(rpc: &Rpc, request: &PaymentRequest, tx_hash: &str) -> Result<VerifyResult> {
    request.validate()?;
    request.obligation.asset.validate_mainnet_ousd()?;
    wallet_contract::validate_address(&request.obligation.payer)?;
    wallet_contract::validate_address(&request.obligation.recipient)?;
    ensure!(
        crate::rpc::address(&request.obligation.payer)?
            != crate::rpc::address(&request.obligation.recipient)?,
        "self-payment evidence refused"
    );
    ensure!(
        tx_hash.len() == 66 && tx_hash.starts_with("0x") && hex::decode(&tx_hash[2..]).is_ok(),
        "invalid transaction hash"
    );
    rpc.check_network().await?;
    let receipt = rpc
        .call("eth_getTransactionReceipt", serde_json::json!([tx_hash]))
        .await?;
    if receipt.is_null() {
        return Ok(waiting("transaction has no canonical receipt yet"));
    }
    ensure!(
        equal(&receipt["transactionHash"], tx_hash),
        "RPC returned another transaction receipt"
    );
    let height = quantity(&receipt["blockNumber"])?;
    let canonical = rpc
        .call(
            "eth_getBlockByNumber",
            serde_json::json!([format!("0x{height:x}"), false]),
        )
        .await?;
    let finalized = rpc.finalized().await?;
    verify_data(
        request,
        tx_hash,
        &receipt,
        &canonical,
        quantity(&finalized["number"])?,
    )
}

pub fn verify_data(
    request: &PaymentRequest,
    tx_hash: &str,
    receipt: &Value,
    canonical: &Value,
    finalized_height: u64,
) -> Result<VerifyResult> {
    request.validate()?;
    request.obligation.asset.validate_mainnet_ousd()?;
    wallet_contract::validate_address(&request.obligation.payer)?;
    wallet_contract::validate_address(&request.obligation.recipient)?;
    ensure!(
        crate::rpc::address(&request.obligation.payer)?
            != crate::rpc::address(&request.obligation.recipient)?,
        "self-payment evidence refused"
    );
    let height = quantity(&receipt["blockNumber"])?;
    let block_hash = receipt["blockHash"]
        .as_str()
        .context("receipt lacks block hash")?;
    if !equal(&canonical["hash"], block_hash) {
        return Ok(VerifyResult {
            state: PaymentState::Uncertain,
            receipt: None,
            reason: Some("receipt block is not canonical".into()),
        });
    }
    if height > finalized_height {
        return Ok(waiting("waiting for finalized Tempo block"));
    }
    if quantity(&receipt["status"])? == 0 {
        return Ok(VerifyResult {
            state: PaymentState::Reverted,
            receipt: None,
            reason: Some("canonical transaction reverted; gas may have been charged".into()),
        });
    }
    ensure!(quantity(&receipt["status"])? == 1, "invalid receipt status");
    ensure!(
        equal(&receipt["from"], &request.obligation.payer),
        "receipt sender does not match obligation payer"
    );
    let logs = receipt["logs"]
        .as_array()
        .context("receipt has no logs array")?;
    let transfer_topic = format!("{}", keccak256("Transfer(address,address,uint256)"));
    let memo_topic = format!(
        "{}",
        keccak256("TransferWithMemo(address,address,uint256,bytes32)")
    );
    let matches_delivery = |log: &&Value| -> bool {
        equal(&log["address"], wallet_contract::OUSD)
            && log.get("removed").and_then(Value::as_bool) != Some(true)
            && topic_address(&log["topics"][1], &request.obligation.payer)
            && topic_address(&log["topics"][2], &request.obligation.recipient)
            && log["data"]
                .as_str()
                .and_then(|d| {
                    alloy::primitives::U256::from_str_radix(d.trim_start_matches("0x"), 16).ok()
                })
                .is_some_and(|v| v == alloy::primitives::U256::from(request.obligation.amount))
    };
    let transfer = logs.iter().filter(matches_delivery).find(|l| {
        equal(&l["topics"][0], &transfer_topic)
            && l["topics"].as_array().is_some_and(|t| t.len() == 3)
    });
    let memo = logs.iter().filter(matches_delivery).find(|l| {
        equal(&l["topics"][0], &memo_topic)
            && l["topics"].as_array().is_some_and(|t| t.len() == 4)
            && equal(&l["topics"][3], &request.memo)
    });
    let (Some(transfer), Some(_)) = (transfer, memo) else {
        return Ok(VerifyResult {state:PaymentState::Held,receipt:None,reason:Some("no matching actual OUSD delivery and memo; funds may be held by receiver policy or evidence is invalid".into())});
    };
    Ok(VerifyResult {
        state: PaymentState::Finalized,
        reason: None,
        receipt: Some(Receipt {
            tx_hash: tx_hash.to_ascii_lowercase(),
            block_hash: block_hash.to_ascii_lowercase(),
            block_number: height,
            transfer_log_index: quantity(&transfer["logIndex"])?,
            payer: request.obligation.payer.clone(),
            recipient: request.obligation.recipient.clone(),
            amount: request.obligation.amount,
            memo: request.memo.clone(),
            finalized: true,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wallet_contract::{Asset, Obligation, Purpose};
    fn fixture() -> (PaymentRequest, Value, Value) {
        let payer = "0x1111111111111111111111111111111111111111";
        let recipient = "0x2222222222222222222222222222222222222222";
        let request = PaymentRequest::new(Obligation {
            version: 1,
            id: "fixture".into(),
            purpose: Purpose::Usage,
            buyer_peer: "buyer".into(),
            seller_peer: "seller".into(),
            request_id: "req".into(),
            model: "model".into(),
            offer_id: "offer".into(),
            asset: Asset::default(),
            payer: payer.into(),
            recipient: recipient.into(),
            amount: 7,
            expires_at: u64::MAX,
        })
        .unwrap();
        let from = format!("0x{:0>64}", &payer[2..]);
        let to = format!("0x{:0>64}", &recipient[2..]);
        let receipt = json!({"transactionHash":format!("0x{}","a".repeat(64)),"blockNumber":"0xa","blockHash":"0xblock","status":"0x1","from":payer,"logs":[
          {"address":wallet_contract::OUSD,"topics":[format!("{}",keccak256("Transfer(address,address,uint256)")),from,to],"data":format!("0x{:064x}",7),"logIndex":"0x2"},
          {"address":wallet_contract::OUSD,"topics":[format!("{}",keccak256("TransferWithMemo(address,address,uint256,bytes32)")),from,to,request.memo],"data":format!("0x{:064x}",7),"logIndex":"0x3"}]});
        (request, receipt, json!({"hash":"0xblock"}))
    }
    #[test]
    fn requires_finalized_delivery_and_exact_memo() {
        let (request, mut receipt, canonical) = fixture();
        assert_eq!(
            verify_data(&request, "tx", &receipt, &canonical, 10)
                .unwrap()
                .state,
            PaymentState::Finalized
        );
        assert_eq!(
            verify_data(&request, "tx", &receipt, &canonical, 9)
                .unwrap()
                .state,
            PaymentState::Submitted
        );
        receipt["logs"][1]["topics"][3] = json!(format!("0x{}", "0".repeat(64)));
        assert_eq!(
            verify_data(&request, "tx", &receipt, &canonical, 10)
                .unwrap()
                .state,
            PaymentState::Held
        );
    }
    #[test]
    fn rejects_wrong_token_and_reorgs() {
        let (request, mut receipt, canonical) = fixture();
        assert_eq!(
            verify_data(&request, "tx", &receipt, &json!({"hash":"different"}), 10)
                .unwrap()
                .state,
            PaymentState::Uncertain
        );
        receipt["logs"][0]["address"] = json!("0x3333333333333333333333333333333333333333");
        assert_eq!(
            verify_data(&request, "tx", &receipt, &canonical, 10)
                .unwrap()
                .state,
            PaymentState::Held
        );
    }
    #[test]
    fn memo_without_actual_delivery_and_malformed_topics_never_settle() {
        let (request, mut receipt, canonical) = fixture();
        receipt["logs"].as_array_mut().unwrap().remove(0);
        assert_eq!(
            verify_data(&request, "tx", &receipt, &canonical, 10)
                .unwrap()
                .state,
            PaymentState::Held
        );
        let (request, mut receipt, canonical) = fixture();
        receipt["logs"][0]["topics"][1] = json!(format!("0x{}{}0", "0".repeat(23), "é".repeat(20)));
        assert_eq!(
            verify_data(&request, "tx", &receipt, &canonical, 10)
                .unwrap()
                .state,
            PaymentState::Held
        );
        assert!(!topic_address(
            &json!("0000000000000000000000001111111111111111111111111111111111111111"),
            &request.obligation.payer
        ));
    }
}

//! A loopback-only JSON-RPC fixture exercises the complete signer/journal/broadcast path.
use crate::{
    keys::Profile,
    wallet::{Settings, Wallet},
};
use alloy::{
    primitives::{B256, keccak256},
    signers::local::PrivateKeySigner,
};
use rand::SeedableRng;
use serde_json::{Value, json};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};
use wallet_contract::{
    Asset, CHAIN_ID, OUSD, Obligation, PayRequest, PaymentRequest, PaymentState, Purpose,
};

struct Fixture {
    request: PaymentRequest,
    hash: Mutex<Option<String>>,
    broadcasts: AtomicUsize,
}
impl Fixture {
    fn result(&self, method: &str, params: &Value) -> Value {
        match method {
            "eth_chainId" => json!("0x1079"),
            "eth_getCode" => json!("0xef"),
            "eth_call" => match params[0]["data"].as_str().unwrap_or("") {
                "0x313ce567" => json!("0x6"),
                "0x5c975abb" => json!("0x0"),
                _ => json!("0xf4240"),
            },
            "eth_estimateGas" => {
                assert_eq!(
                    params[0]["feeToken"].as_str().unwrap().to_ascii_lowercase(),
                    OUSD.to_ascii_lowercase()
                );
                assert!(params[0].get("validBefore").is_some());
                json!("0xc350")
            }
            "eth_gasPrice" => json!("0x23c34600"),
            "eth_getBlockByNumber" => {
                json!({"number":"0xa","hash":format!("0x{}","b".repeat(64)),"timestamp":format!("0x{:x}",SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())})
            }
            "eth_getTransactionCount" => json!(if self.hash.lock().unwrap().is_some() {
                "0x1"
            } else {
                "0x0"
            }),
            "eth_sendRawTransaction" => {
                let raw =
                    hex::decode(params[0].as_str().unwrap().trim_start_matches("0x")).unwrap();
                assert_eq!(raw[0], 0x76);
                let hash = keccak256(raw).to_string();
                self.broadcasts.fetch_add(1, Ordering::SeqCst);
                *self.hash.lock().unwrap() = Some(hash.clone());
                json!(hash)
            }
            "eth_getTransactionReceipt" => {
                let Some(hash) = self.hash.lock().unwrap().clone() else {
                    return Value::Null;
                };
                let payer = &self.request.obligation.payer;
                let recipient = &self.request.obligation.recipient;
                let from = format!("0x{:0>64}", &payer[2..]);
                let to = format!("0x{:0>64}", &recipient[2..]);
                let amount = format!("0x{:064x}", self.request.obligation.amount);
                json!({"transactionHash":hash,"blockNumber":"0xa","blockHash":format!("0x{}","b".repeat(64)),"status":"0x1","from":payer,"logs":[
                    {"address":OUSD,"topics":[keccak256("Transfer(address,address,uint256)").to_string(),from,to],"data":amount,"logIndex":"0x2"},
                    {"address":OUSD,"topics":[keccak256("TransferWithMemo(address,address,uint256,bytes32)").to_string(),from,to,self.request.memo],"data":amount,"logIndex":"0x3"}]})
            }
            _ => panic!("unexpected fixture RPC {method}"),
        }
    }
}

async fn serve_connection(mut stream: TcpStream, fixture: Arc<Fixture>) {
    let mut buffer = Vec::new();
    let mut scratch = [0u8; 4096];
    let (header_end, content_length) = loop {
        let read = stream.read(&mut scratch).await.unwrap();
        assert!(read > 0);
        buffer.extend_from_slice(&scratch[..read]);
        assert!(buffer.len() < 128 * 1024);
        if let Some(end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            let header = std::str::from_utf8(&buffer[..end]).unwrap();
            let length = header
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            break (end + 4, length);
        }
    };
    while buffer.len() < header_end + content_length {
        let read = stream.read(&mut scratch).await.unwrap();
        assert!(read > 0);
        buffer.extend_from_slice(&scratch[..read]);
    }
    let request: Value =
        serde_json::from_slice(&buffer[header_end..header_end + content_length]).unwrap();
    let result = fixture.result(request["method"].as_str().unwrap(), &request["params"]);
    let body =
        serde_json::to_vec(&json!({"jsonrpc":"2.0","id":request["id"],"result":result})).unwrap();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await.unwrap();
    stream.write_all(&body).await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn full_local_payment_then_restart_reconciles_without_second_broadcast() {
    use std::os::unix::fs::PermissionsExt;
    let signer = PrivateKeySigner::from_bytes(&B256::repeat_byte(1)).unwrap();
    let request = PaymentRequest::new(Obligation {
        version: 1,
        id: "fixture-obligation".into(),
        purpose: Purpose::Usage,
        buyer_peer: "buyer".into(),
        seller_peer: "seller".into(),
        request_id: "request".into(),
        model: "fixture".into(),
        offer_id: "offer".into(),
        asset: Asset::default(),
        payer: signer.address().to_string(),
        recipient: "0x2222222222222222222222222222222222222222".into(),
        amount: 7,
        expires_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 600,
    })
    .unwrap();
    let fixture = Arc::new(Fixture {
        request: request.clone(),
        hash: Mutex::new(None),
        broadcasts: AtomicUsize::new(0),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let rpc_url = format!("http://{}", listener.local_addr().unwrap());
    let fixture_server = fixture.clone();
    let server = tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            let fixture = fixture_server.clone();
            tokio::spawn(serve_connection(stream, fixture));
        }
    });
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let password_file = dir.path().join("credential");
    std::fs::write(&password_file, b"fixture-password-not-for-funding").unwrap();
    std::fs::set_permissions(&password_file, std::fs::Permissions::from_mode(0o600)).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    let key_file = String::from("fixture-key.json");
    let (_, _uuid) = PrivateKeySigner::encrypt_keystore(
        dir.path(),
        &mut rng,
        signer.to_bytes(),
        b"fixture-password-not-for-funding",
        Some(&key_file),
    )
    .unwrap();
    std::fs::set_permissions(
        dir.path().join(&key_file),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    let profile = Profile {
        wallet_id: "fixture".into(),
        address: signer.address().to_string(),
        chain_id: CHAIN_ID,
        token: OUSD.into(),
        key_file,
    };
    std::fs::write(
        dir.path().join("wallet.json"),
        serde_json::to_vec(&profile).unwrap(),
    )
    .unwrap();
    let settings = Settings {
        data_dir: dir.path().into(),
        rpc_url,
        password_file: Some(password_file),
        ..Default::default()
    };
    let wallet = Wallet::load(settings.clone()).unwrap();
    let payment = PayRequest {
        payment_id: "fixture-payment".into(),
        request,
        max_fee: 1_000,
        max_total: 1_007,
    };
    let mut refused = payment.clone();
    refused.max_fee = 0;
    assert!(wallet.pay(refused).await.is_err());
    assert_eq!(fixture.broadcasts.load(Ordering::SeqCst), 0);
    assert!(wallet.journal.list().unwrap().is_empty());
    let paid = wallet.pay(payment.clone()).await.unwrap();
    assert_eq!(paid.state, PaymentState::Finalized);
    assert!(paid.receipt.as_ref().unwrap().finalized);
    assert_eq!(fixture.broadcasts.load(Ordering::SeqCst), 1);
    drop(wallet);
    let reopened = Wallet::load(settings).unwrap();
    let recovered = reopened.pay(payment).await.unwrap();
    assert_eq!(recovered.state, PaymentState::Finalized);
    assert_eq!(recovered.tx_hashes, paid.tx_hashes);
    assert_eq!(fixture.broadcasts.load(Ordering::SeqCst), 1);
    server.abort();
}

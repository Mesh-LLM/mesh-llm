//! Payment interoperability through two differently named, real wallet.v1 children.
#![cfg(all(unix, feature = "offline-mock"))]
use mesh_llm_pay_lightning::{Engine, Message, wallet_child::ChildWalletFactory};
use mesh_llm_payments::{
    ledger::{ApprovalMode, Policy},
    service::PaymentService,
};
use mesh_llm_payments_types::pricing::Pricing;
use mesh_llm_wallet::provisioning::{WalletFactory, WalletPin};
use serde_json::json;
use std::{path::Path, sync::Arc};

async fn engine(root: &Path, network: &Path, name: &str, buyer: bool) -> Engine {
    // A wrapper supplies only synthetic test-network configuration to this child.
    let executable = root.join("wallet.sh");
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(&executable, format!(
        "#!/bin/sh\nexport MESH_LLM_OFFLINE_WALLET_CHILD=1\nexport MESH_LLM_OFFLINE_NETWORK_DIR='{}'\nexec '{}'\n",
        network.display(), env!("CARGO_BIN_EXE_pay-lightning-mock"),
    )).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let factory = ChildWalletFactory {
        executable,
        plugin: name.into(),
        directory: root.join("wallet"),
    };
    let service =
        Arc::new(PaymentService::with_factory(&root.join("ledger"), Arc::new(factory)).unwrap());
    let pricing = Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        cache_discount_percent: 50,
    };
    service.ledger.set_pricing("model", Some(&pricing)).unwrap();
    service
        .ledger
        .set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(100_000),
        })
        .unwrap();
    let mut engine = Engine::new(
        service,
        "7061792d6c696768746e696e67",
        if buyer { 100_000 } else { 0 },
        &root.join("ledger"),
    )
    .unwrap();
    engine.buying.insert("model".into(), pricing);
    engine.buy_peers.insert("seller".into());
    engine.start().await.unwrap();
    engine
}

#[tokio::test]
async fn distinct_wallet_children_settle_the_common_payment_protocol() {
    let root = tempfile::Builder::new()
        .prefix("ln-child-")
        .tempdir_in("/tmp")
        .unwrap();
    let network = root.path().join("network");
    let seller = engine(&root.path().join("seller"), &network, "wallet-alpha", false).await;
    let buyer = engine(&root.path().join("buyer"), &network, "wallet-beta", true).await;
    let admission = seller
        .channel("buyer", Message::Quote)
        .await
        .unwrap()
        .remove(0);
    let paid = buyer.channel("seller", admission).await.unwrap().remove(0);
    seller.channel("buyer", paid).await.unwrap();
    let decision = seller
        .gate(
            &json!({"phase":"request_received", "observation_point":"serving_host_ingress",
        "model":"model", "requested_by_node_id":"buyer", "exchange_id":"cross-wallet"}),
        )
        .unwrap();
    assert_eq!(
        decision.decision,
        mesh_llm_plugin::openai_exchange::OpenAiAdmissionDecision::Allow
    );
    let (_, bill) = seller
        .terminal(
            &json!({"phase":"exchange_finished", "ingress_observation_point":"serving_host_ingress",
        "exchange_id":"cross-wallet", "execution_outcome":"success", "status":200,
        "usage":{"prompt_tokens":2000,"completion_tokens":1000}}),
        )
        .await
        .unwrap()
        .unwrap();
    let paid = buyer.channel("seller", bill).await.unwrap().remove(0);
    seller.channel("buyer", paid).await.unwrap();
    assert_eq!(
        seller
            .service
            .ledger
            .metered_unpaid_finals("buyer")
            .unwrap(),
        0
    );
    for (role, name) in [("seller", "wallet-alpha"), ("buyer", "wallet-beta")] {
        let pin = WalletPin::load(&root.path().join(role).join("ledger"))
            .unwrap()
            .unwrap();
        assert_eq!(pin.plugin, name);
        assert_eq!(
            WalletPin::load(&root.path().join(role).join("wallet")).unwrap(),
            Some(pin)
        );
    }
}

#[tokio::test]
async fn child_factory_discovery_never_spawns_or_creates_state() {
    let root = tempfile::tempdir().unwrap();
    let factory = ChildWalletFactory {
        executable: "/not/a/wallet".into(),
        plugin: "not-installed".into(),
        directory: root.path().join("absent"),
    };
    assert!(!factory.is_provisioned(root.path()));
    assert!(!factory.directory.exists());
}

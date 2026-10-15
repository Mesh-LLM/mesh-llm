//! Old hosts bypass the contract check: exercise the real provider IPC decoder.
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mesh_llm_payments_types::contract::ops;
use mesh_llm_plugin::PluginRuntime;
use serde_json::json;

use crate::plugin::{
    InProcessPluginRunner, InProcessPlugins, PluginHostMode, PluginManager, ResolvedPlugins,
};

#[tokio::test]
async fn old_admissions_are_refused_before_opening_the_engine() {
    let opens = Arc::new(AtomicUsize::new(0));
    let source: mesh_llm_payments::plugin_server::ServiceSource = {
        let opens = opens.clone();
        Arc::new(move || {
            opens.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { anyhow::bail!("engine-open sentinel") })
        })
    };
    let runner: InProcessPluginRunner = Arc::new(move |stream| {
        let plugin = mesh_llm_payments::plugin_server::payments_plugin(
            "revision-test",
            "0.0.0",
            source.clone(),
            Arc::new(|| Box::pin(async { Ok(None) })),
        );
        Box::pin(PluginRuntime::run_with_stream(plugin, stream))
    });
    let (tx, _rx) = tokio::sync::mpsc::channel(32);
    let manager = PluginManager::start_with_in_process(
        &ResolvedPlugins {
            externals: vec![crate::plugin::in_process_builtin_spec("revision-test")],
            inactive: vec![],
        },
        PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        tx,
        InProcessPlugins::default().with("revision-test", runner),
    )
    .await
    .unwrap();
    let contract = manager
        .invoke_operation("revision-test", ops::BILLING_CONTRACT, "{}")
        .await
        .unwrap();
    assert!(!contract.is_error);
    let contract: mesh_llm_payments_types::contract::BillingContract =
        serde_json::from_str(&contract.content_json).unwrap();
    assert_eq!(
        contract.revision,
        mesh_llm_payments_types::contract::BillingRevision::V3
    );
    assert_eq!(
        opens.load(Ordering::SeqCst),
        0,
        "contract check must not open the engine"
    );
    let pricing = json!({"input_msat_per_million":1000,"output_msat_per_million":1000});
    let requests = [
        (
            ops::SERVE_ADMIT,
            json!({"request_id":"r","buyer":"peer","model":"test","pricing":pricing}),
        ),
        (
            ops::ACCEPT_ADMISSION,
            json!({
                "terms":{"id":"r","peer":"peer","model":"test","pricing":pricing,
                    "max_output_tokens":100,"max_total_msat":5000,"expires_at_ms":u64::MAX},
                "admission":{"id":"r","model":"test","pricing":pricing,"deposit_msat":1000}
            }),
        ),
    ];
    for (operation, original) in &requests {
        for revision in [
            None,
            Some(json!("v2")),
            Some(json!("v5")),
            Some(json!(null)),
        ] {
            let mut request = original.clone();
            if let Some(revision) = revision {
                request["billing_revision"] = revision;
            }
            let result = manager
                .invoke_operation("revision-test", operation, &request.to_string())
                .await
                .unwrap();
            assert!(result.is_error, "{operation}: {request}");
            assert_eq!(
                opens.load(Ordering::SeqCst),
                0,
                "decoder must reject before engine side effects"
            );
        }
    }
    for (operation, mut request) in requests {
        request["billing_revision"] = json!("v3");
        let result = manager
            .invoke_operation("revision-test", operation, &request.to_string())
            .await
            .unwrap();
        assert!(result.is_error); // The valid request reaches our engine-open sentinel.
    }
    assert_eq!(opens.load(Ordering::SeqCst), 2);
    manager.shutdown().await;
}

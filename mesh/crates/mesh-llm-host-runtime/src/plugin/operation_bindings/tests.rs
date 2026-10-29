use super::*;
use crate::plugin::{InProcessPluginRunner, proto};
use mesh_llm_plugin::{LocalStream, read_envelope, write_envelope};
use std::sync::atomic::{AtomicUsize, Ordering};

async fn serve(mut stream: LocalStream, writes: Arc<AtomicUsize>) -> anyhow::Result<()> {
    loop {
        let incoming = read_envelope(&mut stream).await?;
        let payload = match incoming.payload {
            Some(proto::envelope::Payload::InitializeRequest(_)) => {
                proto::envelope::Payload::InitializeResponse(proto::InitializeResponse {
                    plugin_id: incoming.plugin_id.clone(),
                    plugin_protocol_version: super::super::PROTOCOL_VERSION,
                    plugin_version: "fixture".into(),
                    server_info_json: serde_json::to_string(&rmcp::model::ServerConfig::default())?,
                    manifest: Some(proto::PluginManifest::default()),
                    ..Default::default()
                })
            }
            Some(proto::envelope::Payload::InvokeServiceRequest(request)) => {
                if request.service_name == "write" {
                    writes.fetch_add(1, Ordering::SeqCst);
                }
                proto::envelope::Payload::InvokeServiceResponse(proto::InvokeServiceResponse {
                    output_json: request.input_json,
                    is_error: false,
                })
            }
            Some(proto::envelope::Payload::ShutdownRequest(_)) => return Ok(()),
            _ => continue,
        };
        write_envelope(
            &mut stream,
            &proto::Envelope {
                protocol_version: super::super::PROTOCOL_VERSION,
                plugin_id: incoming.plugin_id,
                request_id: incoming.request_id,
                payload: Some(payload),
            },
        )
        .await?;
    }
}

#[tokio::test]
async fn fixed_payload_provider_uses_fixed_setup_allowlist_generation_and_revocation() {
    let writes = Arc::new(AtomicUsize::new(0));
    let runner: InProcessPluginRunner = {
        let writes = writes.clone();
        Arc::new(move |stream| Box::pin(serve(stream, writes.clone())))
    };
    let caller = super::super::runtime::tests::in_process_plugin("client", runner.clone());
    let provider = super::super::runtime::tests::in_process_plugin("state-store", runner);
    // Normal plugin initialization over the actual control connection.
    caller.call_tool("probe", "{}").await.unwrap();
    provider.call_tool("probe", "{}").await.unwrap();
    let generation = caller.bound_generation().await.unwrap();
    let grant = PluginOperationBinding {
        capability: None,
        setup_directory: false,
        name: "state".into(),
        provider: "state-store".into(),
        profile: "one".into(),
        setup_operation: "open".into(),
        setup_input_json: "{\"profile\":\"one\"}".into(),
        operations: vec!["write".into()],
    };
    let mut manager = PluginManager::for_test_summaries(Vec::new());
    let inner = Arc::get_mut(&mut manager.inner).unwrap();
    inner.plugins.insert("client".into(), caller);
    inner.plugins.insert("state-store".into(), provider);
    inner.operation_bindings.0.lock().unwrap().insert(
        ("client".into(), "state".into()),
        Binding {
            grant,
            handle: None,
            capacity: Arc::new(Semaphore::new(8)),
            installed_grant: false,
            resolved: None,
        },
    );
    let request = |handle, operation: &str, input: &str| BoundOperationRequest {
        binding: "state".into(),
        handle,
        operation: operation.into(),
        input_json: input.into(),
    };
    assert!(matches!(
        manager
            .bound_operation("client", generation, request(None, "open", "{}"))
            .await,
        Response::NotDispatched { .. }
    ));
    let Response::Responded {
        handle,
        output_json,
        ..
    } = manager
        .bound_operation("client", generation, request(None, "", ""))
        .await
    else {
        panic!("setup must respond")
    };
    assert_eq!(output_json, "{\"profile\":\"one\"}");
    assert!(!handle.is_empty());
    assert!(matches!(
        manager
            .bound_operation(
                "client",
                generation,
                request(Some(handle.clone()), "open", "{}")
            )
            .await,
        Response::NotDispatched { .. }
    ));
    assert!(matches!(
        manager
            .bound_operation(
                "client",
                generation,
                request(Some(handle.clone()), "write", "{}")
            )
            .await,
        Response::Responded { .. }
    ));
    assert_eq!(writes.load(Ordering::SeqCst), 1);
    assert!(matches!(
        manager
            .bound_operation(
                "client",
                generation + 1,
                request(Some(handle.clone()), "write", "{}")
            )
            .await,
        Response::NotDispatched { .. }
    ));
    // A valid handle is worthless to any other plugin, including the provider
    // itself: the binding is keyed by the caller the operator named.
    let provider_generation = manager
        .inner
        .plugins
        .get("state-store")
        .unwrap()
        .bound_generation()
        .await
        .unwrap();
    assert!(matches!(
        manager
            .bound_operation(
                "state-store",
                provider_generation,
                request(Some(handle.clone()), "write", "{}")
            )
            .await,
        Response::NotDispatched { reason } if reason == "binding is not authorized"
    ));
    assert_eq!(writes.load(Ordering::SeqCst), 1);
    let Response::Responded { handle: fresh, .. } = manager
        .bound_operation("client", generation, request(None, "", ""))
        .await
    else {
        panic!("repeat setup must respond")
    };
    assert_ne!(fresh, handle);
    assert!(matches!(
        manager
            .bound_operation(
                "client",
                generation,
                request(Some(handle.clone()), "write", "{}")
            )
            .await,
        Response::NotDispatched { .. }
    ));
    // Hold the final admission lock: the call has passed initial authorization
    // and reserved queue capacity, but cannot yet reach the enqueue boundary.
    let pending = manager.inner.plugins["state-store"]
        .bound_test_pending()
        .await;
    let blocked = pending.lock().await;
    let gate = manager
        .inner
        .operation_bindings
        .0
        .lock()
        .unwrap()
        .get(&("client".into(), "state".into()))
        .unwrap()
        .capacity
        .clone();
    let queued_manager = manager.clone();
    let queued = tokio::spawn(async move {
        queued_manager
            .bound_operation(
                "client",
                generation,
                BoundOperationRequest {
                    binding: "state".into(),
                    handle: Some(fresh),
                    operation: "write".into(),
                    input_json: "{}".into(),
                },
            )
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while gate.available_permits() == 8 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    manager.revoke_changed_operation_bindings(&MeshConfig::default());
    drop(blocked);
    assert!(
        matches!(tokio::time::timeout(std::time::Duration::from_secs(2), queued)
        .await.unwrap().unwrap(), Response::NotDispatched { reason } if reason == "binding revoked")
    );
    assert_eq!(writes.load(Ordering::SeqCst), 1);
    manager.revoke_changed_operation_bindings(&MeshConfig::default());

    assert!(matches!(
        manager
            .bound_operation("client", generation, request(Some(handle), "write", "{}"))
            .await,
        Response::NotDispatched { .. }
    ));
    manager.shutdown().await;
}

#[test]
fn setup_directory_payload_scopes_the_caller_base_to_the_provider() {
    let base = tempfile::tempdir().unwrap();
    let vaults = base.path().join("vaults");
    let payload = super::setup_directory_payload(
        &serde_json::json!({ "directory": vaults }).to_string(),
        "vault-a",
    )
    .unwrap();
    let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(
        value["directory"].as_str().unwrap(),
        vaults.join("vault-a").display().to_string(),
        "main's layout: <state>/vaults/<provider>"
    );
    assert!(vaults.join("vault-a").is_dir());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(vaults.join("vault-a"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o700);
    }
    // The caller cannot pick the provider segment or leave the base: a relative
    // base, an empty payload and a provider that is not a plain name are refused.
    assert!(
        super::setup_directory_payload(r#"{"directory":"relative/vaults"}"#, "vault-a").is_err()
    );
    assert!(super::setup_directory_payload("", "vault-a").is_err());
    assert!(
        super::setup_directory_payload(r#"{"directory":"/tmp","extra":1}"#, "vault-a").is_err()
    );
    assert!(
        super::setup_directory_payload(
            &serde_json::json!({ "directory": vaults }).to_string(),
            "../x"
        )
        .is_err()
    );
}

/// A binding granted by an installed package has no config entry; applying
/// config must keep it until an entry replaces it or disables the plugin.
#[test]
fn installed_package_grant_survives_config_apply_until_replaced() {
    fn grant(name: &str) -> PluginOperationBinding {
        PluginOperationBinding {
            capability: Some("vault.v1".into()),
            setup_directory: true,
            name: name.into(),
            provider: String::new(),
            profile: "default".into(),
            setup_operation: "vault_open".into(),
            setup_input_json: "{}".into(),
            operations: vec!["vault_read".into()],
        }
    }
    fn manager_with(installed_grant: bool) -> PluginManager {
        let manager = PluginManager::for_test_summaries(Vec::new());
        manager.inner.operation_bindings.0.lock().unwrap().insert(
            ("pay".into(), "vault".into()),
            Binding {
                grant: grant("vault"),
                handle: None,
                capacity: Arc::new(Semaphore::new(8)),
                installed_grant,
                resolved: None,
            },
        );
        manager
    }
    fn entry(bindings: Vec<PluginOperationBinding>, enabled: Option<bool>) -> MeshConfig {
        let mut config = MeshConfig::default();
        config.plugins.push(crate::plugin::PluginConfigEntry {
            operation_bindings: bindings,
            name: "pay".into(),
            enabled,
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            allow_peer_blocks: None,
            allow_inference_observation: None,
            openai_exchange_grant: None,
            command: None,
            args: Vec::new(),
            url: None,
            settings: std::collections::BTreeMap::new(),
            startup: crate::plugin::PluginStartupConfig::default(),
        });
        config
    }
    let kept = |manager: &PluginManager| {
        manager
            .inner
            .operation_bindings
            .0
            .lock()
            .unwrap()
            .contains_key(&("pay".to_owned(), "vault".to_owned()))
    };

    let manager = manager_with(true);
    manager.revoke_changed_operation_bindings(&MeshConfig::default());
    assert!(kept(&manager), "no config entry: the package grant stands");
    manager.revoke_changed_operation_bindings(&entry(Vec::new(), None));
    assert!(
        kept(&manager),
        "an entry without bindings does not replace the grant"
    );
    manager.revoke_changed_operation_bindings(&entry(vec![grant("other")], None));
    assert!(
        !kept(&manager),
        "an entry naming its own bindings replaces the grant"
    );

    let manager = manager_with(true);
    manager.revoke_changed_operation_bindings(&entry(Vec::new(), Some(false)));
    assert!(!kept(&manager), "a disabled plugin keeps no grant");

    let manager = manager_with(false);
    manager.revoke_changed_operation_bindings(&MeshConfig::default());
    assert!(
        !kept(&manager),
        "a config grant whose entry is gone is revoked"
    );
}

/// A provider belongs to one binding: a second capability binding resolving to
/// it, or a capability binding resolving to a provider another binding names,
/// is refused; and resolving keeps the grant itself unchanged so an unchanged
/// owner config entry does not revoke the binding afterwards.
#[test]
fn resolved_provider_is_exclusive_and_leaves_the_grant_comparable() {
    fn capability_grant(name: &str) -> PluginOperationBinding {
        PluginOperationBinding {
            capability: Some("vault.v1".into()),
            setup_directory: true,
            name: name.into(),
            provider: String::new(),
            profile: "default".into(),
            setup_operation: "vault_open".into(),
            setup_input_json: "{}".into(),
            operations: vec!["vault_read".into()],
        }
    }
    fn binding(grant: PluginOperationBinding) -> Binding {
        Binding {
            grant,
            handle: None,
            capacity: Arc::new(Semaphore::new(8)),
            installed_grant: false,
            resolved: None,
        }
    }
    let pay = ("pay".to_owned(), "vault".to_owned());
    let other = ("other".to_owned(), "vault".to_owned());
    let named = ("named".to_owned(), "vault".to_owned());
    let mut registry = BTreeMap::new();
    registry.insert(pay.clone(), binding(capability_grant("vault")));
    registry.insert(other.clone(), binding(capability_grant("vault")));
    let mut fixed = capability_grant("vault");
    fixed.capability = None;
    fixed.provider = "vault-b".into();
    registry.insert(named.clone(), binding(fixed));

    let capacity = registry[&pay].capacity.clone();
    reserve_resolved_provider(&mut registry, &pay, &capacity, "vault-a").unwrap();
    assert_eq!(registry[&pay].provider(), Some("vault-a"));
    assert!(
        registry[&pay].grant.provider.is_empty(),
        "resolution must not rewrite the grant"
    );

    let capacity = registry[&other].capacity.clone();
    assert_eq!(
        reserve_resolved_provider(&mut registry, &other, &capacity, "vault-a"),
        Err("vault-a is already bound exclusively by consumer/vault".into())
    );
    assert_eq!(
        reserve_resolved_provider(&mut registry, &other, &capacity, "vault-b"),
        Err("vault-b is already bound exclusively by named/vault".into())
    );
    assert_eq!(registry[&other].provider(), None);

    // Re-resolving the owner to the same provider is idempotent; a stale
    // registration (different capacity) is refused.
    let capacity = registry[&pay].capacity.clone();
    reserve_resolved_provider(&mut registry, &pay, &capacity, "vault-a").unwrap();
    assert!(
        reserve_resolved_provider(&mut registry, &pay, &Arc::new(Semaphore::new(8)), "vault-a")
            .is_err()
    );

    // The owner config entry still carries the grant as written (no provider);
    // reapplying it after resolution keeps the resolved binding.
    let manager = PluginManager::for_test_summaries(Vec::new());
    *manager.inner.operation_bindings.0.lock().unwrap() = registry;
    let mut config = MeshConfig::default();
    config.plugins.push(crate::plugin::PluginConfigEntry {
        operation_bindings: vec![capability_grant("vault")],
        name: "pay".into(),
        enabled: None,
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        allow_peer_blocks: None,
        allow_inference_observation: None,
        openai_exchange_grant: None,
        command: None,
        args: Vec::new(),
        url: None,
        settings: std::collections::BTreeMap::new(),
        startup: crate::plugin::PluginStartupConfig::default(),
    });
    manager.revoke_changed_operation_bindings(&config);
    let registry = manager.inner.operation_bindings.0.lock().unwrap();
    assert_eq!(registry[&pay].provider(), Some("vault-a"));
    // Disabling the resolved provider's plugin revokes the binding bound to it.
    drop(registry);
    config.plugins.push(crate::plugin::PluginConfigEntry {
        operation_bindings: Vec::new(),
        name: "vault-a".into(),
        enabled: Some(false),
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        allow_peer_blocks: None,
        allow_inference_observation: None,
        openai_exchange_grant: None,
        command: None,
        args: Vec::new(),
        url: None,
        settings: std::collections::BTreeMap::new(),
        startup: crate::plugin::PluginStartupConfig::default(),
    });
    manager.revoke_changed_operation_bindings(&config);
    assert!(
        !manager
            .inner
            .operation_bindings
            .0
            .lock()
            .unwrap()
            .contains_key(&pay)
    );
}

use super::*;
#[cfg(test)]
mod requests {
    use super::*;

    fn fixture(
        timeout: u64,
    ) -> (
        Arc<NativeModelLifecycle>,
        mpsc::Receiver<ModelIntent>,
        watch::Sender<ModelTargets>,
    ) {
        let (intents, receiver) = mpsc::channel(8);
        let (targets, snapshot) = watch::channel(ModelTargets::default());
        let residency = Arc::new(Residency {
            policy: ModelLifecycle {
                load: ModelLoadPolicy::OnDemand,
                idle_timeout_secs: timeout,
                prefetch: false,
            },
            spec: "test".into(),
            names: vec!["test".into()],
            loading: Mutex::new(None),
            gate: Arc::default(),
            operation: Arc::default(),
            idle_since: std::sync::Mutex::new(std::time::Instant::now() - Duration::from_secs(60)),
            instance: std::sync::Mutex::new(None),
        });
        (
            Arc::new(NativeModelLifecycle {
                models: BTreeMap::from([("test".into(), residency)]),
                targets: snapshot,
                intents,
            }),
            receiver,
            targets,
        )
    }

    fn finish_load(
        lifecycle: &NativeModelLifecycle,
        intent: ModelIntent,
        targets: &watch::Sender<ModelTargets>,
    ) {
        let ModelIntent::Load {
            completion: Some(completion),
            ..
        } = intent
        else {
            panic!("expected load")
        };
        targets.send_modify(|targets| {
            targets
                .targets
                .insert("loaded-test".into(), vec![InferenceTarget::Local(1234)]);
        });
        let response = crate::api::RuntimeLoadResponse {
            model_ref: "test".into(),
            model: "loaded-test".into(),
            instance_id: "runtime-1".into(),
            profile: String::new(),
            backend: None,
            context_length: None,
        };
        lifecycle.record_loaded("test", &response.instance_id, &response.model, 1234);
        completion.send(Ok(response)).unwrap();
    }

    #[tokio::test]
    async fn simultaneous_requests_share_one_load_and_hold_stream_leases() {
        let (lifecycle, mut intents, targets) = fixture(1);
        let first = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test").await.unwrap().unwrap() })
        };
        let second = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test").await.unwrap().unwrap() })
        };
        finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
        let (first, second) = (first.await.unwrap(), second.await.unwrap());
        assert!(intents.try_recv().is_err());
        assert_eq!(lifecycle.execution_model("test"), "loaded-test");
        assert!(lifecycle.manages("loaded-test"));
        let alias = lifecycle.acquire("loaded-test").await.unwrap();
        assert!(intents.try_recv().is_err());
        drop(alias);
        lifecycle.evict_idle().await;
        assert!(intents.try_recv().is_err());
        drop(first);
        lifecycle.evict_idle().await;
        assert!(intents.try_recv().is_err());
        drop(second);
        assert!(
            lifecycle.models["test"]
                .idle_since
                .lock()
                .unwrap()
                .elapsed()
                < Duration::from_secs(1)
        );
    }

    #[tokio::test]
    async fn idle_eviction_targets_exact_instance_and_next_request_reloads() {
        let (lifecycle, mut intents, targets) = fixture(1);
        *lifecycle.models["test"].instance.lock().unwrap() =
            Some(("runtime-1".into(), "loaded-test".into(), 1234));
        let eviction = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.evict_idle().await })
        };
        let ModelIntent::Unload {
            target,
            completion: Some(completion),
            options,
            ..
        } = intents.recv().await.unwrap()
        else {
            panic!("expected unload")
        };
        assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));
        assert!(!options.force);
        completion
            .send(Ok(crate::api::RuntimeUnloadResponse {
                model: "loaded-test".into(),
                instance_id: "runtime-1".into(),
                unloaded: true,
            }))
            .unwrap();
        eviction.await.unwrap();
        let request = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test").await.unwrap() })
        };
        finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
        assert!(request.await.unwrap().is_some());
    }

    /// Rejections and lost completions wait another idle interval before retrying.
    #[tokio::test]
    async fn deferred_idle_unloads_back_off_before_retrying() {
        for error in [
            Some("maintained owner intent"),
            Some("instance no longer exists"),
            None,
        ] {
            let (lifecycle, mut intents, _targets) = fixture(30);
            let residency = &lifecycle.models["test"];
            *residency.instance.lock().unwrap() =
                Some(("runtime-1".into(), "loaded-test".into(), 1234));
            let eviction = {
                let lifecycle = lifecycle.clone();
                tokio::spawn(async move { lifecycle.evict_idle().await })
            };
            let ModelIntent::Unload {
                completion: Some(completion),
                ..
            } = intents.recv().await.unwrap()
            else {
                panic!("expected unload")
            };
            if let Some(error) = error {
                completion.send(Err(anyhow::anyhow!(error))).unwrap();
            } else {
                drop(completion);
            }
            eviction.await.unwrap();
            assert!(residency.instance.lock().unwrap().is_some());
            lifecycle.evict_idle().await;
            assert!(
                intents.try_recv().is_err(),
                "must not retry on the next maintenance tick"
            );
            *residency.idle_since.lock().unwrap() =
                std::time::Instant::now() - Duration::from_secs(15);
            lifecycle.evict_idle().await;
            assert!(
                intents.try_recv().is_err(),
                "must wait the entire idle interval"
            );
            *residency.idle_since.lock().unwrap() =
                std::time::Instant::now() - Duration::from_secs(31);
            let retry = {
                let lifecycle = lifecycle.clone();
                tokio::spawn(async move { lifecycle.evict_idle().await })
            };
            let ModelIntent::Unload {
                completion: Some(completion),
                ..
            } = intents.recv().await.unwrap()
            else {
                panic!("expected retry")
            };
            completion
                .send(Ok(crate::api::RuntimeUnloadResponse {
                    model: "loaded-test".into(),
                    instance_id: "runtime-1".into(),
                    unloaded: true,
                }))
                .unwrap();
            retry.await.unwrap();
            assert!(residency.instance.lock().unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn resident_policy_and_unowned_instances_are_never_evicted() {
        for timeout in [0, 1] {
            let (lifecycle, mut intents, _targets) = fixture(timeout);
            if timeout == 0 {
                *lifecycle.models["test"].instance.lock().unwrap() =
                    Some(("runtime-1".into(), "loaded-test".into(), 1234));
            }
            lifecycle.evict_idle().await;
            assert!(intents.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn failed_load_releases_waiters_and_allows_retry() {
        let (lifecycle, mut intents, targets) = fixture(1);
        let first = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test").await })
        };
        let ModelIntent::Load {
            completion: Some(completion),
            ..
        } = intents.recv().await.unwrap()
        else {
            panic!("expected load")
        };
        completion
            .send(Err(anyhow::anyhow!("capacity exhausted")))
            .unwrap();
        assert!(first.await.unwrap().is_err());
        let retry = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test").await })
        };
        finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
        assert!(retry.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn cancelling_request_does_not_lose_load_ownership() {
        let (lifecycle, mut intents, targets) = fixture(1);
        let request = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test").await })
        };
        let intent = intents.recv().await.unwrap();
        request.abort();
        finish_load(&lifecycle, intent, &targets);
        // Acquiring again synchronizes with the detached load job.
        let _lease = lifecycle.acquire("test").await.unwrap();
        assert!(lifecycle.models["test"].instance.lock().unwrap().is_some());
        assert!(intents.try_recv().is_err());
    }

    #[tokio::test]
    async fn failed_load_is_shared_and_cancelled_waiters_do_not_queue_retries() {
        for cancel in [false, true] {
            let (lifecycle, mut intents, _targets) = fixture(0);
            let mut waiters = (0..16)
                .map(|_| Box::pin(lifecycle.acquire("test")))
                .collect::<Vec<_>>();
            std::future::poll_fn(|cx| {
                for waiter in &mut waiters {
                    assert!(std::future::Future::poll(waiter.as_mut(), cx).is_pending());
                }
                std::task::Poll::Ready(())
            })
            .await;
            let ModelIntent::Load {
                completion: Some(completion),
                ..
            } = intents.recv().await.unwrap()
            else {
                panic!("expected load")
            };
            if cancel {
                drop(waiters);
                completion.send(Err(anyhow::anyhow!("capacity"))).unwrap();
            } else {
                completion.send(Err(anyhow::anyhow!("capacity"))).unwrap();
                for waiter in waiters {
                    assert!(waiter.await.is_err());
                }
            }
            let _finished = lifecycle.models["test"].operation.lock().await;
            assert!(
                intents.try_recv().is_err(),
                "failed/cancelled waiters must not spawn sequential loads"
            );
        }
    }

    #[tokio::test]
    async fn actual_instance_guard_protects_a_port_selected_under_another_model_name() {
        let (mut lifecycle, mut intents, targets) = fixture(1);
        let other = Arc::new(Residency {
            policy: ModelLifecycle {
                load: ModelLoadPolicy::OnDemand,
                idle_timeout_secs: 1,
                prefetch: false,
            },
            spec: "other".into(),
            names: vec!["other".into()],
            loading: Mutex::new(None),
            gate: Arc::default(),
            operation: Arc::default(),
            idle_since: std::sync::Mutex::new(std::time::Instant::now() - Duration::from_secs(60)),
            instance: std::sync::Mutex::new(Some(("runtime-2".into(), "shared".into(), 2345))),
        });
        Arc::get_mut(&mut lifecycle)
            .unwrap()
            .models
            .insert("other".into(), other.clone());
        *lifecycle.models["test"].instance.lock().unwrap() =
            Some(("runtime-1".into(), "shared".into(), 1234));
        targets.send_modify(|targets| {
            targets.targets.insert(
                "shared".into(),
                vec![InferenceTarget::Local(1234), InferenceTarget::Local(2345)],
            );
        });
        assert_eq!(
            lifecycle.targets_for("test").candidates("shared"),
            vec![InferenceTarget::Local(1234)]
        );
        assert_eq!(
            lifecycle.targets_for("other").candidates("shared"),
            vec![InferenceTarget::Local(2345)]
        );
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        assert!(node.native_model_lifecycle.set(lifecycle.clone()).is_ok());
        node.register_runtime_instance_lifecycle(
            2345,
            Arc::new(tokio::sync::Mutex::new(
                crate::runtime::InstanceLifecycleRecord::new(
                    crate::runtime::InstanceLifecycleState::Serving,
                    10,
                ),
            )),
        );
        let outer = lifecycle.acquire("test").await.unwrap();
        let selected = node.begin_runtime_instance_request(2345).await.unwrap();
        lifecycle.evict_idle().await;
        assert!(intents.try_recv().is_err());
        drop(outer);
        lifecycle.evict_idle().await;
        assert!(
            intents.try_recv().is_err(),
            "selected port must remain leased after an unrelated model lease ends"
        );
        drop(selected);
        assert!(other.idle_since.lock().unwrap().elapsed() < Duration::from_secs(1));
        node.endpoint.close().await;
    }

    #[test]
    fn assigned_peer_execution_never_uses_a_remote_ready_target() {
        let peer = iroh::SecretKey::from_bytes(&[7; 32]).public();
        let mut targets = ModelTargets::default();
        targets
            .targets
            .insert("test".into(), vec![InferenceTarget::Remote(peer)]);
        assert!(ready(&targets, "test", false));
        assert!(!ready(&targets, "test", true));
    }
}

#[cfg(test)]
mod intent_tests {
    use super::*;

    #[test]
    fn idle_eviction_respects_owner_ensure_and_refuses_split_controllers() {
        use crate::runtime::model_reconciliation::ModelTargetReconciliationState;
        use crate::runtime::run_auto::{RuntimeUnloadCandidate, RuntimeUnloadOwner};
        let mut state = ModelTargetReconciliationState::default();
        let mut candidate = RuntimeUnloadCandidate {
            owner: RuntimeUnloadOwner::Runtime,
            instance_id: "runtime-1".into(),
            model_name: "loaded-test".into(),
            profile: String::new(),
        };
        assert!(idle_unload_allowed(&state, Some(&candidate), Some("test")));
        state.add_desired("test", "", IntentSource::OwnerEnsure);
        assert!(!idle_unload_allowed(&state, Some(&candidate), Some("test")));
        candidate.owner = RuntimeUnloadOwner::Managed;
        assert!(!idle_unload_allowed(
            &ModelTargetReconciliationState::default(),
            Some(&candidate),
            Some("test")
        ));
        assert!(!idle_unload_allowed(&state, None, None));
    }
}

#[cfg(test)]
mod configuration_tests {
    use super::*;
    use crate::runtime::{RuntimeOptions, startup_models::build_startup_model_specs};

    /// Each uniquely named entry has either a startup spec or a residency.
    #[test]
    fn unique_on_demand_entries_are_managed_and_startup_entries_remain_eager() {
        let config: crate::plugin::MeshConfig = toml::from_str(
            r#"
[[models]]
model = "cold-model"
[models.lifecycle]
load = "on_demand"
[[models]]
model = "warm-model"
"#,
        )
        .unwrap();
        let residencies = configured_residencies(&config);
        let specs = build_startup_model_specs(&RuntimeOptions::default(), &config).unwrap();
        assert_eq!(
            residencies.keys().map(String::as_str).collect::<Vec<_>>(),
            ["cold-model"]
        );
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].config_model_id.as_deref(), Some("warm-model"));
    }

    /// Programmatic duplicate profiles must not fall between both launch paths.
    #[test]
    fn duplicate_model_profiles_keep_their_startup_specs() {
        let config: crate::plugin::MeshConfig = toml::from_str(
            r#"
[[models]]
model = "shared-model"
ctx_size = 1024
[models.lifecycle]
load = "on_demand"
[[models]]
model = "shared-model"
ctx_size = 2048
[models.lifecycle]
load = "on_demand"
"#,
        )
        .unwrap();
        assert!(configured_residencies(&config).is_empty());
        let specs = build_startup_model_specs(&RuntimeOptions::default(), &config).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(
            specs.iter().map(|spec| spec.ctx_size).collect::<Vec<_>>(),
            [Some(1024), Some(2048)]
        );
        assert_ne!(specs[0].profile, specs[1].profile);
    }
}

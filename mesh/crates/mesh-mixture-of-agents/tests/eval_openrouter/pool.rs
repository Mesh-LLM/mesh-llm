use super::*;

#[derive(Clone, Copy)]
pub(crate) enum Tier {
    Small,
    Big,
}

#[derive(Clone, Copy)]
pub(crate) struct PoolModel {
    pub(crate) id: &'static str,
    pub(crate) tier: Tier,
    pub(crate) fault: MeshFault,
}

/// A smorgasbord of tool-capable open-weight models a mesh would plausibly
/// solo-serve: small models on laptops/minis, bigger ones on a single good
/// GPU. Nothing that would require splitting. All verified tool-capable
/// during corpus recording. Fault profiles spread across the realism space so
/// one turn exercises fast, typical, slow-strong, and flaky peers together.
pub(crate) fn mesh_pool() -> Vec<PoolModel> {
    vec![
        PoolModel {
            id: "qwen/qwen3-8b",
            tier: Tier::Small,
            fault: MeshFault::RELIABLE_FAST,
        },
        PoolModel {
            id: "mistralai/ministral-8b-2512",
            tier: Tier::Small,
            fault: MeshFault::TYPICAL,
        },
        PoolModel {
            id: "meta-llama/llama-3.2-3b-instruct",
            tier: Tier::Small,
            fault: MeshFault::FLAKY,
        },
        PoolModel {
            id: "qwen/qwen3-14b",
            tier: Tier::Big,
            fault: MeshFault::TYPICAL,
        },
        PoolModel {
            id: "mistralai/mistral-small-3.2-24b-instruct",
            tier: Tier::Big,
            fault: MeshFault::TYPICAL,
        },
        PoolModel {
            id: "qwen/qwen3-32b",
            tier: Tier::Big,
            fault: MeshFault::SLOW_STRONG,
        },
    ]
}
pub(crate) fn moa_config(pool: &[PoolModel], api_key: &str, realism: bool) -> GatewayConfig {
    let mut backends: Vec<Arc<dyn ModelBackend>> = Vec::new();
    let mut models = Vec::new();
    for (i, m) in pool.iter().enumerate() {
        let base: Arc<dyn ModelBackend> = Arc::new(OpenRouterBackend::new(api_key.to_string()));
        let backend = if realism {
            MeshRealismBackend::wrap(base, m.fault, 0xE7A1_u64.wrapping_add(i as u64))
        } else {
            base
        };
        models.push(ModelEntry::new(m.id.to_string(), backends.len()));
        backends.push(backend);
    }
    GatewayConfig {
        backends,
        models,
        // Generous worker timeout: realism latency + real model latency can
        // stack, and we want to see slow workers land, not time out.
        worker_timeout: Duration::from_secs(90),
        hedge_delay: Duration::from_secs(5),
        reducer_timeout: Duration::from_secs(60),
        first_answer_grace: Duration::from_secs(3),
        strong_patience: Duration::from_secs(20),
        // MoA policy: thinking always off (matches effective_enable_thinking_for_moa).
        enable_thinking: Some(false),
        actor_candidates: Vec::new(),
        reference_policy: Default::default(),
        refinement_policy: Default::default(),
    }
}

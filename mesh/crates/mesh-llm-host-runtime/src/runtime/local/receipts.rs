//! Inference receipts for a model this node serves alone, when `[receipts]`
//! is on (`mesh/docs/design/RECEIPTS.md`). The node key signs them, so a
//! client can tell which peer served a turn it audits.
//!
//! A single stage has no receipts address of its own: clients open its
//! receipts on its OpenAI address (`POST /v1/receipts/open`), which the
//! mesh routes like any request for the model.

use std::path::Path;

use super::LocalOpenAiModelStartSpec;

/// What the runtime of `model_name` needs to issue receipts, or `None` when
/// the node issues none.
pub(super) fn receipts_options(
    spec: &LocalOpenAiModelStartSpec<'_>,
    model_name: &str,
) -> Option<skippy_serving::receipts::ReceiptsOptions> {
    spec.mesh_config.receipts.enabled.then(|| {
        skippy_serving::receipts::ReceiptsOptions::embedded(
            receipt_model(spec.model_path, model_name),
            spec.receipts_signing_key,
        )
    })
}

/// What a loaded runtime does about receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum RuntimeReceipts {
    /// Its recorder did not start.
    None,
    /// It issues them, except for a turn that continues a cached prefix:
    /// its stage keeps recurrent state (a hybrid model).
    NotForCachedPrefixes,
    /// It issues them for every turn.
    EveryTurn,
}

impl RuntimeReceipts {
    pub(in crate::runtime) fn of(model: &crate::inference::skippy::SkippyModelHandle) -> Self {
        if !model.issues_receipts() {
            Self::None
        } else if model.receipts_keep_recurrent_state() {
            Self::NotForCachedPrefixes
        } else {
            Self::EveryTurn
        }
    }

    /// Why a runtime that does this is advertised without receipts, if it
    /// is.
    pub(in crate::runtime) fn shortfall(self) -> Result<(), NoReceipts> {
        match self {
            Self::None => Err(NoReceipts::RuntimeIssuesNone),
            Self::NotForCachedPrefixes => Err(NoReceipts::RecurrentState),
            Self::EveryTurn => Ok(()),
        }
    }
}

/// Why a runtime is advertised without receipts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum NoReceipts {
    /// `[receipts] enabled` is off: nothing to explain.
    NotAsked,
    /// The node asked for receipts and the loaded runtime issues none: its
    /// recorder did not start (a backend or a graph it does not support,
    /// weights no audit can check, or a quantized KV cache).
    RuntimeIssuesNone,
    /// The runtime's stage keeps recurrent state, so a turn that continues
    /// a cached prefix gets no receipt. A peer that advertises receipts
    /// owes one for every turn, and is struck for each it misses
    /// (`network/receipt_audit/intake.rs`).
    RecurrentState,
    /// The model does not serve chat completions, the endpoint receipts
    /// cover.
    NotChat,
    /// A split with stages on other nodes; see
    /// `runtime/local_split/receipts.rs`.
    RemoteStages(usize),
    /// The receipts name the model by a name of this node's own, as for a
    /// GGUF served from a local path: no client can fetch what it names.
    NoPublicModel,
}

impl NoReceipts {
    /// What the operator reads when receipts were asked for and the model
    /// is advertised without them. `None` when they were not asked for.
    pub(in crate::runtime) fn explanation(&self) -> Option<String> {
        match self {
            Self::NotAsked => None,
            Self::RuntimeIssuesNone => Some(
                "its runtime issues no receipts a client can audit: the receipt recorder did not start on this model, its KV cache is quantized (cache types q8_0 or q4_0, which no audit covers; f16 is covered), its backend is not CPU or CUDA (Vulkan, Metal and ROCm have no rounding profiles, so no audit covers them), or its weights have a type without a rounding profile on its backend (f16 or bf16 on CUDA); the stage log names the reason"
                    .to_owned(),
            ),
            Self::RecurrentState => Some(
                "it is a hybrid model with recurrent state, and a turn that continues a cached prefix, as every later turn of a chat does, gets no receipt; a peer that advertises receipts owes one for every turn and is struck for each it misses"
                    .to_owned(),
            ),
            Self::NotChat => Some(
                "it does not serve chat completions, the endpoint receipts cover".to_owned(),
            ),
            Self::RemoteStages(stages) => Some(format!(
                "{stages} of its stages run on other nodes: nothing confirms that those nodes record receipts, and the turns this node coordinates across them are not audited"
            )),
            Self::NoPublicModel => Some(
                "its receipts name no file of a Hugging Face repository at a commit, so no client can fetch the model to audit them: serve it from the Hugging Face cache, by a model reference, and not from a local path"
                    .to_owned(),
            ),
        }
    }
}

/// Whether a model this node serves alone is advertised with receipts, and
/// satisfies the `receipts` trust tier for this node's own clients: only
/// when the loaded runtime says it issues them for every turn, the model
/// serves chat, and its receipts name a model a client can fetch
/// (`receipt_model` is that name; see [`names_a_public_model`]).
/// `[receipts] enabled` says what the operator asked for, not what the
/// backend could do.
pub(in crate::runtime) fn receipts_advertisement(
    asked: bool,
    runtime: RuntimeReceipts,
    workload_class: crate::mesh::ModelWorkloadClass,
    receipt_model: &str,
) -> Result<(), NoReceipts> {
    if !asked {
        return Err(NoReceipts::NotAsked);
    }
    runtime.shortfall()?;
    if workload_class != crate::mesh::ModelWorkloadClass::CausalGeneration {
        return Err(NoReceipts::NotChat);
    }
    if !names_a_public_model(receipt_model) {
        return Err(NoReceipts::NoPublicModel);
    }
    Ok(())
}

/// Whether an audit can cover receipts that name `receipt_model`: it is a
/// file of a Hugging Face repository at a commit, which a client reads to
/// check the turn. This is the test an auditor applies to a receipt before
/// it fetches anything (`network/receipt_audit/requested_model.rs`). A
/// model served from a local path is named `local-gguf/sha256-…`, and
/// every audit of it ends as no audit: it must not satisfy the tier.
pub(in crate::runtime) fn names_a_public_model(receipt_model: &str) -> bool {
    skippy_receipt_audit::claimed_file::HfFile::parse(receipt_model)
        .is_some_and(|file| file.is_pinned())
}

/// Says once, when a load ends, why a model receipts were asked for is
/// advertised without them. Returns whether it is advertised with them.
pub(in crate::runtime) fn report_advertisement(
    model_name: &str,
    advertisement: Result<(), NoReceipts>,
) -> bool {
    match advertisement {
        Ok(()) => true,
        Err(why) => {
            if let Some(explanation) = why.explanation() {
                tracing::warn!(
                    model = model_name,
                    "[receipts] enabled is set, but {model_name} is advertised without receipts and does not satisfy the receipts trust tier: {explanation}"
                );
            }
            false
        }
    }
}

/// Whether a model issues receipts, given whether each of its loaded
/// instances does. A serving ID may select several loaded instances, and a
/// request can land on any of them: the model issues receipts only when
/// every one does.
pub(in crate::runtime) fn model_issues_receipts(instances: impl IntoIterator<Item = bool>) -> bool {
    let mut instances = instances.into_iter().peekable();
    instances.peek().is_some() && instances.all(|issues| issues)
}

/// Whether the node advertises receipts for the model it just loaded.
pub(super) fn advertised(
    spec: &LocalOpenAiModelStartSpec<'_>,
    model_name: &str,
    model: &crate::inference::skippy::SkippyModelHandle,
    workload_class: crate::mesh::ModelWorkloadClass,
) -> bool {
    report_advertisement(
        model_name,
        receipts_advertisement(
            spec.mesh_config.receipts.enabled,
            RuntimeReceipts::of(model),
            workload_class,
            &receipt_model(spec.model_path, model_name),
        ),
    )
}

/// The KV cache settings, in the order of the cache types they choose.
const CACHE_TYPE_SETTINGS: [&str; 2] = ["model_fit.cache_type_k", "model_fit.cache_type_v"];

/// Whether an audit covers KV rows of the cache type `name`, as the
/// auditor tests it (`network/receipt_audit/intake.rs`).
fn audited_cache_type(name: &str) -> bool {
    use skippy_receipts::cache_types;
    (0..64).any(|id| cache_types::audited(id).is_some() && cache_types::name(id) == name)
}

/// What the operator reads when receipts were asked for and `model_name`
/// loads with a KV cache no audit covers because its layer package
/// defaults to one, not because a setting chose it: its runtime then
/// issues no receipts, and the setting that fixes it is named.
/// `package_default` is the K and V cache types the package defaults to,
/// `overrides` the ones the model's settings chose (`auto` chooses none).
pub(in crate::runtime) fn quantized_default_cache(
    asked: bool,
    model_name: &str,
    package_default: [&str; 2],
    overrides: [Option<&str>; 2],
) -> Option<String> {
    if !asked {
        return None;
    }
    let fixes: Vec<String> = CACHE_TYPE_SETTINGS
        .iter()
        .zip(package_default)
        .zip(overrides)
        .filter(|((_, default), chosen)| {
            chosen.is_none_or(|chosen| chosen == "auto") && !audited_cache_type(default)
        })
        .map(|((setting, default), _)| {
            format!("{setting} = \"f16\" (the package default is {default})")
        })
        .collect();
    (!fixes.is_empty()).then(|| {
        format!(
            "[receipts] enabled is set, but the layer package of {model_name} defaults to a quantized KV cache, which no audit covers, so it is served without receipts; set {} for it in its [[models]] entry",
            fixes.join(" and ")
        )
    })
}

/// Says, when a model loads, that its layer package's KV cache default
/// keeps it from issuing receipts ([`quantized_default_cache`]).
pub(in crate::runtime) fn warn_quantized_default_cache(
    asked: bool,
    model_name: &str,
    package_default: [&str; 2],
    overrides: [Option<&str>; 2],
) {
    if let Some(warning) = quantized_default_cache(asked, model_name, package_default, overrides) {
        tracing::warn!(model = model_name, "{warning}");
    }
}

/// The model a receipt names: `repo@commit/file` when the GGUF is a Hugging
/// Face file, or the first shard of a split one, which a client then reads
/// by range to audit the turn without the model; else `model_name`.
pub(in crate::runtime) fn receipt_model(path: &Path, model_name: &str) -> String {
    hugging_face_ref(path).unwrap_or_else(|| model_name.to_owned())
}

/// `repo@commit/file` for a GGUF in the Hugging Face cache: a whole GGUF,
/// or the first shard of a split one, which names the split. Another shard
/// is not the model, and names nothing.
fn hugging_face_ref(path: &Path) -> Option<String> {
    let file = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let names_model =
        skippy_model_ref::split_gguf_shard_info(file).is_none_or(|shard| shard.part == "00001");
    names_model
        .then(|| crate::models::huggingface_identity_for_path(path))
        .flatten()
        .map(|identity| identity.canonical_ref)
}

#[cfg(test)]
mod tests {
    use super::{
        NoReceipts, RuntimeReceipts, model_issues_receipts, names_a_public_model,
        quantized_default_cache, receipt_model, receipts_advertisement, report_advertisement,
    };
    use crate::mesh::{self, ModelWorkloadClass};
    use crate::runtime::runtime_registry::{
        RuntimeModelRegistration, register_runtime_instance, unregister_runtime_instance,
    };
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn a_package_that_defaults_to_a_quantized_cache_names_the_setting_that_fixes_it() {
        let warning = quantized_default_cache(true, "m", ["q8_0", "f16"], [None, None])
            .expect("a quantized default is said");
        assert!(
            warning.contains("model_fit.cache_type_k = \"f16\""),
            "{warning}"
        );
        assert!(!warning.contains("model_fit.cache_type_v"), "{warning}");
        let both = quantized_default_cache(true, "m", ["q8_0", "q4_0"], [None, Some("auto")])
            .expect("auto chooses nothing");
        assert!(both.contains("cache_type_v = \"f16\" (the package default is q4_0)"));
        // receipts not asked for, an audited default, or a setting that chose
        assert_eq!(
            quantized_default_cache(false, "m", ["q8_0", "q8_0"], [None, None]),
            None
        );
        assert_eq!(
            quantized_default_cache(true, "m", ["f16", "f32"], [None, None]),
            None
        );
        assert_eq!(
            quantized_default_cache(true, "m", ["q8_0", "q8_0"], [Some("f16"), Some("q8_0")]),
            None
        );
    }

    #[test]
    fn a_model_with_no_instance_issues_no_receipts() {
        assert!(!model_issues_receipts([]));
        assert!(model_issues_receipts([true, true]));
        assert!(!model_issues_receipts([true, false]));
    }

    #[tokio::test]
    async fn a_model_issues_receipts_only_while_every_instance_does() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let registry = Arc::default();
        let register = |instance_id: &'static str, receipts: bool| {
            let (registry, node) = (&registry, &node);
            async move {
                register_runtime_instance(
                    registry,
                    node,
                    RuntimeModelRegistration {
                        primary_model_name: "test-model",
                        model_name: "test-model",
                        instance_id,
                        context_length: Some(8192),
                        thinking: None,
                        backend: None,
                        port: 0,
                        capabilities: Default::default(),
                        workload_class: mesh::ModelWorkloadClass::CausalGeneration,
                        receipts,
                    },
                )
                .await;
            }
        };
        register("receipted", true).await;
        assert!(node.issues_receipts("test-model"));
        // a request may land on either instance, and one issues none
        register("plain", false).await;
        assert!(!node.issues_receipts("test-model"));
        let advertised = node.local_descriptors_with_plugins().await;
        assert!(advertised.iter().all(|descriptor| !descriptor.receipts));

        assert!(!unregister_runtime_instance(&registry, &node, "test-model", "plain").await);
        assert!(node.issues_receipts("test-model"));
        assert!(unregister_runtime_instance(&registry, &node, "test-model", "receipted").await);
        assert!(!node.issues_receipts("test-model"));
    }

    /// A model as a receipt names it when its GGUF is a Hugging Face file.
    const PUBLIC: &str =
        "unsloth/Qwen3-4B-GGUF@0123456789abcdef0123456789abcdef01234567/Qwen3-4B-Q8_0.gguf";
    /// A runtime that issues receipts for every turn.
    const EVERY: RuntimeReceipts = RuntimeReceipts::EveryTurn;

    #[test]
    fn receipts_are_advertised_only_when_the_runtime_issues_them_for_chat() {
        let chat = ModelWorkloadClass::CausalGeneration;
        assert_eq!(receipts_advertisement(true, EVERY, chat, PUBLIC), Ok(()));
        // the setting alone advertises nothing: the recorder did not start
        assert_eq!(
            receipts_advertisement(true, RuntimeReceipts::None, chat, PUBLIC),
            Err(NoReceipts::RuntimeIssuesNone)
        );
        assert_eq!(
            receipts_advertisement(false, RuntimeReceipts::None, chat, PUBLIC),
            Err(NoReceipts::NotAsked)
        );
        // receipts cover chat completions, not embeddings or reranking
        for workload in [ModelWorkloadClass::Embedding, ModelWorkloadClass::Rerank] {
            assert_eq!(
                receipts_advertisement(true, EVERY, workload, PUBLIC),
                Err(NoReceipts::NotChat)
            );
        }
    }

    /// A hybrid stage gives a turn that continues a cached prefix no
    /// receipt, and a peer that advertises receipts is struck for every
    /// turn without one: an ordinary chat would block an honest node.
    #[test]
    fn a_hybrid_model_is_not_advertised_with_receipts() {
        let chat = ModelWorkloadClass::CausalGeneration;
        let hybrid = RuntimeReceipts::NotForCachedPrefixes;
        assert_eq!(
            receipts_advertisement(true, hybrid, chat, PUBLIC),
            Err(NoReceipts::RecurrentState)
        );
        assert_eq!(
            receipts_advertisement(false, hybrid, chat, PUBLIC),
            Err(NoReceipts::NotAsked)
        );
        let why = NoReceipts::RecurrentState.explanation().unwrap();
        assert!(why.contains("cached prefix"), "{why}");
        assert!(why.contains("recurrent state"), "{why}");
        assert!(!report_advertisement("m", Err(NoReceipts::RecurrentState)));
    }

    #[test]
    fn receipts_no_client_can_audit_are_not_advertised() {
        let chat = ModelWorkloadClass::CausalGeneration;
        // a GGUF served from a local path, a runtime's own name, a repo
        // without a file, and a file at a branch: nothing public and pinned
        // says what the node ran, so every audit would end as no audit
        for named in [
            "local-gguf/sha256-0123456789abcdef",
            "Qwen3-4B-Q8_0",
            "unsloth/Qwen3-4B-GGUF",
            "unsloth/Qwen3-4B-GGUF@main/Qwen3-4B-Q8_0.gguf",
            "",
        ] {
            assert!(!names_a_public_model(named), "{named}");
            assert_eq!(
                receipts_advertisement(true, EVERY, chat, named),
                Err(NoReceipts::NoPublicModel),
                "{named}"
            );
        }
        assert!(names_a_public_model(PUBLIC));
        // the names this node gives its own models
        let cached = Path::new(
            "/cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/snapshots/0123456789abcdef0123456789abcdef01234567/Qwen3-4B-Q8_0.gguf",
        );
        assert!(names_a_public_model(&receipt_model(cached, "Qwen3-4B")));
        let local = receipt_model(Path::new("/models/m.gguf"), "local-gguf/sha256-abc");
        assert!(!names_a_public_model(&local));
        let why = NoReceipts::NoPublicModel.explanation().unwrap();
        assert!(why.contains("local path"), "{why}");
    }

    #[test]
    fn a_model_without_receipts_is_explained_only_when_they_were_asked_for() {
        assert!(NoReceipts::NotAsked.explanation().is_none());
        for why in [
            NoReceipts::RuntimeIssuesNone,
            NoReceipts::NotChat,
            NoReceipts::RemoteStages(2),
            NoReceipts::NoPublicModel,
            NoReceipts::RecurrentState,
        ] {
            assert!(why.explanation().is_some(), "{why:?}");
            assert!(!report_advertisement("m", Err(why)));
        }
        assert!(report_advertisement("m", Ok(())));
        // a runtime with a quantized KV cache starts no recorder, so this is
        // what its operator reads
        let none = NoReceipts::RuntimeIssuesNone.explanation().unwrap();
        assert!(none.contains("KV cache is quantized"), "{none}");
        // and so does a runtime on Vulkan
        assert!(none.contains("Vulkan"), "{none}");
    }

    #[test]
    fn a_hugging_face_file_is_named_by_its_pinned_revision() {
        let path = Path::new(
            "/cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/snapshots/0123456789abcdef0123456789abcdef01234567/Qwen3-4B-Q8_0.gguf",
        );
        let named = receipt_model(path, "Qwen3-4B");
        assert!(named.starts_with("unsloth/Qwen3-4B-GGUF@"), "{named}");
        assert!(named.ends_with("Qwen3-4B-Q8_0.gguf"), "{named}");
    }

    #[test]
    fn a_split_gguf_is_named_by_its_first_shard() {
        let snapshot = "/cache/huggingface/hub/models--o--m/snapshots/0123456789abcdef0123456789abcdef01234567";
        let first = format!("{snapshot}/M-00001-of-00002.gguf");
        let named = receipt_model(Path::new(&first), "M");
        assert!(named.starts_with("o/m@"), "{named}");
        assert!(named.ends_with("/M-00001-of-00002.gguf"), "{named}");
        let second = format!("{snapshot}/M-00002-of-00002.gguf");
        assert_eq!(receipt_model(Path::new(&second), "M"), "M");
    }

    #[test]
    fn a_local_file_is_named_by_the_model_name() {
        assert_eq!(receipt_model(Path::new("/models/m.gguf"), "m"), "m");
    }

    #[tokio::test]
    async fn a_runtime_that_issues_receipts_is_advertised_with_them() {
        // names a client can tie to a Hugging Face file (`mesh/node/receipts_advertisement.rs`)
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .expect("test node should initialize");
        let registry = Arc::new(tokio::sync::Mutex::new(HashMap::new()));
        for (model_name, receipts) in [("acme/receipted:Q8_0", true), ("acme/plain:Q8_0", false)] {
            register_runtime_instance(
                &registry,
                &node,
                RuntimeModelRegistration {
                    primary_model_name: "acme/receipted:Q8_0",
                    model_name,
                    instance_id: model_name,
                    context_length: Some(4096),
                    thinking: None,
                    backend: None,
                    port: 0,
                    capabilities: crate::models::ModelCapabilities::default(),
                    workload_class: mesh::ModelWorkloadClass::CausalGeneration,
                    receipts,
                },
            )
            .await;
        }
        let advertised = |descriptors: Vec<mesh::ServedModelDescriptor>| {
            descriptors
                .into_iter()
                .map(|d| (d.identity.model_name, d.receipts))
                .collect::<HashMap<_, _>>()
        };
        let gossiped = advertised(node.local_descriptors_with_plugins().await);
        assert_eq!(gossiped.get("acme/receipted:Q8_0"), Some(&true));
        assert_eq!(gossiped.get("acme/plain:Q8_0"), Some(&false));

        node.remove_served_model_descriptor("acme/receipted:Q8_0")
            .await;
        assert!(
            !node
                .receipt_models
                .lock()
                .unwrap()
                .contains("acme/receipted:Q8_0")
        );
    }
}

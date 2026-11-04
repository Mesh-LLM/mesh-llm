//! Inference receipts for a split this node coordinates, when `[receipts]`
//! is on (`mesh/docs/design/RECEIPTS.md`). Stage 0 issues them and signs
//! them with the node key; clients open them on its OpenAI address
//! (`POST /v1/receipts/open`). Every later stage records its own part, and
//! stage 0 commits and opens that part over a receipts connection to the
//! stage's binary address, which crosses peers as the activations do.
//!
//! Recording is one thing, vouching for it another. A split is advertised
//! with receipts, and satisfies the `receipts` trust tier for this node's
//! own clients, only when all of it runs on this node
//! ([`split_receipts_advertisement`]). With a stage on another node:
//!
//! - nothing confirms that the node records its part. A peer that predates
//!   split receipts ignores the request and loads without a word, and the
//!   stage status it reports back carries no acknowledgement;
//! - the turns this node serves its own clients through the split are not
//!   audited: audits watch the turns a peer serves, and here the peers do
//!   the compute behind this node's own runtime. Every stage host has to
//!   satisfy the policy (`mesh/docs/design/TRUST_TIERS.md`), and none of
//!   them is checked.

use super::loading::SplitGenerationLoadSpec;
use crate::runtime::local::receipts::{
    NoReceipts, RuntimeReceipts, names_a_public_model, receipt_model, report_advertisement,
    warn_quantized_default_cache,
};

/// Whether every stage of the split is asked to record receipts.
pub(super) fn split_issues_receipts(spec: &SplitGenerationLoadSpec<'_>) -> bool {
    spec.mesh_config.receipts.enabled
}

/// What stage 0 needs to issue receipts, or `None` when the node issues
/// none.
pub(super) fn stage0_receipts(
    spec: &SplitGenerationLoadSpec<'_>,
) -> Option<skippy_serving::receipts::ReceiptsOptions> {
    warn_if_the_package_quantizes_the_cache(spec);
    split_issues_receipts(spec).then(|| {
        skippy_serving::receipts::ReceiptsOptions::embedded(
            stage0_receipt_model(spec),
            Some(spec.node.endpoint_secret_key.to_bytes()),
        )
    })
}

/// Says when the KV cache the stages load with is one no audit covers
/// because the layer package defaults to it, and names the setting that
/// fixes it. The stages resolve their cache types as
/// `local_package::split_effective_kv_cache_quant` does.
fn warn_if_the_package_quantizes_the_cache(spec: &SplitGenerationLoadSpec<'_>) {
    let package_default = crate::inference::skippy::KvCachePolicy::from_publisher_defaults(
        spec.package.publisher_defaults.as_ref(),
    )
    .guarded_for_model(Some(spec.compact_meta));
    warn_quantized_default_cache(
        split_issues_receipts(spec),
        spec.model_ref,
        [
            package_default.cache_type_k(),
            package_default.cache_type_v(),
        ],
        [spec.cache_type_k_override, spec.cache_type_v_override],
    );
}

/// The model the receipts of the split name.
fn stage0_receipt_model(spec: &SplitGenerationLoadSpec<'_>) -> String {
    receipt_model(&spec.package.source_model_path, spec.model_ref)
}

/// Whether a split is advertised with receipts and satisfies the `receipts`
/// trust tier: stage 0's runtime issues them for every turn, no stage
/// runs on another node (`remote_stages` is how many do), and the receipts
/// name a model a client can fetch (`receipt_model` is that name).
pub(super) fn split_receipts_advertisement(
    asked: bool,
    stage0: RuntimeReceipts,
    remote_stages: usize,
    receipt_model: &str,
) -> Result<(), NoReceipts> {
    if !asked {
        return Err(NoReceipts::NotAsked);
    }
    stage0.shortfall()?;
    if remote_stages > 0 {
        return Err(NoReceipts::RemoteStages(remote_stages));
    }
    if !names_a_public_model(receipt_model) {
        return Err(NoReceipts::NoPublicModel);
    }
    Ok(())
}

/// Whether the node advertises receipts for the split it just loaded, whose
/// stage 0 runtime is `stage0`.
pub(super) fn advertised(
    spec: &SplitGenerationLoadSpec<'_>,
    stage0: &crate::inference::skippy::SkippyModelHandle,
) -> bool {
    let this_node = spec.node.id();
    let remote_stages = spec
        .generation
        .stages
        .iter()
        .filter(|stage| stage.node_id != this_node)
        .count();
    report_advertisement(
        spec.model_ref,
        split_receipts_advertisement(
            split_issues_receipts(spec),
            RuntimeReceipts::of(stage0),
            remote_stages,
            &stage0_receipt_model(spec),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_with_a_stage_on_another_node_does_not_satisfy_the_receipts_tier() {
        const PUBLIC: &str = "o/m@0123456789abcdef0123456789abcdef01234567/m-00001-of-00002.gguf";
        const EVERY: RuntimeReceipts = RuntimeReceipts::EveryTurn;
        // every stage here: stage 0 vouches for what it ran itself
        assert_eq!(split_receipts_advertisement(true, EVERY, 0, PUBLIC), Ok(()));
        // a stage elsewhere: unconfirmed and unaudited, so no receipts tier
        for remote_stages in [1, 3] {
            assert_eq!(
                split_receipts_advertisement(true, EVERY, remote_stages, PUBLIC),
                Err(NoReceipts::RemoteStages(remote_stages))
            );
        }
        // and never from the setting alone
        assert_eq!(
            split_receipts_advertisement(true, RuntimeReceipts::None, 0, PUBLIC),
            Err(NoReceipts::RuntimeIssuesNone)
        );
        assert_eq!(
            split_receipts_advertisement(false, RuntimeReceipts::None, 1, PUBLIC),
            Err(NoReceipts::NotAsked)
        );
        // a split of a local package names nothing a client can fetch
        assert_eq!(
            split_receipts_advertisement(true, EVERY, 0, "local-gguf/sha256-abc"),
            Err(NoReceipts::NoPublicModel)
        );
        // a hybrid stage 0 gives a turn that continues a cached prefix no
        // receipt, which a peer that advertises receipts is struck for
        assert_eq!(
            split_receipts_advertisement(true, RuntimeReceipts::NotForCachedPrefixes, 0, PUBLIC),
            Err(NoReceipts::RecurrentState)
        );
    }
}

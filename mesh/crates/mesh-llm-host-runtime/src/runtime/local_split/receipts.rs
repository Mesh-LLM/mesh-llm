//! Inference receipts for a split this node coordinates, when `[receipts]`
//! is on (`mesh/docs/design/RECEIPTS.md`). Stage 0 issues them and signs
//! them with the node key; clients open them on its OpenAI address
//! (`POST /v1/receipts/open`). Every later stage records its own part, and
//! stage 0 commits and opens that part over a receipts connection to the
//! stage's binary address, which crosses peers as the activations do.

use super::loading::SplitGenerationLoadSpec;

/// Whether every stage of the split records receipts.
pub(super) fn split_issues_receipts(spec: &SplitGenerationLoadSpec<'_>) -> bool {
    spec.mesh_config.receipts.enabled
}

/// What stage 0 needs to issue receipts, or `None` when the node issues
/// none.
pub(super) fn stage0_receipts(
    spec: &SplitGenerationLoadSpec<'_>,
) -> Option<skippy_serving::receipts::ReceiptsOptions> {
    split_issues_receipts(spec).then(|| {
        skippy_serving::receipts::ReceiptsOptions::embedded(
            super::super::local::receipts::receipt_model(
                &spec.package.source_model_path,
                spec.model_ref,
            ),
            Some(spec.node.endpoint_secret_key.to_bytes()),
        )
    })
}

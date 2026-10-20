//! The native MTP generation declaration a package carries, inferred from
//! its source GGUF.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail, ensure};
use skippy_package_format::{
    Generation, SpeculativeDecoding, StrategyKind, StrategySpec, WindowPolicy,
};

use crate::source_inventory::SourceInventory;

/// Infers the native MTP generation declaration from the source GGUF
/// evidence. Fail-closed: emitted only when `{arch}.nextn_predict_layers` and
/// the `blk.<layer>.nextn.*` tensor names agree exactly — a wrong declaration
/// silently breaks speculative decoding for every consuming runtime.
pub(super) fn infer_native_mtp_generation(
    inventory: &SourceInventory,
    draft_depth: Option<u32>,
) -> Result<Option<Generation>> {
    let metadata = &inventory.shards[0].directory.metadata;
    let arch = metadata
        .get("general.architecture")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let key = format!("{arch}.nextn_predict_layers");
    let declared = metadata.contains_key(&key);
    let depth = metadata.get(&key).and_then(|value| value.as_u64());

    let mut unparseable = Vec::new();
    let indices = mtp_layer_indices(inventory, &mut unparseable);
    if !unparseable.is_empty() {
        bail!(
            "native MTP tensor names without a parseable `blk.<layer>.nextn.` prefix: {unparseable:?}"
        );
    }
    if declared && depth.is_none() {
        bail!("`{key}` is present in the source metadata but is not an integer");
    }
    if let Some(draft_depth) = draft_depth {
        ensure!(
            depth == Some(1),
            "--native-mtp-draft-depth needs a source with exactly one NextN block (`{key} = 1`)"
        );
        ensure!(
            (1..=skippy_ffi::NATIVE_MTP_MAX_DRAFT_TOKENS as u32).contains(&draft_depth),
            "--native-mtp-draft-depth must be between 1 and {}",
            skippy_ffi::NATIVE_MTP_MAX_DRAFT_TOKENS
        );
    }
    match depth {
        None if indices.is_empty() => Ok(None),
        Some(0) if indices.is_empty() => Ok(None),
        Some(depth) if depth > 1 => bail!(
            "source declares {depth}-step native MTP prediction; the runtime executes depth 1 \
             only, so a correct `generation.speculative_decoding` declaration cannot be emitted"
        ),
        Some(1) => {
            let Some(expected_layer) = inventory.layer_count.checked_sub(1) else {
                bail!("source declares native MTP but has no transformer layers");
            };
            let expected = vec![expected_layer];
            if indices != expected {
                bail!(
                    "native MTP evidence is inconsistent: `{key} = 1` but `.nextn.` tensors are \
                     at layers {indices:?}, expected {expected:?}"
                );
            }
            Ok(Some(native_mtp_generation(
                &indices,
                draft_depth.unwrap_or(1),
            )))
        }
        _ => bail!(
            "native MTP evidence is inconsistent: `.nextn.` tensors are present at layers \
             {indices:?} but `{key}` is {}",
            if declared { "0" } else { "absent" }
        ),
    }
}

fn mtp_layer_indices(inventory: &SourceInventory, unparseable: &mut Vec<String>) -> Vec<u32> {
    let mut indices = BTreeSet::new();
    for shard in &inventory.shards {
        for tensor in &shard.tensors.entries {
            let name = &tensor.name;
            if !name.contains(".nextn.") {
                continue;
            }
            match parse_mtp_layer_index(name) {
                Some(layer) => {
                    indices.insert(layer);
                }
                None => unparseable.push(name.clone()),
            }
        }
    }
    indices.into_iter().collect()
}

fn parse_mtp_layer_index(name: &str) -> Option<u32> {
    let after = name.strip_prefix("blk.")?;
    let (number, rest) = after.split_once('.')?;
    let layer: u32 = number.parse().ok()?;
    rest.starts_with("nextn.").then_some(layer)
}

/// The native-MTP generation declaration the host-runtime resolver executes
/// for `strategy=auto` and `strategy=mtp`: a single `mtp` strategy in the
/// inline `NativeMtp` form. One draft token uses a fixed window of 1; a
/// chained head drafts `draft_depth` tokens per proposal behind an adaptive
/// verify window that starts at two.
fn native_mtp_generation(layer_indices: &[u32], draft_depth: u32) -> Generation {
    let window_policy = if draft_depth == 1 {
        WindowPolicy {
            default: "fixed".to_string(),
            initial_window: 1,
            min_window: 1,
            max_window: 1,
            pipeline_depth: None,
        }
    } else {
        WindowPolicy {
            default: "adaptive".to_string(),
            initial_window: draft_depth.min(2),
            min_window: 1,
            max_window: draft_depth,
            pipeline_depth: None,
        }
    };
    Generation {
        // The native-MTP declaration is target-only by contract; publisher request
        // defaults are applied by the host resolver, not by this battery.
        request_defaults: None,
        speculative_decoding: Some(SpeculativeDecoding {
            default: "mtp".to_string(),
            proposers: BTreeMap::new(),
            strategies: BTreeMap::from([(
                "mtp".to_string(),
                StrategySpec {
                    kind: StrategyKind::NativeMtp {
                        proposer: None,
                        prediction_depth: Some(draft_depth),
                        layer_indices: layer_indices.to_vec(),
                        window_policy: Some(window_policy),
                    },
                },
            )]),
        }),
    }
}

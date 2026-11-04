//! Whether this verifier can audit a receipt at all.
//!
//! An audit needs the node's runtime, a rounding profile for every weight
//! type each stage multiplies on its backend, a KV cache whose rows the
//! attention check covers ([`crate::cache_types::AUDITED`]), and a
//! reference decoding of the output head. When one is missing the checks
//! would fail an honest node, so the receipt is not audited instead: an
//! audit that cannot run says nothing about the node
//! (`mesh/docs/design/RECEIPTS.md`).
//!
//! A node can choose a runtime or backend no verifier covers and so never
//! be audited. Its receipts then count as unaudited turns, which a client's
//! trust policy can weigh, and never as passes.

use crate::audit::{BackendFamily, profiles_for};
use crate::ggml_types::type_name;
use crate::head::head_weight;
use crate::weights::WeightFile;
use crate::wire::{RECEIPT_VERSION, Receipt};

/// Why this verifier, which runs `runtime`, cannot audit `receipt`, if it
/// cannot.
///
/// Everything this reads is in the receipt, so it is decided before the
/// challenge is sent: a node cannot wait for the challenge and then become
/// unauditable.
pub fn unauditable(receipt: &Receipt, weights: &WeightFile, runtime: &str) -> Option<String> {
    if receipt.version != RECEIPT_VERSION {
        return Some(format!(
            "the receipt is version {}, and this verifier checks version {RECEIPT_VERSION}",
            receipt.version
        ));
    }
    if let Some(binding) = &receipt.binding
        && binding.runtime != runtime
    {
        return Some(format!(
            "the node ran {}, and this verifier runs {runtime}",
            binding.runtime
        ));
    }
    for stage in &receipt.stages {
        if let Some(cache_type) = stage
            .cache_types
            .iter()
            .find(|name| !crate::cache_types::audited_name(name))
        {
            return Some(format!(
                "stage {} stored KV rows of type {cache_type}: an audit covers a cache of \
                 f16 or f32 rows, and no quantized one",
                stage.stage_index
            ));
        }
        let Some(family) = BackendFamily::of(&stage.backend) else {
            return Some(format!(
                "stage {} ran on backend {}, whose rounding no profile models: an audit \
                 covers CPU and CUDA stages",
                stage.stage_index, stage.backend
            ));
        };
        let layers = stage.layer_start..stage.layer_end;
        let uncovered = weights.tensors.values().find(|t| {
            t.is_matrix()
                && t.layer().is_some_and(|l| layers.contains(&l))
                && profiles_for(family, t.ggml_type).is_none()
        });
        if let Some(tensor) = uncovered {
            return Some(format!(
                "no rounding profile for {} weights ({}) on {family:?}",
                type_name(tensor.ggml_type),
                tensor.name
            ));
        }
    }
    match weights.tensors.get(head_weight(weights)) {
        Some(tensor) if !crate::dequant::supports(tensor.ggml_type) => Some(format!(
            "no reference decoding of {} output weights",
            type_name(tensor.ggml_type)
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_types::{GGML_F32, GGML_Q4_1, GGML_Q8_0};
    use crate::turn::TurnBinding;
    use crate::weights::test_gguf;
    use crate::wire::{RECEIPT_VERSION, StageCommitment};

    const RUNTIME: &str = "skippy-abi 1.2.3";

    fn receipt(backend: &str, runtime: Option<&str>) -> Receipt {
        Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r".into(),
            model: "m".into(),
            position_start: 0,
            position_end: 2,
            tokens: vec![1, 2],
            generation_start: 1,
            final_token: None,
            stages: vec![StageCommitment {
                stage_index: 0,
                layer_start: 0,
                layer_end: 1,
                root: "00".into(),
                leaf_count: 1,
                backend: backend.into(),
                cache_types: Vec::new(),
                prefix: None,
                ubatches: Vec::new(),
                recurrent_state: None,
            }],
            binding: runtime.map(|runtime| TurnBinding {
                client_key: String::new(),
                parent: None,
                runtime: runtime.into(),
                sampler: crate::turn::GREEDY.into(),
                node_key: String::new(),
                turn_root: String::new(),
                signature: String::new(),
            }),
        }
    }

    /// Only the tensor types matter here, so the tensors are written as f32
    /// and given their types after. Layer 1, which no stage of the receipt
    /// holds, has a type without profiles.
    fn weights(layer_type: u32, head_type: u32) -> WeightFile {
        let names = [
            "blk.0.attn_q.weight",
            "blk.1.attn_q.weight",
            "output.weight",
        ];
        let tensors: Vec<_> = names
            .iter()
            .map(|name| (*name, vec![32, 4], vec![0.0; 128]))
            .collect();
        let mut weights = test_gguf::open(&[], &tensors);
        for (name, ggml_type) in names.iter().zip([layer_type, GGML_Q4_1, head_type]) {
            weights.tensors.get_mut(*name).unwrap().ggml_type = ggml_type;
        }
        weights
    }

    #[test]
    fn a_covered_receipt_is_auditable() {
        let weights = weights(GGML_Q8_0, GGML_Q8_0);
        let bound = receipt("CUDA0", Some(RUNTIME));
        assert_eq!(unauditable(&bound, &weights, RUNTIME), None);
        assert_eq!(unauditable(&receipt("CPU", None), &weights, RUNTIME), None);
    }

    #[test]
    fn what_this_verifier_cannot_check_is_not_audited() {
        let covered = weights(GGML_Q8_0, GGML_F32);
        let other = receipt("CPU", Some("skippy-abi 1.2.4"));
        assert!(unauditable(&other, &covered, RUNTIME).is_some());
        // a backend without measured rounding profiles: an honest Vulkan
        // node fails audits, so its receipts are not audited, also when a
        // node built before Vulkan lost its profiles still issues them
        for backend in ["ROCm0", "Metal", "Vulkan0", "Vulkan1"] {
            let reason = unauditable(&receipt(backend, None), &covered, RUNTIME).unwrap();
            assert!(
                reason.contains(backend) && reason.contains("no profile models"),
                "{reason}"
            );
        }
        let q4_1 = weights(GGML_Q4_1, GGML_F32);
        let reason = unauditable(&receipt("CUDA0", None), &q4_1, RUNTIME).unwrap();
        assert!(reason.contains("blk.0.attn_q.weight"), "{reason}");
        let head = weights(GGML_Q8_0, 19);
        assert!(unauditable(&receipt("CPU", None), &head, RUNTIME).is_some());
    }

    #[test]
    fn another_receipt_version_or_cache_type_is_not_audited() {
        let covered = weights(GGML_Q8_0, GGML_F32);
        let mut old = receipt("CPU", None);
        old.version = RECEIPT_VERSION - 1;
        let reason = unauditable(&old, &covered, RUNTIME).unwrap();
        assert!(reason.contains("version"), "{reason}");
        // a cache that is not quantized is audited, and no other: an honest
        // stage with a quantized cache fails the loose attention check
        for (cache_type, audited) in [
            ("f16", true),
            ("f32", true),
            ("q8_0", false),
            ("q4_0", false),
            ("q4_1", false),
            ("type39", false),
        ] {
            let mut named = receipt("CPU", None);
            named.stages[0].cache_types = vec![cache_type.into()];
            let reason = unauditable(&named, &covered, RUNTIME);
            assert_eq!(reason.is_none(), audited, "{cache_type}: {reason:?}");
        }
    }
}

//! Auto model pack selection for Nostr smart-auto.

/// Model tiers by VRAM requirement (approximate loaded size × 1.1 headroom).
/// Model tiers for auto-selection, ordered largest-first.
/// min_vram = file_size * 1.1 rounded up. Prefer Qwen3 over 2.5 at same tier.
/// Parse a size string like "2.5GB" to GB as f64.
fn parse_size_gb(s: &str) -> f64 {
    s.trim_end_matches("GB").parse::<f64>().unwrap_or(0.0)
}

/// Build model tiers from the catalog, sorted largest first.
/// Each entry is (model_ref, min_vram_gb) where min_vram = file_size * 1.1.
/// Excludes draft models (< 1GB).
fn model_tiers() -> Vec<(String, f64)> {
    let _ = crate::models::remote_catalog::ensure_catalog();
    let mut tiers: Vec<_> = crate::models::remote_catalog::loaded_models()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|m| {
            let size = m.size.as_deref()?;
            if parse_size_gb(size) < 1.0 {
                return None;
            }
            (
                crate::models::remote_catalog_model_ref(&m),
                parse_size_gb(size) * 1.1,
            )
                .into()
        })
        .collect();
    tiers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    tiers
}

/// Pick the model to SERVE for `--auto` based on VRAM.
/// Returns a single-element vec (the model this node should load).
///
/// One model per node. Biggest model that fits with 15% KV cache headroom.
///
/// Tiers:
///   <8GB:    Qwen3-4B (2.5G)
///   8-24GB:  Gemma-4-E4B-it (4.6G)
///   24-50GB: Qwen3.5-27B (17G) — vision + text
///   50-63GB: GLM-4.7-Flash (18G) — fast, tool calling
///   63-179GB: Qwen3-Coder-Next (48G) — frontier coder ~85B
///   179GB+:  MiniMax-M2.5 (138G) — flagship
pub fn auto_model_pack(vram_gb: f64) -> Vec<String> {
    let local_models = crate::models::scan_local_models();
    let tiers = model_tiers();
    auto_model_pack_with(vram_gb, &local_models, &tiers, &catalog_ref)
}

fn catalog_ref(name: &str) -> String {
    crate::models::find_remote_catalog_model_exact(name)
        .map(|model| crate::models::remote_catalog_model_ref(&model))
        .unwrap_or_else(|| name.to_string())
}

fn auto_model_pack_with<F>(
    vram_gb: f64,
    local_models: &[String],
    tiers: &[(String, f64)],
    catalog_ref: &F,
) -> Vec<String>
where
    F: Fn(&str) -> String,
{
    // Helper: check if a model is on disk
    let on_disk = |name: &str| local_models.contains(&name.to_string());
    // Helper: model size from tiers
    let size_of = |name: &str| -> f64 {
        tiers
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| *s)
            .unwrap_or(0.0)
    };
    let usable = vram_gb * 0.85; // 15% headroom for KV cache

    // Opinionated packs — each is (generalist, optional specialist(s))
    // The order within a tier prefers: on-disk first, then opinionated default.
    struct Pack {
        min_vram: f64,
        models: Vec<String>,
    }
    let packs: Vec<Pack> = vec![
        // One model per tier. Node serves one model at a time.
        Pack {
            min_vram: 179.0,
            models: vec![catalog_ref("MiniMax-M2.5-Q4_K_M")],
        },
        Pack {
            min_vram: 63.0,
            models: vec![catalog_ref("Qwen3-Coder-Next-Q4_K_M")],
        },
        Pack {
            min_vram: 50.0,
            models: vec![catalog_ref("GLM-4.7-Flash-Q4_K_M")],
        },
        Pack {
            min_vram: 24.0,
            models: vec![catalog_ref("Qwen3.5-27B-Q4_K_M")],
        },
        Pack {
            min_vram: 8.0,
            models: vec![catalog_ref("Gemma-4-E4B-it-Q4_K_M")],
        },
        Pack {
            min_vram: 0.0,
            models: vec![catalog_ref("Qwen3-4B-Q4_K_M")],
        },
    ];

    // Find the best pack that fits
    for pack in packs {
        if vram_gb < pack.min_vram {
            continue;
        }
        // Check all models in the pack actually fit within usable VRAM
        let total: f64 = pack.models.iter().map(|m| size_of(m)).sum();
        if total <= usable {
            return pack.models.clone();
        }
    }

    // Fallback: largest single model that fits, prefer on-disk
    let on_disk_fit = tiers
        .iter()
        .find(|(name, min_vram)| *min_vram <= usable && on_disk(name));
    let any_fit = tiers.iter().find(|(_, min_vram)| *min_vram <= usable);

    let primary = on_disk_fit
        .or(any_fit)
        .map(|(name, _)| catalog_ref(name))
        .unwrap_or_else(|| catalog_ref("Qwen3-4B-Q4_K_M"));

    vec![primary]
}

/// Build the model refs advertised as demand hints for every VRAM tier.
fn demand_seed_models_with<F>(catalog_ref: &F) -> Vec<String>
where
    F: Fn(&str) -> String,
{
    [
        "Qwen3-Coder-Next-Q4_K_M",
        "Qwen3.5-27B-Q4_K_M",
        "GLM-4.7-Flash-Q4_K_M",
        "Qwen3-8B-Q4_K_M",
        "Qwen3-4B-Q4_K_M",
        "Qwen3-0.6B-Q4_K_M",
    ]
    .into_iter()
    .map(catalog_ref)
    .collect()
}

/// Legacy wrapper — returns serving models + demand seeds combined.
/// Used by `smart_auto` for the StartNew decision.
pub fn default_models_for_vram(vram_gb: f64) -> Vec<String> {
    let local_models = crate::models::scan_local_models();
    let tiers = model_tiers();
    default_models_for_vram_with(vram_gb, &local_models, &tiers, &catalog_ref)
}

fn default_models_for_vram_with<F>(
    vram_gb: f64,
    local_models: &[String],
    tiers: &[(String, f64)],
    catalog_ref: &F,
) -> Vec<String>
where
    F: Fn(&str) -> String,
{
    let mut models = auto_model_pack_with(vram_gb, local_models, tiers, catalog_ref);
    for m in demand_seed_models_with(catalog_ref) {
        if !models.contains(&m) {
            models.push(m);
        }
    }
    models
}

#[cfg(test)]
mod auto_pack_tests {
    use super::*;

    fn identity_ref(name: &str) -> String {
        name.to_string()
    }

    fn test_tiers() -> Vec<(String, f64)> {
        [
            ("MiniMax-M2.5-Q4_K_M", 151.8),
            ("Qwen3-Coder-Next-Q4_K_M", 52.8),
            ("GLM-4.7-Flash-Q4_K_M", 19.8),
            ("Qwen3.5-27B-Q4_K_M", 18.7),
            ("Gemma-4-E4B-it-Q4_K_M", 5.1),
            ("Qwen3-4B-Q4_K_M", 2.75),
        ]
        .into_iter()
        .map(|(name, size)| (name.to_string(), size))
        .collect()
    }

    fn test_auto_model_pack(vram_gb: f64) -> Vec<String> {
        auto_model_pack_with(vram_gb, &[], &test_tiers(), &identity_ref)
    }

    fn assert_single_pack_model(pack: &[String], alias: &str) {
        assert_eq!(pack.len(), 1);
        assert_eq!(pack[0], alias);
    }

    fn assert_contains_catalog_alias(models: &[String], alias: &str) {
        assert!(models.iter().any(|model| model == alias));
    }

    #[test]
    fn pack_4gb_starter() {
        let pack = test_auto_model_pack(4.0);
        assert_single_pack_model(&pack, "Qwen3-4B-Q4_K_M");
    }

    #[test]
    fn pack_8gb_single_model() {
        let pack = test_auto_model_pack(8.0);
        assert_single_pack_model(&pack, "Gemma-4-E4B-it-Q4_K_M");
    }

    #[test]
    fn pack_16gb_single() {
        let pack = test_auto_model_pack(16.0);
        assert_single_pack_model(&pack, "Gemma-4-E4B-it-Q4_K_M");
    }

    #[test]
    fn pack_24gb_vision() {
        let pack = test_auto_model_pack(24.0);
        assert_single_pack_model(&pack, "Qwen3.5-27B-Q4_K_M");
    }

    #[test]
    fn pack_50gb_glm_flash() {
        let pack = test_auto_model_pack(50.0);
        assert_single_pack_model(&pack, "GLM-4.7-Flash-Q4_K_M");
    }

    #[test]
    fn pack_63gb_frontier_coder() {
        let pack = test_auto_model_pack(63.0);
        assert_single_pack_model(&pack, "Qwen3-Coder-Next-Q4_K_M");
    }

    #[test]
    fn pack_85gb_frontier_coder() {
        let pack = test_auto_model_pack(85.0);
        assert_single_pack_model(&pack, "Qwen3-Coder-Next-Q4_K_M");
    }

    #[test]
    fn pack_206gb_minimax() {
        let pack = test_auto_model_pack(206.0);
        assert_single_pack_model(&pack, "MiniMax-M2.5-Q4_K_M");
    }

    #[test]
    fn pack_between_tiers_falls_through() {
        // 40GB: below 50GB tier, falls to 24GB tier (Qwen3.5-27B)
        let pack = test_auto_model_pack(40.0);
        assert_single_pack_model(&pack, "Qwen3.5-27B-Q4_K_M");
    }

    #[test]
    fn demand_seeds_are_separate() {
        let seeds = demand_seed_models_with(&identity_ref);
        assert!(seeds.len() >= 4);
        assert_contains_catalog_alias(&seeds, "Qwen3-0.6B-Q4_K_M");
        assert_contains_catalog_alias(&seeds, "Qwen3-Coder-Next-Q4_K_M");
    }

    #[test]
    fn default_models_includes_both() {
        let pack = test_auto_model_pack(30.0);
        let seeds = demand_seed_models_with(&identity_ref);
        let all = default_models_for_vram_with(30.0, &[], &test_tiers(), &identity_ref);
        // Pack models come first
        for m in &pack {
            assert!(
                all.contains(m),
                "pack model {m} missing from default_models"
            );
        }
        // Seeds are also present
        for m in &seeds {
            assert!(
                all.contains(m),
                "seed model {m} missing from default_models"
            );
        }
        // No duplicates
        let mut deduped = all.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(all.len(), deduped.len());
    }
}

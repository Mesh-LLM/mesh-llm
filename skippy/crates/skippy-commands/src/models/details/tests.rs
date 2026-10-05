use super::*;
use serial_test::serial;
use std::{cmp::Ordering, collections::HashMap};
struct HfRepoFixture {
    repo: String,
    siblings: Vec<String>,
    size_bytes: HashMap<String, u64>,
}
fn load_gemma_live_fixture() -> HfRepoFixture {
    let sizes = [
        ("gemma-4-31B-it-UD-Q4_K_XL.gguf", 17_000_000_000),
        ("gemma-4-31B-it-Q4_0.gguf", 16_000_000_000),
        (
            "BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf",
            40_000_000_000,
        ),
        (
            "BF16/gemma-4-31B-it-BF16-00002-of-00002.gguf",
            20_000_000_000,
        ),
    ];
    HfRepoFixture {
        repo: "unsloth/gemma-4-31B-it-GGUF".into(),
        siblings: sizes.iter().map(|(name, _)| (*name).to_string()).collect(),
        size_bytes: sizes
            .into_iter()
            .map(|(name, size)| (name.to_string(), size))
            .collect(),
    }
}
fn empty_catalog_guard() -> skippy_model_hf::remote_catalog::CatalogEntriesOverrideGuard {
    skippy_model_hf::remote_catalog::set_catalog_entries_for_test(Vec::new())
}

#[test]
fn quant_selector_resolves_to_single_file_gguf() {
    let fixture = load_gemma_live_fixture();
    let resolved = resolve_hf_file_from_siblings("UD-Q4_K_XL", &fixture.siblings).unwrap();
    assert_eq!(resolved, "gemma-4-31B-it-UD-Q4_K_XL.gguf");
}

#[test]
fn dotted_quant_selector_resolves_to_single_file_gguf() {
    let siblings = vec![
        "Qwen3-Tiny.Q2_K.gguf".to_string(),
        "Qwen3-Tiny.Q4_K_M.gguf".to_string(),
    ];
    let resolved = resolve_hf_file_from_siblings("Q2_K", &siblings).unwrap();
    assert_eq!(resolved, "Qwen3-Tiny.Q2_K.gguf");
}

#[test]
fn gemma_bf16_selector_resolves_to_first_split_shard() {
    let fixture = load_gemma_live_fixture();
    let resolved = resolve_hf_file_from_siblings("BF16", &fixture.siblings).unwrap();
    assert_eq!(resolved, "BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf");
}

#[test]
fn fit_aware_gguf_prefers_largest_comfortable_candidate() {
    let available = 20_000_000_000u64;
    let ordering = compare_gguf_candidates_by_fit(
        "repo/model-q4.gguf",
        Some(12_000_000_000),
        "repo/model-q5.gguf",
        Some(17_000_000_000),
        available,
    );
    assert_eq!(ordering, Ordering::Greater);
}

#[test]
fn fit_aware_gguf_prefers_smaller_when_both_too_large() {
    let available = 20_000_000_000u64;
    let ordering = compare_gguf_candidates_by_fit(
        "repo/model-q8.gguf",
        Some(29_000_000_000),
        "repo/model-bf16.gguf",
        Some(35_000_000_000),
        available,
    );
    assert_eq!(ordering, Ordering::Less);
}

#[test]
fn gemma_repo_default_prefers_q4_over_bf16_at_local_fit_budget() {
    let fixture = load_gemma_live_fixture();
    let q4 = fixture
        .size_bytes
        .get("gemma-4-31B-it-Q4_0.gguf")
        .copied()
        .expect("fixture Q4_0 size");
    let bf16 = fixture
        .size_bytes
        .get("BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf")
        .copied()
        .expect("fixture BF16 size");
    let available = 19_300_000_000u64;
    let ordering = compare_gguf_candidates_by_fit(
        "unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_0.gguf",
        Some(q4),
        "unsloth/gemma-4-31B-it-GGUF/BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf",
        Some(bf16),
        available,
    );
    assert_eq!(ordering, Ordering::Less);
}

#[test]
fn collect_show_gguf_variants_excludes_mmproj_and_nonfirst_split() {
    let siblings = vec![
        ("mmproj-BF16.gguf".to_string(), Some(1_200_000_000)),
        (
            "gemma-4-26B-A4B-it-UD-Q3_K_S-00002-of-00009.gguf".to_string(),
            Some(12_500_000_000),
        ),
        (
            "gemma-4-26B-A4B-it-UD-Q3_K_S-00001-of-00009.gguf".to_string(),
            Some(12_500_000_000),
        ),
        (
            "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".to_string(),
            Some(16_900_000_000),
        ),
    ];
    let files: Vec<_> = collect_show_gguf_variants_from_siblings(&siblings, 0)
        .into_iter()
        .map(|(file, _)| file)
        .collect();
    assert_eq!(
        files,
        vec![
            "gemma-4-26B-A4B-it-UD-Q3_K_S-00001-of-00009.gguf".to_string(),
            "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".to_string(),
        ]
    );
}

#[test]
fn collect_show_gguf_variants_uses_total_split_size() {
    let siblings = vec![
        (
            "IQ3_K/Kimi-K2.6-IQ3_K-00001-of-00012.gguf".to_string(),
            Some(6_912_800),
        ),
        (
            "IQ3_K/Kimi-K2.6-IQ3_K-00002-of-00012.gguf".to_string(),
            Some(45_004_320_032),
        ),
        (
            "IQ3_K/Kimi-K2.6-IQ3_K-00003-of-00012.gguf".to_string(),
            Some(45_669_680_480),
        ),
    ];
    let variants = collect_show_gguf_variants_from_siblings(&siblings, 0);
    assert_eq!(variants.len(), 1);
    assert_eq!(variants[0].0, "IQ3_K/Kimi-K2.6-IQ3_K-00001-of-00012.gguf");
    assert_eq!(variants[0].1, Some(90_680_913_312));
}

#[test]
fn collect_show_gguf_variants_orders_by_fit_when_memory_known() {
    let siblings = vec![
        ("model-UD-Q5_K_M.gguf".to_string(), Some(21_200_000_000)),
        ("model-UD-Q4_K_M.gguf".to_string(), Some(16_900_000_000)),
        ("model-UD-Q3_K_S.gguf".to_string(), Some(12_500_000_000)),
    ];
    let files: Vec<_> = collect_show_gguf_variants_from_siblings(&siblings, 19_300_000_000)
        .into_iter()
        .map(|(file, _)| file)
        .collect();
    assert_eq!(
        files,
        vec![
            "model-UD-Q4_K_M.gguf".to_string(),
            "model-UD-Q3_K_S.gguf".to_string(),
            "model-UD-Q5_K_M.gguf".to_string(),
        ]
    );
}

#[tokio::test]
#[serial]
async fn show_model_variants_accepts_selected_quant_ref() {
    let _catalog = empty_catalog_guard();
    let fixture = load_gemma_live_fixture();
    let _siblings_guard = RepoSiblingEntriesOverrideGuard::set(Arc::new({
        let repo = fixture.repo.clone();
        let siblings = fixture
            .siblings
            .iter()
            // Keep this fixture hermetic: missing sizes would trigger live HEAD requests.
            .map(|file| {
                (
                    file.clone(),
                    Some(fixture.size_bytes.get(file).copied().unwrap_or(1)),
                )
            })
            .collect::<Vec<_>>();
        move |requested_repo, requested_revision| {
            if requested_repo == repo && requested_revision == "main" {
                Some(siblings.clone())
            } else {
                None
            }
        }
    }));

    let variants = show_model_variants_with_progress("unsloth/gemma-4-31B-it-GGUF:BF16", |_| {})
        .await
        .unwrap()
        .expect("repo-backed GGUF refs should enumerate variants");

    assert!(!variants.is_empty());
    assert!(
        variants
            .iter()
            .any(|variant| { variant.exact_ref == "unsloth/gemma-4-31B-it-GGUF:BF16" })
    );
    assert!(
        variants
            .iter()
            .any(|variant| { variant.exact_ref == "unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL" })
    );
}

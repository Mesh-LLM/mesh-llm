//! Smart auto-join scoring and decisions.

use super::contracts::DiscoveredMesh;
#[cfg(test)]
use super::contracts::MeshListing;
use super::model_packs::default_models_for_vram;

// ---------------------------------------------------------------------------
// Smart auto-join: score meshes, detect staleness, prefer geo match
// ---------------------------------------------------------------------------

/// Is this mesh eligible for `--auto` when the user did not specify `--mesh-name`?
///
/// `--auto` joins the default community mesh. Eligible listings are:
///   - unnamed (the implicit default), or
///   - the blessed community name "mesh-llm".
///
/// Any other named mesh is still publicly discoverable on Nostr, but it is
/// not the default — the user must opt in by name via `--mesh-name`.
pub fn is_auto_eligible(mesh: &DiscoveredMesh) -> bool {
    match mesh.listing.name.as_deref() {
        None => true,
        Some(name) => name.eq_ignore_ascii_case("mesh-llm"),
    }
}

/// Score a mesh for auto-join. Higher = better.
/// Considers region match, capacity, and model availability.
/// Freshness is mostly irrelevant since Nostr listings expire at 120s (TTL=2×60s),
/// so anything we see from discover() is already reasonably fresh.
pub fn score_mesh(mesh: &DiscoveredMesh, _now_secs: u64, last_mesh_id: Option<&str>) -> i64 {
    let mut score: i64 = 100; // base score — if we can see it, it's alive

    // The canonical community mesh is an unnamed listing (`name: None`) — that
    // is what you get by default when you don't pass `--mesh-name`, and it's
    // what the public relay shows today. Give it a bonus so it ranks above
    // anything else in `--auto`. The literal name "mesh-llm" is treated as a
    // defensive alias for the same thing: nothing in the wild publishes with
    // that name right now, but older docs and test runs may, and if one ever
    // appears it should rank alongside unnamed rather than below it.
    //
    // Other named meshes are excluded from `--auto` entirely by
    // `is_auto_eligible`, so they don't get a score adjustment here — when
    // the user targets one via `--mesh-name`, the raw score is what matters
    // and any bonus or penalty would skew ranking.
    match mesh.listing.name.as_deref() {
        None => score += 300,
        Some(n) if n.eq_ignore_ascii_case("mesh-llm") => score += 300,
        Some(_) => {}
    }

    // Sticky preference: strong bonus for the mesh we were last on
    if let (Some(last_id), Some(mesh_id)) = (last_mesh_id, &mesh.listing.mesh_id)
        && last_id == mesh_id
    {
        score += 500; // strong preference, not infinite — dead/degraded mesh loses on other factors
    }

    // Capacity: prefer meshes that aren't full
    if mesh.listing.max_clients > 0 {
        if mesh.listing.client_count >= mesh.listing.max_clients {
            score -= 1000; // full — don't join
        } else {
            let headroom = mesh.listing.max_clients - mesh.listing.client_count;
            score += (headroom as i64).min(20); // some capacity bonus
        }
    }

    // Size: prefer meshes with more nodes (more resilient)
    score += (mesh.listing.node_count as i64).min(10) * 5;

    // Models: prefer meshes with more warm models
    score += (mesh.listing.serving.len() as i64) * 10;

    // Wanted models: mesh needs help — bonus if we'd be useful
    score += (mesh.listing.wanted.len() as i64) * 15;

    score
}

/// Decision from smart auto-join.
#[derive(Debug)]
pub enum AutoDecision {
    /// Ranked list of meshes to try joining (best first)
    Join {
        candidates: Vec<(String, DiscoveredMesh)>,
    },
    /// No suitable mesh found — start a new one with these models
    StartNew { models: Vec<String> },
}

/// Pick meshes to join, ranked by score, or decide to start a new one.
///
/// - Scores all discovered meshes (freshness, region, capacity)
/// - Filters out stale/full meshes
/// - Returns all viable candidates ranked by score so the caller
///   can probe each in order and fall back to the next on failure
pub fn smart_auto(
    meshes: &[DiscoveredMesh],
    my_vram_gb: f64,
    target_name: Option<&str>,
) -> AutoDecision {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let last_mesh_id = crate::mesh::load_last_mesh_id();

    // If target name is set, only consider meshes with that exact name.
    // Otherwise `--auto` considers only the community mesh: unnamed listings
    // plus the blessed name "mesh-llm". Other named meshes are still publicly
    // discoverable on Nostr but must be opted into by name via `--mesh-name`.
    let candidates: Vec<&DiscoveredMesh> = if let Some(target) = target_name {
        meshes
            .iter()
            .filter(|m| {
                m.listing
                    .name
                    .as_ref()
                    .map(|n| n.eq_ignore_ascii_case(target))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        meshes.iter().filter(|m| is_auto_eligible(m)).collect()
    };

    // Score and rank
    let mut scored: Vec<(&DiscoveredMesh, i64)> = candidates
        .iter()
        .map(|m| (*m, score_mesh(m, now, last_mesh_id.as_deref())))
        .collect();
    scored.sort_by_key(|entry| std::cmp::Reverse(entry.1));

    // Collect viable candidates.
    // If the user specified --mesh-name, take all candidates (they already
    // filtered by name above — the user explicitly asked for this mesh).
    // Otherwise, require positive score to filter out stale meshes.
    let viable: Vec<(String, DiscoveredMesh)> = scored
        .iter()
        .filter(|(_, score)| target_name.is_some() || *score > 0)
        .map(|(m, _)| (m.listing.invite_token.clone(), (*m).clone()))
        .collect();

    if !viable.is_empty() {
        return AutoDecision::Join { candidates: viable };
    }

    // No suitable mesh — recommend models for a new one based on VRAM
    let models = default_models_for_vram(my_vram_gb);
    AutoDecision::StartNew { models }
}

// ---------------------------------------------------------------------------
// Unit tests: score_mesh, smart_auto, MeshFilter
// ---------------------------------------------------------------------------
#[cfg(test)]
mod scoring_tests {
    use super::*;

    fn make_mesh(
        name: Option<&str>,
        mesh_id: Option<&str>,
        serving: &[&str],
        node_count: usize,
        vram: u64,
        clients: usize,
        max_clients: usize,
    ) -> DiscoveredMesh {
        DiscoveredMesh {
            listing: MeshListing {
                invite_token: format!("invite-{}", mesh_id.unwrap_or("test")),
                serving: serving.iter().map(|s| s.to_string()).collect(),
                wanted: vec![],
                on_disk: vec![],
                total_vram_bytes: vram,
                node_count,
                client_count: clients,
                max_clients,
                name: name.map(|s| s.to_string()),
                region: None,
                mesh_id: mesh_id.map(|s| s.to_string()),
            },
            publisher_npub: format!("npub-{}", mesh_id.unwrap_or("test")),
            published_at: 1000,
            expires_at: Some(2000),
        }
    }

    #[test]
    fn score_unnamed_community_mesh_bonus() {
        // Unnamed is the canonical community mesh and should get the bonus.
        let mesh = make_mesh(
            None,
            Some("abc"),
            &["Qwen3-8B-Q4_K_M"],
            3,
            48_000_000_000,
            1,
            10,
        );
        let score = score_mesh(&mesh, 1500, None);
        // base(100) + community(300) + headroom + nodes(15) + models(10)
        assert!(
            score > 400,
            "unnamed community mesh should score high, got {score}"
        );
    }

    #[test]
    fn score_mesh_llm_alias_matches_unnamed() {
        // "mesh-llm" is a defensive alias for the community mesh and must
        // score identically to an unnamed listing with equivalent stats.
        let unnamed = make_mesh(None, Some("u"), &["m1"], 2, 24_000_000_000, 0, 0);
        let alias = make_mesh(
            Some("mesh-llm"),
            Some("a"),
            &["m1"],
            2,
            24_000_000_000,
            0,
            0,
        );
        assert_eq!(
            score_mesh(&unnamed, 1500, None),
            score_mesh(&alias, 1500, None)
        );
    }

    #[test]
    fn public_listing_json_uses_public_discovery_schema() {
        let mesh = make_mesh(
            Some("mesh-llm"),
            Some("public-mesh"),
            &["Qwen3-8B-Q4_K_M"],
            2,
            24_000_000_000,
            0,
            0,
        );

        let listing_json = serde_json::to_value(&mesh.listing).expect("listing must serialize");
        let discovered_json = serde_json::to_value(&mesh).expect("discovered mesh must serialize");
        let listing = listing_json.as_object().expect("listing JSON object");
        let discovered = discovered_json.as_object().expect("discovered JSON object");

        let listing_keys = listing
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let discovered_keys = discovered
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(
            listing_keys,
            std::collections::BTreeSet::from([
                "invite_token",
                "serving",
                "wanted",
                "on_disk",
                "total_vram_bytes",
                "node_count",
                "client_count",
                "max_clients",
                "name",
                "mesh_id",
            ])
        );
        assert_eq!(
            discovered_keys,
            std::collections::BTreeSet::from([
                "listing",
                "publisher_npub",
                "published_at",
                "expires_at",
            ])
        );
        assert!(!listing.contains_key("control_endpoint"));
        assert!(!listing.contains_key("owner_control_endpoint"));
        assert!(!discovered.contains_key("control_endpoint"));
        assert!(!discovered.contains_key("owner_control_endpoint"));
    }

    #[test]
    fn score_other_named_mesh_no_community_bonus() {
        // Non-community named meshes are excluded from --auto entirely by
        // `is_auto_eligible`; within `score_mesh` they simply don't get the
        // community bonus. When the user targets one via --mesh-name, the
        // raw score is what's used to rank.
        let mesh = make_mesh(
            Some("bobs-cluster"),
            Some("xyz"),
            &["Qwen3-8B-Q4_K_M"],
            3,
            48_000_000_000,
            0,
            0,
        );
        let score = score_mesh(&mesh, 1500, None);
        // base(100) + nodes(15) + models(10) — no community bonus, no penalty
        assert!(
            score < 300,
            "non-community named mesh should not get community bonus, got {score}"
        );
        assert!(score > 0, "named mesh with real nodes should be positive");
    }

    #[test]
    fn other_named_mesh_not_auto_eligible() {
        let bobs = make_mesh(Some("bobs-cluster"), Some("x"), &[], 1, 0, 0, 0);
        let community = make_mesh(Some("mesh-llm"), Some("c"), &[], 1, 0, 0, 0);
        let community_caps = make_mesh(Some("MESH-LLM"), Some("c2"), &[], 1, 0, 0, 0);
        let unnamed = make_mesh(None, Some("u"), &[], 1, 0, 0, 0);
        assert!(!is_auto_eligible(&bobs));
        assert!(is_auto_eligible(&community));
        assert!(is_auto_eligible(&community_caps));
        assert!(is_auto_eligible(&unnamed));
    }

    #[test]
    fn score_full_mesh_penalty() {
        let mesh = make_mesh(
            None,
            Some("full"),
            &["Qwen3-8B-Q4_K_M"],
            2,
            16_000_000_000,
            5,
            5,
        );
        let score = score_mesh(&mesh, 1500, None);
        assert!(score < 0, "full mesh should score negative, got {score}");
    }

    #[test]
    fn score_sticky_mesh_bonus() {
        let mesh = make_mesh(
            None,
            Some("my-mesh"),
            &["Qwen3-8B-Q4_K_M"],
            2,
            16_000_000_000,
            0,
            0,
        );
        let score_sticky = score_mesh(&mesh, 1500, Some("my-mesh"));
        let score_fresh = score_mesh(&mesh, 1500, None);
        assert!(
            score_sticky > score_fresh + 400,
            "sticky bonus should be large, sticky={score_sticky} fresh={score_fresh}"
        );
    }

    #[test]
    fn score_more_nodes_better() {
        let small = make_mesh(
            None,
            Some("s"),
            &["Qwen3-8B-Q4_K_M"],
            1,
            8_000_000_000,
            0,
            0,
        );
        let big = make_mesh(
            None,
            Some("b"),
            &["Qwen3-8B-Q4_K_M"],
            5,
            40_000_000_000,
            0,
            0,
        );
        assert!(score_mesh(&big, 1500, None) > score_mesh(&small, 1500, None));
    }

    #[test]
    fn score_more_models_better() {
        let one = make_mesh(
            None,
            Some("1"),
            &["Qwen3-8B-Q4_K_M"],
            2,
            16_000_000_000,
            0,
            0,
        );
        let two = make_mesh(
            None,
            Some("2"),
            &["Qwen3-8B-Q4_K_M", "Qwen3-32B-Q4_K_M"],
            2,
            40_000_000_000,
            0,
            0,
        );
        assert!(score_mesh(&two, 1500, None) > score_mesh(&one, 1500, None));
    }
}

#[cfg(test)]
mod smart_auto_tests {
    use super::*;
    use std::ffi::OsString;

    struct HomeEnvGuard {
        previous: Option<OsString>,
    }

    impl HomeEnvGuard {
        fn set(path: &std::path::Path) -> Self {
            let previous = std::env::var_os("HOME");
            unsafe { std::env::set_var("HOME", path) };
            Self { previous }
        }
    }

    impl Drop for HomeEnvGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => unsafe { std::env::set_var("HOME", value) },
                None => unsafe { std::env::remove_var("HOME") },
            }
        }
    }

    fn make_mesh(
        name: Option<&str>,
        mesh_id: &str,
        serving: &[&str],
        node_count: usize,
        vram: u64,
        clients: usize,
        max_clients: usize,
    ) -> DiscoveredMesh {
        DiscoveredMesh {
            listing: MeshListing {
                invite_token: format!("invite-{mesh_id}"),
                serving: serving.iter().map(|s| s.to_string()).collect(),
                wanted: vec![],
                on_disk: vec![],
                total_vram_bytes: vram,
                node_count,
                client_count: clients,
                max_clients,
                name: name.map(|s| s.to_string()),
                region: None,
                mesh_id: Some(mesh_id.to_string()),
            },
            publisher_npub: format!("npub-{mesh_id}"),
            published_at: 1000,
            expires_at: Some(2000),
        }
    }

    #[test]
    fn smart_auto_both_community_aliases_eligible() {
        // Both unnamed listings and the "mesh-llm" alias are eligible for
        // `--auto` and score equally on name. With other factors equal they
        // tie at the same score; what matters here is that both appear as
        // candidates and that `"mesh-llm"` is no longer penalised relative
        // to unnamed.
        let meshes = vec![
            make_mesh(None, "ccc", &["Qwen3-8B-Q4_K_M"], 2, 24_000_000_000, 0, 0),
            make_mesh(
                Some("mesh-llm"),
                "aaa",
                &["Qwen3-8B-Q4_K_M"],
                2,
                24_000_000_000,
                0,
                0,
            ),
        ];
        let now = 1500;
        let unnamed_score = score_mesh(&meshes[0], now, None);
        let alias_score = score_mesh(&meshes[1], now, None);
        assert_eq!(
            unnamed_score, alias_score,
            "None and 'mesh-llm' should score equally as community aliases",
        );
        match smart_auto(&meshes, 8.0, None) {
            AutoDecision::Join { candidates } => {
                assert_eq!(candidates.len(), 2);
            }
            AutoDecision::StartNew { .. } => panic!("should join, not start new"),
        }
    }

    #[test]
    fn smart_auto_excludes_other_named_meshes() {
        // Without --mesh-name, --auto must only consider the community mesh
        // (unnamed or name == "mesh-llm"). Other named meshes — even though
        // they are publicly discoverable on Nostr — should never appear as
        // candidates unless the user targets them by name.
        let meshes = vec![
            make_mesh(
                Some("bobs-cluster"),
                "bbb",
                &["Qwen3-8B-Q4_K_M"],
                5,
                80_000_000_000,
                0,
                0,
            ),
            make_mesh(
                Some("alice-cluster"),
                "aac",
                &["Qwen3-8B-Q4_K_M"],
                3,
                24_000_000_000,
                0,
                0,
            ),
        ];
        match smart_auto(&meshes, 8.0, None) {
            AutoDecision::Join { .. } => {
                panic!("other named meshes must not be joined by --auto")
            }
            AutoDecision::StartNew { models } => {
                assert!(!models.is_empty());
            }
        }
    }

    #[test]
    fn smart_auto_larger_unnamed_beats_smaller_alias() {
        // Both unnamed and "mesh-llm" are eligible with the same name bonus.
        // With capacity as the tiebreaker, the larger unnamed mesh wins —
        // which is what we want, since unnamed is the canonical identity
        // of the public community mesh.
        let meshes = vec![
            make_mesh(
                None,
                "unnamed-1",
                &["Qwen3-8B-Q4_K_M"],
                5,
                40_000_000_000,
                0,
                0,
            ),
            make_mesh(
                Some("mesh-llm"),
                "alias-1",
                &["Qwen3-8B-Q4_K_M"],
                2,
                16_000_000_000,
                0,
                0,
            ),
        ];
        match smart_auto(&meshes, 8.0, None) {
            AutoDecision::Join { candidates } => {
                assert_eq!(candidates.len(), 2);
                assert_eq!(candidates[0].0, "invite-unnamed-1");
            }
            AutoDecision::StartNew { .. } => panic!("should join"),
        }
    }

    #[test]
    fn smart_auto_filters_full_mesh() {
        let meshes = vec![make_mesh(
            None,
            "full",
            &["Qwen3-8B-Q4_K_M"],
            2,
            16_000_000_000,
            10,
            10,
        )];
        match smart_auto(&meshes, 8.0, None) {
            AutoDecision::Join { candidates } => {
                // Full mesh should still appear (score might be negative but target_name is None
                // so it filters on score > 0)
                assert!(candidates.is_empty(), "full mesh should be filtered out");
            }
            AutoDecision::StartNew { models } => {
                assert!(!models.is_empty());
            }
        }
    }

    #[test]
    fn smart_auto_target_name_filters() {
        let meshes = vec![
            make_mesh(
                Some("mesh-llm"),
                "aaa",
                &["Qwen3-8B-Q4_K_M"],
                3,
                48_000_000_000,
                1,
                10,
            ),
            make_mesh(
                Some("private"),
                "bbb",
                &["Qwen3-32B-Q4_K_M"],
                2,
                40_000_000_000,
                0,
                0,
            ),
        ];
        match smart_auto(&meshes, 8.0, Some("private")) {
            AutoDecision::Join { candidates } => {
                assert!(!candidates.is_empty());
                // Only "private" mesh should match
                for (token, _) in &candidates {
                    assert_eq!(token, "invite-bbb");
                }
            }
            AutoDecision::StartNew { .. } => panic!("should find the named mesh"),
        }
    }

    #[test]
    fn smart_auto_empty_starts_new() {
        match smart_auto(&[], 24.0, None) {
            AutoDecision::StartNew { models } => {
                assert!(!models.is_empty());
            }
            AutoDecision::Join { .. } => panic!("no meshes should mean start new"),
        }
    }

    #[test]
    #[serial_test::serial]
    fn smart_auto_sticky_preference() {
        let temp = tempfile::tempdir().expect("temp home");
        let _home = HomeEnvGuard::set(temp.path());
        crate::mesh::save_last_mesh_id("sticky-mesh").expect("save last mesh id");

        let meshes = vec![
            make_mesh(None, "other", &["Qwen3-8B-Q4_K_M"], 3, 24_000_000_000, 0, 0),
            make_mesh(
                None,
                "sticky-mesh",
                &["Qwen3-8B-Q4_K_M"],
                2,
                16_000_000_000,
                0,
                0,
            ),
        ];
        let result = smart_auto(&meshes, 8.0, None);

        match result {
            AutoDecision::Join { candidates } => {
                assert!(!candidates.is_empty());
                // Sticky mesh should be first despite fewer nodes
                assert_eq!(candidates[0].0, "invite-sticky-mesh");
            }
            AutoDecision::StartNew { .. } => panic!("should join"),
        }
    }
}

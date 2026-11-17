use super::*;

const LAYER_PACKAGE_REF: &str = "hf://meshllm/Qwen3-8B-Q4_K_M-layers@main";

fn layer_package() -> skippy::SkippyPackageIdentity {
    skippy::SkippyPackageIdentity {
        package_ref: LAYER_PACKAGE_REF.to_string(),
        source_model_bytes: 1_000,
        layer_count: 10,
        ..package(10)
    }
}

fn layer_inventory(
    package: &skippy::SkippyPackageIdentity,
    available_layer_end: u32,
) -> skippy::StageLayerInventory {
    let layer_count = package.layer_count;
    skippy::StageLayerInventory {
        model_id: "model-a".to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        layer_count,
        ready_ranges: Vec::new(),
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: available_layer_end,
        }],
        missing_ranges: (available_layer_end < layer_count)
            .then_some(skippy::LayerRange {
                layer_start: available_layer_end,
                layer_end: layer_count,
            })
            .into_iter()
            .collect(),
        source_model_path: Some("/cache/layer-package".to_string()),
        source_model_bytes: Some(package.source_model_bytes),
        source_model_sha256: None,
        content_addressed_local_source: None,
        source_model_kind: skippy::SourceModelKind::LayerPackage,
    }
}

fn empty_inventory(package: &skippy::SkippyPackageIdentity) -> skippy::StageLayerInventory {
    skippy::StageLayerInventory {
        layer_count: 0,
        available_ranges: Vec::new(),
        missing_ranges: Vec::new(),
        source_model_path: None,
        source_model_bytes: None,
        source_model_kind: skippy::SourceModelKind::Unknown,
        ..layer_inventory(package, 0)
    }
}

fn package_signal(
    inventory: &skippy::StageLayerInventory,
    package: &skippy::SkippyPackageIdentity,
) -> std::result::Result<SplitParticipantPackageSignal, SplitParticipantExclusionReason> {
    split_inventory_package_signal_result(inventory, "model-a", package, true, false)
}

#[test]
fn peer_accepting_downloads_may_fetch_missing_hf_layers() {
    let package = layer_package();
    let signal = package_signal(&layer_inventory(&package, 4), &package);
    assert!(signal.is_ok_and(|signal| signal.missing_artifact_bytes > 0));

    assert_eq!(split_peer_download_refusal(false, signal), signal);
}

#[test]
fn peer_refusing_downloads_is_excluded_when_it_would_fetch_layers() {
    let package = layer_package();
    let signal = package_signal(&layer_inventory(&package, 4), &package);

    assert_eq!(
        split_peer_download_refusal(true, signal),
        Err(SplitParticipantExclusionReason::PeerModelDownloadsRefused)
    );
}

#[test]
fn peer_refusing_downloads_stays_eligible_with_the_package_on_disk() {
    let package = layer_package();
    let signal = package_signal(&layer_inventory(&package, 10), &package);
    assert!(signal.is_ok_and(|signal| signal.missing_artifact_bytes == 0));

    assert_eq!(split_peer_download_refusal(true, signal), signal);
}

#[test]
fn peer_refusing_downloads_names_the_refusal_for_an_uncached_package() {
    let package = layer_package();
    let signal = package_signal(&empty_inventory(&package), &package);
    assert_eq!(
        signal,
        Err(SplitParticipantExclusionReason::StageInventoryEmpty)
    );

    assert_eq!(
        split_peer_download_refusal(true, signal),
        Err(SplitParticipantExclusionReason::PeerModelDownloadsRefused)
    );
}

#[test]
fn peer_refusing_downloads_keeps_unrelated_exclusions() {
    let package = layer_package();
    let mut inventory = layer_inventory(&package, 10);
    inventory.manifest_sha256 = "other-manifest".to_string();
    let signal = package_signal(&inventory, &package);

    assert_eq!(
        split_peer_download_refusal(true, signal),
        Err(SplitParticipantExclusionReason::PackageManifestMismatch)
    );
}

#[test]
fn peer_model_download_refusal_is_a_reported_blocker() {
    let excluded = vec![SplitParticipantExclusion {
        node_id: make_id(7),
        reason: SplitParticipantExclusionReason::PeerModelDownloadsRefused,
    }];

    let labels = split_participant_blocker_labels(&excluded);

    assert_eq!(labels.len(), 1);
    assert!(labels[0].starts_with("peer_model_downloads_refused=1"));
    assert!(split_participant_next_step(&excluded).contains("runtime.peer_model_downloads"));
}

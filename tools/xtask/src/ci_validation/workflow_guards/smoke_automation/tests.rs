use super::*;

fn smoke() -> String {
    include_str!("../../../../../../.github/workflows/smoke.yml").into()
}

#[test]
fn swift_artifact_consumer_requires_restore_before_build_steps() {
    let source = include_str!("../../../../../../.github/workflows/swift-sdk-artifact.yml");
    consumers(&workflow_yaml::parse(source).unwrap()).unwrap();
    for (restore_index, build_step) in ["Build Swift target", "Build host-only Swift SDK input"]
        .into_iter()
        .enumerate()
    {
        let restore_marker = "      - name: Restore product automation\n";
        let (restore_start, _) = source
            .match_indices(restore_marker)
            .nth(restore_index)
            .unwrap();
        let restore_end = source[restore_start..]
            .find("\n      - name:")
            .map_or(source.len(), |offset| restore_start + offset + 1);
        let restore_step = &source[restore_start..restore_end];
        let without_restore = format!("{}{}", &source[..restore_start], &source[restore_end..]);
        let build_marker = format!("      - name: {build_step}\n");
        let (build_start, _) = without_restore.match_indices(&build_marker).next().unwrap();
        let build_end = without_restore[build_start..]
            .find("\n      - name:")
            .map_or(without_restore.len(), |offset| build_start + offset + 1);
        let mutated = format!(
            "{}{}\n{}",
            &without_restore[..build_end],
            restore_step,
            &without_restore[build_end..]
        );
        assert!(
            consumers(&workflow_yaml::parse(&mutated).unwrap()).is_err(),
            "restore must precede {build_step}"
        );
    }
}

#[test]
fn restored_and_standalone_paths_are_admitted() {
    consumers(&workflow_yaml::parse(&smoke()).unwrap()).unwrap();
}

#[test]
fn missing_restore_duplicate_prepare_and_source_substitution_are_rejected() {
    for (before, after) in [
        (
            "./.github/actions/restore-automation",
            "./.github/actions/prepare-automation",
        ),
        ("inputs.product_automation == ''", "true"),
        ("require_product: 'true'", "require_product: 'false'"),
        (
            "fromJson(inputs.product_automation).source_sha",
            "fromJson(inputs.authority).source_sha",
        ),
    ] {
        let source = smoke();
        assert!(source.contains(before));
        assert!(
            consumers(&workflow_yaml::parse(&source.replace(before, after)).unwrap()).is_err(),
            "{before}"
        );
    }
}

#[test]
fn shared_restore_cannot_prepare_over_verified_automation() {
    let source = include_str!("../../../../../../.github/actions/restore-smoke-inputs/action.yml");
    shared(&workflow_yaml::parse(source).unwrap()).unwrap();
    for before in [
        "(runner.os != 'Linux' || job.container.id != '') && env.MESH_LLM_AUTOMATION_BIN == ''",
        "runner.os == 'Linux' && job.container.id == '' && runner.environment == 'github-hosted' && env.MESH_LLM_AUTOMATION_BIN == ''",
    ] {
        assert!(source.contains(before));
        assert!(shared(&workflow_yaml::parse(&source.replace(before, "true")).unwrap()).is_err());
    }
}

#[test]
fn eligible_lane_must_need_and_forward_product_producer() {
    let source = include_str!("../../../../../../.github/workflows/ci-linux-lane.yml");
    for (before, after) in [
        (
            "needs: [product_cpu, product_automation_linux_x64]",
            "needs: [product_cpu]",
        ),
        (
            "product_automation: ${{ needs.product_automation_linux_x64.outputs.identity_json }}",
            "product_automation: ${{ needs.authority_linux_x64.outputs.identity_json }}",
        ),
    ] {
        let baseline = BTreeMap::from([(
            "ci-linux-lane.yml".into(),
            workflow_yaml::parse(source).unwrap(),
        )]);
        forwarding(&baseline).unwrap();
        assert!(source.contains(before));
        let mutated = BTreeMap::from([(
            "ci-linux-lane.yml".into(),
            workflow_yaml::parse(&source.replace(before, after)).unwrap(),
        )]);
        assert!(forwarding(&mutated).is_err());
    }
}

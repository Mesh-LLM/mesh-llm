use super::*;

fn smoke() -> String {
    include_str!("../../../../../../.github/workflows/smoke.yml").into()
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

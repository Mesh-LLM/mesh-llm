use super::*;

pub(super) fn current_attempt_inputs(jobs: &Node, index: u8) {
    let repair_name = format!("repair_{index}");
    let verify_name = format!("verify_{index}");
    let repair = jobs.get(&repair_name).unwrap();
    let verify = jobs.get(&verify_name).unwrap();
    for job in [repair, verify] {
        assert_eq!(
            field(job, "uses"),
            "./.github/workflows/llama-canary-family-pass.yml"
        );
        let with = job.get("with").unwrap();
        for key in ["source", "upstream"] {
            assert_eq!(
                field(with, key),
                format!("${{{{ needs.resolve.outputs.{key} }}}}")
            );
        }
    }
    assert_eq!(
        needs(verify),
        BTreeSet::from(["resolve", repair_name.as_str()])
    );
    let with = verify.get("with").unwrap();
    assert_eq!(
        field(with, "mode"),
        if index == 1 {
            "${{ needs.resolve.outputs.publish_selected == 'true' && 'pinned-build' || 'verify-build' }}"
        } else {
            "verify-build"
        }
    );
    for key in ["package", "identity", "head"] {
        assert_eq!(
            field(with, &format!("previous_{key}")),
            format!("${{{{ needs.{repair_name}.outputs.{key} }}}}")
        );
    }
    if index == 1 {
        assert_eq!(needs(repair), BTreeSet::from(["resolve", "preflight"]));
        assert_eq!(
            field(repair, "if"),
            "${{ !cancelled() && needs.preflight.result == 'success' }}"
        );
        assert_eq!(
            field(verify, "if"),
            "${{ !cancelled() && (needs.resolve.outputs.changed == 'true' || needs.resolve.outputs.publish_selected == 'true') && needs.repair_1.outputs.green == 'true' && needs.repair_1.outputs.retry_recovered != 'true' }}"
        );
    } else {
        let previous = format!("attempt_{}", index - 1);
        assert_eq!(
            needs(repair),
            BTreeSet::from(["resolve", previous.as_str()])
        );
        assert_eq!(
            field(repair, "if"),
            format!(
                "${{{{ !cancelled() && needs.resolve.outputs.changed == 'true' && needs.{previous}.outputs.state == 'repairable' }}}}"
            )
        );
        assert_eq!(
            field(verify, "if"),
            format!(
                "${{{{ !cancelled() && needs.{repair_name}.outputs.green == 'true' && needs.{repair_name}.outputs.retry_recovered != 'true' }}}}"
            )
        );
        let with = repair.get("with").unwrap();
        assert_eq!(field(with, "mode"), "repair-build");
        for key in ["package", "identity", "head", "feedback"] {
            assert_eq!(
                field(with, &format!("previous_{key}")),
                format!("${{{{ needs.{previous}.outputs.resume_{key} }}}}")
            );
        }
    }
    current_selector_inputs(jobs, index, &repair_name, &verify_name);
}
fn current_selector_inputs(jobs: &Node, index: u8, repair_name: &str, verify_name: &str) {
    let selector = jobs.get(&format!("attempt_{index}")).unwrap();
    let mut dependencies = BTreeSet::from(["resolve", repair_name, verify_name]);
    let previous = format!("attempt_{}", index - 1);
    if index > 1 {
        dependencies.insert(previous.as_str());
    }
    assert_eq!(needs(selector), dependencies);
    let (_, select) = named(selector, "id", "select");
    let env = select.get("env").unwrap();
    assert_eq!(
        field(env, "REPAIR_JSON"),
        format!("${{{{ toJSON(needs.{repair_name}) }}}}")
    );
    assert_eq!(
        field(env, "VERIFY_JSON"),
        format!("${{{{ toJSON(needs.{verify_name}) }}}}")
    );
    assert!(field(select, "run").contains("automation canary-receipts select-attempt --changed \"$CHANGED\" --repair-json \"$REPAIR_JSON\" --verification-json \"$VERIFY_JSON\""));
}

#[test]
fn retry_recovery_reaches_the_controller_only_through_reconciliation() {
    let worker = document(".github/workflows/llama-canary-family-pass.yml");
    let outputs = worker
        .get("on")
        .unwrap()
        .get("workflow_call")
        .unwrap()
        .get("outputs")
        .unwrap();
    assert_eq!(
        field(outputs.get("retry_recovered").unwrap(), "value"),
        "${{ jobs.reconcile.outputs.retry_recovered }}"
    );
    let reconcile_outputs = worker
        .get("jobs")
        .unwrap()
        .get("reconcile")
        .unwrap()
        .get("outputs")
        .unwrap();
    assert_eq!(
        field(reconcile_outputs, "retry_recovered"),
        "${{ steps.reconcile.outputs.retry_recovered }}"
    );
}

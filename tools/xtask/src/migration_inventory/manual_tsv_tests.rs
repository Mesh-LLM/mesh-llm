use super::{check_other_shard, fixture, scan};
use crate::command::DynResult;
use std::fs;

fn fixture_with_tsv() -> DynResult<(std::path::PathBuf, Vec<scan::Candidate>, serde_json::Value)> {
    let (root, observed, mut ledger) = fixture()?;
    let directory = root.join("docs/skippy/manual-smoke");
    fs::create_dir_all(&directory)?;
    let rows = [
        "key_path\tfixture_path\tstartup_apply_command\tmodel_identifier\tverification_command\texpected_result\tactual_evidence_path\tpass_fail_status",
        "model_fit.ctx_size\tfixtures/first.toml\tpython3 docs/skippy/manual-smoke/runtime_smoke.py --fixture fixtures/first.toml --model-path $MESH_LLM_SMOKE_MODEL_PATH\torg/model:q4\twait for /v1/models\tready\tevidence.txt\tPASS",
        "hardware.device\tfixtures/second.toml\tpython3 docs/skippy/manual-smoke/runtime_smoke.py --fixture fixtures/second.toml --model-path $MESH_LLM_SMOKE_MODEL_PATH\torg/model:q4\twait for /v1/models\tready\tevidence.txt\tPASS",
    ];
    fs::write(directory.join("manifest.tsv"), rows.join("\n"))?;
    ledger["manual_tsv_commands"] = serde_json::json!({
        "first": "python3 docs/skippy/manual-smoke/runtime_smoke.py --fixture fixtures/first.toml --model-path $MESH_LLM_SMOKE_MODEL_PATH",
        "second": "python3 docs/skippy/manual-smoke/runtime_smoke.py --fixture fixtures/second.toml --model-path $MESH_LLM_SMOKE_MODEL_PATH"
    });
    ledger["manual_tsv_rows"] = serde_json::json!([
        [
            2,
            "model_fit.ctx_size",
            "fixtures/first.toml",
            "org/model:q4",
            "first"
        ],
        [
            3,
            "hardware.device",
            "fixtures/second.toml",
            "org/model:q4",
            "second"
        ]
    ]);
    Ok((root, observed, ledger))
}

#[test]
fn rejects_missing_line() -> DynResult<()> {
    let (root, observed, mut ledger) = fixture_with_tsv()?;
    ledger["manual_tsv_rows"]
        .as_array_mut()
        .ok_or("missing rows")?
        .pop();
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("missing instruction"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn rejects_wrong_target() -> DynResult<()> {
    let (root, observed, mut ledger) = fixture_with_tsv()?;
    ledger["manual_tsv_commands"]["second"] = "python3 evals/unrelated.py --fixture fixtures/second.toml --model-path $MESH_LLM_SMOKE_MODEL_PATH".into();
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("command/target"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn rejects_duplicate_source_identity() -> DynResult<()> {
    let (root, observed, mut ledger) = fixture_with_tsv()?;
    let repeated = ledger["manual_tsv_rows"][0].clone();
    ledger["manual_tsv_rows"]
        .as_array_mut()
        .ok_or("missing rows")?
        .push(repeated);
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(
        error.to_string().contains("duplicate instruction"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn rejects_prose_as_command() -> DynResult<()> {
    let (root, observed, ledger) = fixture_with_tsv()?;
    let path = root.join("docs/skippy/manual-smoke/manifest.tsv");
    let original = fs::read_to_string(&path)?;
    let text = original.replace(
        "python3 docs/skippy/manual-smoke/runtime_smoke.py --fixture fixtures/second.toml --model-path $MESH_LLM_SMOKE_MODEL_PATH\torg/model:q4\twait for /v1/models",
        "not-run locally\torg/model:q4\tpython3 docs/skippy/manual-smoke/runtime_smoke.py --fixture fixtures/second.toml",
    );
    fs::write(path, text)?;
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(
        error.to_string().contains("mismatched manual TSV"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

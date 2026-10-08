//! Source-owned cache marker protocol; hosted cache isolation requires separate evidence.
use super::Node;
use std::collections::BTreeMap;
const PROTOCOL: &str = "[{\"workflow\":\"ci-quality-slice.yml\",\"job\":\"authority_sentinel\",\"step\":{\"name\":\"Restore trusted authority marker\",\"id\":\"restore_seed\",\"uses\":\"actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25\",\"with\":{\"path\":\".depot-authority-sentinel\",\"key\":\"${{ steps.validate.outputs.seed_key }}\",\"fail-on-cache-miss\":false}}},{\"workflow\":\"ci-quality-slice.yml\",\"job\":\"authority_sentinel\",\"step\":{\"name\":\"Validate trusted seed marker content on hit\",\"shell\":\"bash\",\"env\":{\"SENTINEL_ID\":\"${{ steps.validate.outputs.sentinel_id }}\",\"CACHE_HIT\":\"${{ steps.restore_seed.outputs.cache-hit }}\"},\"run\":\"set -euo pipefail\\nif [[ \\\"$CACHE_HIT\\\" != \\\"true\\\" ]]; then\\n  exit 0\\nfi\\nexpected_file=\\\"$(mktemp)\\\"\\ntrap 'rm -f \\\"$expected_file\\\"' EXIT\\nprintf 'mesh-llm-depot-authority-marker-v1\\\\nsentinel_id=%s\\\\n' \\\\\\n  \\\"$SENTINEL_ID\\\" > \\\"$expected_file\\\"\\nif [[ ! -f .depot-authority-sentinel/marker ]] ||\\n   ! cmp -s \\\"$expected_file\\\" .depot-authority-sentinel/marker; then\\n  echo \\\"trusted authority marker content mismatch\\\" >&2\\n  exit 1\\nfi\\n\"}},{\"workflow\":\"ci-quality-slice.yml\",\"job\":\"authority_sentinel\",\"step\":{\"name\":\"Replace with deterministic PR poison marker\",\"shell\":\"bash\",\"env\":{\"SENTINEL_ID\":\"${{ steps.validate.outputs.sentinel_id }}\",\"PR_NUMBER\":\"${{ steps.validate.outputs.pr_number }}\"},\"run\":\"set -euo pipefail\\nrm -rf -- .depot-authority-sentinel\\nmkdir -p .depot-authority-sentinel\\nprintf 'mesh-llm-depot-authority-pr-marker-v1\\\\nsentinel_id=%s\\\\npr_number=%s\\\\n' \\\\\\n  \\\"$SENTINEL_ID\\\" \\\"$PR_NUMBER\\\" > .depot-authority-sentinel/marker\\n\"}},{\"workflow\":\"ci-quality-slice.yml\",\"job\":\"authority_sentinel\",\"step\":{\"name\":\"Save PR poison marker\",\"uses\":\"actions/cache/save@caa296126883cff596d87d8935842f9db880ef25\",\"with\":{\"path\":\".depot-authority-sentinel\",\"key\":\"${{ steps.validate.outputs.poison_key }}\"}}},{\"workflow\":\"ci-quality-slice.yml\",\"job\":\"authority_sentinel\",\"step\":{\"name\":\"Clear local poison marker before proof restore\",\"shell\":\"bash\",\"run\":\"rm -rf -- .depot-authority-sentinel\"}},{\"workflow\":\"ci-quality-slice.yml\",\"job\":\"authority_sentinel\",\"step\":{\"name\":\"Restore saved PR poison marker\",\"id\":\"restore_poison\",\"uses\":\"actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25\",\"with\":{\"path\":\".depot-authority-sentinel\",\"key\":\"${{ steps.validate.outputs.poison_key }}\",\"fail-on-cache-miss\":true}}},{\"workflow\":\"ci-quality-slice.yml\",\"job\":\"authority_sentinel\",\"step\":{\"name\":\"Validate saved PR poison marker content\",\"shell\":\"bash\",\"env\":{\"SENTINEL_ID\":\"${{ steps.validate.outputs.sentinel_id }}\",\"PR_NUMBER\":\"${{ steps.validate.outputs.pr_number }}\",\"CACHE_HIT\":\"${{ steps.restore_poison.outputs.cache-hit }}\"},\"run\":\"set -euo pipefail\\nif [[ \\\"$CACHE_HIT\\\" != \\\"true\\\" ]]; then\\n  echo \\\"saved PR poison marker restore did not report a cache hit\\\" >&2\\n  exit 1\\nfi\\nexpected_file=\\\"$(mktemp)\\\"\\ntrap 'rm -f \\\"$expected_file\\\"' EXIT\\nprintf 'mesh-llm-depot-authority-pr-marker-v1\\\\nsentinel_id=%s\\\\npr_number=%s\\\\n' \\\\\\n  \\\"$SENTINEL_ID\\\" \\\"$PR_NUMBER\\\" > \\\"$expected_file\\\"\\nif [[ ! -f .depot-authority-sentinel/marker ]] ||\\n   ! cmp -s \\\"$expected_file\\\" .depot-authority-sentinel/marker; then\\n  echo \\\"saved PR poison marker content mismatch\\\" >&2\\n  exit 1\\nfi\\n\"}},{\"workflow\":\"ci-quality-slice.yml\",\"job\":\"authority_sentinel\",\"step\":{\"name\":\"Require trusted seed isolation after poison publication\",\"shell\":\"bash\",\"env\":{\"CACHE_HIT\":\"${{ steps.restore_seed.outputs.cache-hit }}\"},\"run\":\"set -euo pipefail\\nif [[ \\\"$CACHE_HIT\\\" == \\\"true\\\" ]]; then\\n  echo \\\"Trusted authority marker was readable by the PR probe\\\" >&2\\n  exit 1\\nfi\\necho \\\"Trusted seed was not readable; pending trusted-main verify-pr-write.\\\"\\n\"}},{\"workflow\":\"depot-canary.yml\",\"job\":\"seed_authority_marker\",\"step\":{\"name\":\"Prepare deterministic non-secret marker\",\"shell\":\"bash\",\"env\":{\"SENTINEL_ID\":\"${{ steps.validate.outputs.sentinel_id }}\"},\"run\":\"set -euo pipefail\\nmarker_dir=\\\".depot-authority-sentinel\\\"\\nmkdir -p \\\"$marker_dir\\\"\\nprintf 'mesh-llm-depot-authority-marker-v1\\\\nsentinel_id=%s\\\\n' \\\\\\n  \\\"$SENTINEL_ID\\\" > \\\"$marker_dir/marker\\\"\\n\"}},{\"workflow\":\"depot-canary.yml\",\"job\":\"seed_authority_marker\",\"step\":{\"name\":\"Save trusted authority marker\",\"uses\":\"actions/cache/save@caa296126883cff596d87d8935842f9db880ef25\",\"with\":{\"path\":\".depot-authority-sentinel\",\"key\":\"${{ steps.validate.outputs.seed_key }}\"}}},{\"workflow\":\"depot-canary.yml\",\"job\":\"seed_authority_marker\",\"step\":{\"name\":\"Clear local marker before trusted restore\",\"shell\":\"bash\",\"run\":\"rm -rf -- .depot-authority-sentinel\"}},{\"workflow\":\"depot-canary.yml\",\"job\":\"seed_authority_marker\",\"step\":{\"name\":\"Restore trusted authority marker\",\"id\":\"verify_marker\",\"uses\":\"actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25\",\"with\":{\"path\":\".depot-authority-sentinel\",\"key\":\"${{ steps.validate.outputs.seed_key }}\",\"fail-on-cache-miss\":true}}},{\"workflow\":\"depot-canary.yml\",\"job\":\"seed_authority_marker\",\"step\":{\"name\":\"Require trusted marker hit\",\"shell\":\"bash\",\"env\":{\"CACHE_HIT\":\"${{ steps.verify_marker.outputs.cache-hit }}\"},\"run\":\"set -euo pipefail\\n[[ \\\"$CACHE_HIT\\\" == \\\"true\\\" ]]\\n\"}},{\"workflow\":\"depot-canary.yml\",\"job\":\"seed_authority_marker\",\"step\":{\"name\":\"Validate trusted marker content\",\"shell\":\"bash\",\"env\":{\"SENTINEL_ID\":\"${{ steps.validate.outputs.sentinel_id }}\"},\"run\":\"set -euo pipefail\\nexpected_file=\\\"$(mktemp)\\\"\\ntrap 'rm -f \\\"$expected_file\\\"' EXIT\\nprintf 'mesh-llm-depot-authority-marker-v1\\\\nsentinel_id=%s\\\\n' \\\\\\n  \\\"$SENTINEL_ID\\\" > \\\"$expected_file\\\"\\nif [[ ! -f .depot-authority-sentinel/marker ]] ||\\n   ! cmp -s \\\"$expected_file\\\" .depot-authority-sentinel/marker; then\\n  echo \\\"trusted authority marker content mismatch\\\" >&2\\n  exit 1\\nfi\\n\"}},{\"workflow\":\"depot-canary.yml\",\"job\":\"verify_pr_write\",\"step\":{\"name\":\"Clear local poison marker before trusted restore\",\"shell\":\"bash\",\"run\":\"rm -rf -- .depot-authority-sentinel\"}},{\"workflow\":\"depot-canary.yml\",\"job\":\"verify_pr_write\",\"step\":{\"name\":\"Probe PR poison marker\",\"id\":\"probe\",\"uses\":\"actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25\",\"with\":{\"path\":\".depot-authority-sentinel\",\"key\":\"${{ steps.validate.outputs.poison_key }}\",\"fail-on-cache-miss\":false}}},{\"workflow\":\"depot-canary.yml\",\"job\":\"verify_pr_write\",\"step\":{\"name\":\"Validate PR poison marker content on hit\",\"shell\":\"bash\",\"env\":{\"SENTINEL_ID\":\"${{ steps.validate.outputs.sentinel_id }}\",\"PR_NUMBER\":\"${{ steps.validate.outputs.pr_number }}\",\"CACHE_HIT\":\"${{ steps.probe.outputs.cache-hit }}\"},\"run\":\"set -euo pipefail\\nif [[ \\\"$CACHE_HIT\\\" != \\\"true\\\" ]]; then\\n  exit 0\\nfi\\nexpected_file=\\\"$(mktemp)\\\"\\ntrap 'rm -f \\\"$expected_file\\\"' EXIT\\nprintf 'mesh-llm-depot-authority-pr-marker-v1\\\\nsentinel_id=%s\\\\npr_number=%s\\\\n' \\\\\\n  \\\"$SENTINEL_ID\\\" \\\"$PR_NUMBER\\\" > \\\"$expected_file\\\"\\nif [[ ! -f .depot-authority-sentinel/marker ]] ||\\n   ! cmp -s \\\"$expected_file\\\" .depot-authority-sentinel/marker; then\\n  echo \\\"trusted main observed an unexpected poison marker content\\\" >&2\\n  exit 1\\nfi\\n\"}},{\"workflow\":\"depot-canary.yml\",\"job\":\"verify_pr_write\",\"step\":{\"name\":\"Require PR poison marker miss\",\"shell\":\"bash\",\"env\":{\"CACHE_HIT\":\"${{ steps.probe.outputs.cache-hit }}\"},\"run\":\"set -euo pipefail\\nif [[ \\\"$CACHE_HIT\\\" == \\\"true\\\" ]]; then\\n  echo \\\"PR poison marker was visible to trusted main (cache isolation failure)\\\" >&2\\n  exit 1\\nfi\\n\"}}]";
fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn lines(value: &str) -> String {
    value
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    let protocol: Vec<serde_json::Value> =
        serde_json::from_str(PROTOCOL).map_err(|_| "invalid owned marker declaration")?;
    let mut positions = BTreeMap::new();
    let mut cache_counts = BTreeMap::new();
    for row in protocol {
        let workflow = row["workflow"].as_str().ok_or("marker workflow missing")?;
        let job = row["job"].as_str().ok_or("marker job missing")?;
        let steps = workflows
            .get(workflow)
            .and_then(|d| d.get("jobs"))
            .and_then(|d| d.get(job))
            .and_then(|d| d.get("steps"))
            .ok_or("marker job absent")?;
        let Node::Seq(steps) = steps else {
            return Err("marker steps must be a sequence".into());
        };
        let expected = &row["step"];
        let name = expected["name"]
            .as_str()
            .ok_or("marker step declaration absent")?;
        let found: Vec<_> = steps
            .iter()
            .enumerate()
            .filter(|(_, s)| text(s, "name") == Some(name))
            .collect();
        let [(index, step)] = found.as_slice() else {
            return Err(format!(
                "{workflow}/{job}: unique marker step required: {name}"
            ));
        };
        let key = (workflow.to_owned(), job.to_owned());
        if positions
            .insert(key.clone(), *index)
            .is_some_and(|previous| previous >= *index)
        {
            return Err("marker protocol order changed".into());
        }
        if step.get("if").is_some()
            || step.get("continue-on-error").is_some()
            || step.get("working-directory").is_some()
        {
            return Err("marker step may not bypass its owned protocol".into());
        }
        for field in ["id", "uses", "shell", "run"] {
            let wanted = expected[field].as_str();
            let actual = text(step, field);
            if field == "run" {
                if actual.map(lines) != wanted.map(lines) {
                    return Err(format!("{name}: executed marker body changed"));
                }
            } else if actual != wanted {
                return Err(format!("{name}: {field} changed"));
            }
        }
        for field in ["with", "env"] {
            let wanted = expected[field].as_object();
            let actual = step.get(field);
            match (wanted, actual) {
                (None, None) => (),
                (Some(expected), Some(Node::Map(entries))) if expected.len() == entries.len() => {
                    for (key, value) in expected {
                        let wanted = match value {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            _ => return Err("invalid marker scalar declaration".into()),
                        };
                        if text(actual.unwrap(), key) != Some(wanted.as_str()) {
                            return Err(format!("{name}: {field}.{key} changed"));
                        }
                    }
                }
                _ => return Err(format!("{name}: {field} protocol shape changed")),
            }
        }
        if expected["uses"]
            .as_str()
            .is_some_and(|u| u.starts_with("actions/cache/"))
        {
            *cache_counts.entry(key).or_insert(0usize) += 1;
        }
    }
    for ((workflow, job), expected) in cache_counts {
        let Node::Seq(steps) = workflows[&workflow]
            .get("jobs")
            .unwrap()
            .get(&job)
            .unwrap()
            .get("steps")
            .unwrap()
        else {
            unreachable!()
        };
        if steps
            .iter()
            .filter(|s| text(s, "uses").is_some_and(|u| u.starts_with("actions/cache/")))
            .count()
            != expected
        {
            return Err("undeclared marker cache phase added".into());
        }
    }
    Ok(())
}

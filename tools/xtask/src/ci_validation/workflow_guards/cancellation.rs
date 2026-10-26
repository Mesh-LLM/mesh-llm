//! Workflow status conditions must allow GitHub to cancel interrupted work.
use super::{Node, field};
use crate::command::DynResult;
use std::collections::BTreeMap;

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    for (name, workflow) in workflows {
        for (id, job) in workflow.get("jobs").map_or(&[][..], Node::entries) {
            condition(job).map_err(|error| format!("{name}: job {id}: {error}"))?;
            if let Some(Node::Seq(steps)) = job.get("steps") {
                for (index, step) in steps.iter().enumerate() {
                    condition(step)
                        .map_err(|error| format!("{name}: job {id} step {}: {error}", index + 1))?;
                }
            }
        }
    }
    Ok(())
}

fn condition(node: &Node) -> Result<(), &'static str> {
    if field(node, "if").is_some_and(always_call) {
        Err("always() status conditions resist cancellation; use a cancellable status gate")
    } else {
        Ok(())
    }
}

/// GitHub expressions use single quoted strings and doubled quote escapes.
/// Match function tokens outside those strings, allowing whitespace before `(`.
fn always_call(expression: &str) -> bool {
    let bytes = expression.as_bytes();
    let mut at = 0;
    let mut quoted = false;
    while at < bytes.len() {
        if bytes[at] == b'\'' {
            if quoted && bytes.get(at + 1) == Some(&b'\'') {
                at += 2;
                continue;
            }
            quoted = !quoted;
            at += 1;
        } else if quoted || !identifier_byte(bytes[at]) {
            at += 1;
        } else {
            let start = at;
            while at < bytes.len() && identifier_byte(bytes[at]) {
                at += 1;
            }
            let word = &bytes[start..at];
            let next = bytes[at..]
                .iter()
                .position(|byte| !byte.is_ascii_whitespace());
            if word.eq_ignore_ascii_case(b"always")
                && next.is_some_and(|offset| bytes[at + offset] == b'(')
            {
                return true;
            }
        }
    }
    false
}

fn identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_validation::lane_results::workflow_yaml;

    fn document(source: &str) -> BTreeMap<String, Node> {
        BTreeMap::from([("fixture.yml".into(), workflow_yaml::parse(source).unwrap())])
    }

    #[test]
    fn job_and_step_status_calls_refuse_cancellation_resistance() {
        for call in ["always()", "always ()", "ALWAYS()", "success() || always()"] {
            for source in [
                format!("jobs:\n  build:\n    if: ${{{{ {call} }}}}\n"),
                format!(
                    "jobs:\n  build:\n    steps:\n      - if: ${{{{ {call} }}}}\n        run: echo fixture\n"
                ),
            ] {
                assert!(check(&document(&source)).is_err(), "{source}");
            }
        }
    }

    #[test]
    fn cancellation_and_success_failure_conditions_remain_allowed() {
        for condition in [
            "!cancelled()",
            "success() || failure() || cancelled()",
            "!cancelled() && needs.build.result == 'success'",
        ] {
            let source = format!("jobs:\n  build:\n    if: ${{{{ {condition} }}}}\n");
            assert!(check(&document(&source)).is_ok());
        }
    }

    #[test]
    fn quoted_expression_values_and_similar_identifiers_are_data() {
        for condition in [
            "inputs.label == 'always()'",
            "inputs.label == 'it''s always()'",
            "inputs.always == 'yes'",
            "inputs.always()",
            "not_always()",
        ] {
            assert!(!always_call(condition), "{condition}");
        }
        assert!(always_call("inputs.label == 'always()' || always()"));
    }

    #[test]
    fn shell_text_names_comments_and_inputs_do_not_become_status_gates() {
        let source = "# always()\nname: always()\ninputs:\n  label:\n    default: always()\njobs:\n  build:\n    steps:\n      - name: always()\n        run: |\n          echo always()\n";
        assert!(check(&document(source)).is_ok());
    }

    #[test]
    fn checked_in_workflow_status_conditions_remain_cancellable() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut workflows = BTreeMap::new();
        for entry in std::fs::read_dir(root.join(".github/workflows")).unwrap() {
            let path = entry.unwrap().path();
            if !matches!(
                path.extension().and_then(|v| v.to_str()),
                Some("yml" | "yaml")
            ) {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            workflows.insert(
                path.file_name().unwrap().to_string_lossy().into_owned(),
                workflow_yaml::parse(&source).unwrap(),
            );
        }
        assert!(!workflows.is_empty());
        check(&workflows).unwrap();
    }
}

use super::{Node, field};
use crate::command::DynResult;

fn default_shell(node: &Node) -> Option<&str> {
    node.get("defaults")
        .and_then(|defaults| defaults.get("run"))
        .and_then(|run| field(run, "shell"))
}

pub(super) fn check_expressions(source: &str) -> DynResult<()> {
    for expression in source
        .split("${{")
        .skip(1)
        .filter_map(|rest| rest.split_once("}}"))
    {
        for branch in expression.0.split("&&").skip(1) {
            if let Some((value, _)) = branch.split_once("||")
                && matches!(value.trim(), "''" | "\"\"" | "0" | "false")
            {
                return Err("falsy ternary branch overrides the selected value".into());
            }
        }
    }
    Ok(())
}

pub(super) fn check_containers(document: &Node) -> DynResult<()> {
    for (name, job) in document.get("jobs").into_iter().flat_map(Node::entries) {
        if job.get("container").is_none() {
            continue;
        }
        let shell = default_shell(job).or_else(|| default_shell(document));
        let Some(Node::Seq(steps)) = job.get("steps") else {
            continue;
        };
        for step in steps {
            let Some(run) = field(step, "run") else {
                continue;
            };
            if field(step, "shell")
                .or(shell)
                .is_some_and(|shell| shell.eq_ignore_ascii_case("bash"))
            {
                continue;
            }
            if has_bash_syntax(run) {
                return Err(
                    format!("container job {name} uses Bash syntax without shell: bash").into(),
                );
            }
        }
    }
    Ok(())
}

fn has_bash_syntax(run: &str) -> bool {
    ["<<<", "[[", "&>", "$RANDOM", "+=("]
        .iter()
        .any(|pattern| run.contains(pattern))
        || run
            .split("${")
            .skip(1)
            .filter_map(|part| part.split_once('}'))
            .any(|(expansion, _)| {
                ["//", "^^", ",,"]
                    .iter()
                    .any(|pattern| expansion.contains(pattern))
            })
        || run
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .any(|word| matches!(word, "pipefail" | "mapfile" | "readarray"))
        || run.lines().any(|line| {
            line.trim_start().starts_with("function ")
                || line.trim_start().starts_with("declare -a")
                || line.trim_start().starts_with("source ")
        })
        || run
            .split_whitespace()
            .collect::<Vec<_>>()
            .windows(2)
            .any(|words| words == ["echo", "-e"])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_validation::lane_results::workflow_yaml;

    #[test]
    fn rejects_falsy_branches_but_accepts_truthy_selection() {
        for value in ["''", "\"\"", "0", "false"] {
            assert!(
                check_expressions(&format!("${{{{ selected && {value} || fallback }}}}")).is_err()
            );
        }
        assert!(check_expressions("${{ selected && image || '' }}").is_ok());
    }
    #[test]
    fn rejects_implicit_shell_and_honors_step_job_workflow_defaults() {
        let source =
            "jobs:\n  build:\n    container: image\n    steps:\n      - run: set -o pipefail\n";
        assert!(check_containers(&workflow_yaml::parse(source).unwrap()).is_err());
        for source in [
            source.replace("- run:", "- shell: bash\n        run:"),
            source.replace(
                "container:",
                "defaults:\n      run:\n        shell: bash\n    container:",
            ),
            format!("defaults:\n  run:\n    shell: bash\n{source}"),
        ] {
            assert!(check_containers(&workflow_yaml::parse(&source).unwrap()).is_ok());
        }
    }
    #[test]
    fn accepts_posix_arithmetic_without_bash() {
        let document = workflow_yaml::parse(
            "jobs:\n  build:\n    container: image\n    steps:\n      - run: value=$((1 + 2))\n",
        )
        .unwrap();
        assert!(check_containers(&document).is_ok());
    }

    #[test]
    fn source_argument_is_not_a_source_builtin() {
        assert!(!has_bash_syntax("cargo xtool --source source.json"));
        assert!(has_bash_syntax("source setup.sh"));
    }
}

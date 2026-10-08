use crate::command::DynResult;
use serde::Deserialize;
use std::{collections::BTreeMap, fs::OpenOptions, io::Write};

#[derive(Deserialize)]
struct Job {
    result: String,
    #[serde(default)]
    outputs: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Needs {
    resolve: Job,
    preflight: Job,
    candidate: Job,
    verification: Job,
}

impl Job {
    fn output(&self, name: &str) -> &str {
        self.outputs.get(name).map(String::as_str).unwrap_or("")
    }
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!(
            "usage: cargo xtool automation canary-receipts result\nReads NEEDS_JSON; appends publication outputs to GITHUB_OUTPUT only after successful independent verification."
        );
        return Ok(());
    }
    if !args.is_empty() {
        return Err("canary result accepts no arguments; provide NEEDS_JSON".into());
    }
    let needs = std::env::var("NEEDS_JSON")?;
    let outputs = decide(&needs)?;
    if let Some(outputs) = outputs {
        let path = std::env::var_os("GITHUB_OUTPUT")
            .filter(|path| !path.is_empty())
            .ok_or("GITHUB_OUTPUT is required for publication outputs")?;
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(outputs.as_bytes())?;
        file.flush()?;
    }
    Ok(())
}

fn boolean_output(job: &Job, name: &str) -> DynResult<bool> {
    match job.output(name) {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("resolve output {name} must be true or false").into()),
    }
}

fn decide(input: &str) -> DynResult<Option<String>> {
    let needs: Needs = serde_json::from_str(input)?;
    if needs.resolve.result != "success" {
        return Err("trusted source resolution failed".into());
    }
    if !boolean_output(&needs.resolve, "certify")? {
        return Ok(None);
    }
    if needs.preflight.result != "success" {
        return Err(
            "canary environment preflight failed; candidate source was not evaluated".into(),
        );
    }
    let changed = boolean_output(&needs.resolve, "changed")?;
    let selected = needs.resolve.output("mesh_source");
    let candidate = &needs.candidate;
    if !selected.is_empty() {
        if changed
            || candidate.result != "success"
            || candidate.output("green") != "true"
            || candidate.output("head") != selected
        {
            return Err("selected MeshLLM revision certification failed".into());
        }
        return Ok(None);
    }
    if !changed {
        if candidate.result != "success" || candidate.output("green") != "true" {
            return Err("unchanged-pin family certification failed".into());
        }
        return Ok(None);
    }
    if candidate.result != "success" || candidate.output("green") != "true" {
        return Err(candidate_failure(candidate).into());
    }
    let verification = &needs.verification;
    if verification.result != "success"
        || verification.output("green") != "true"
        || candidate.output("head").is_empty()
        || candidate.output("head") != verification.output("head")
    {
        return Err("independent family verification failed; publication denied".into());
    }
    publication_outputs(verification).map(Some)
}

fn candidate_failure(candidate: &Job) -> String {
    let class = match candidate.output("failure_class") {
        "" => "candidate",
        class => class,
    };
    let stage = match candidate.output("failure_stage") {
        "" => "build-or-family-certification",
        stage => stage,
    };
    format!("{class} failure during {stage}; publication denied")
}

fn publication_outputs(verification: &Job) -> DynResult<String> {
    let mut output = String::new();
    for key in ["package", "identity", "head", "branch"] {
        let value = verification.output(key);
        if value.is_empty() || value.contains(['\r', '\n', '\0']) {
            return Err(format!(
                "verified publication output {key} must be a nonempty single line"
            )
            .into());
        }
        output.push_str(key);
        output.push('=');
        output.push_str(value);
        output.push('\n');
    }
    // The approval flag comes last, after every identity line has been written.
    output.push_str("publish=true\n");
    Ok(output)
}

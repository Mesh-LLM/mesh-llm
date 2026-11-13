//! Verify the exact downstream packaging attempt and its readiness artifact.
use crate::command::DynResult;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::Value as Json;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const REPO: &str = "Mesh-LLM/mesh-packaging";
const LIMIT: usize = 1024 * 1024;
const MAX_PAGES: usize = 10;
const MAX_RECEIPTS: usize = 16;

struct Expected {
    tag: String,
    sha: String,
    manifest: String,
    correlation: String,
    since: String,
}

fn valid_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn valid_utc_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 20
        && [
            (4, b'-'),
            (7, b'-'),
            (10, b'T'),
            (13, b':'),
            (16, b':'),
            (19, b'Z'),
        ]
        .iter()
        .all(|(index, character)| bytes[*index] == *character)
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| [4, 7, 10, 13, 16, 19].contains(&index) || byte.is_ascii_digit())
}

fn parse(args: &[String]) -> Result<Expected, String> {
    if args.len() != 10 {
        return Err(
            "expected --tag TAG --sha SHA --manifest SHA256 --correlation ID --since UTC_TIMESTAMP"
                .into(),
        );
    }
    let mut values = BTreeMap::new();
    for pair in args.as_chunks::<2>().0 {
        if values.insert(pair[0].as_str(), pair[1].clone()).is_some() {
            return Err("duplicate argument".into());
        }
    }
    let get = |key| {
        values
            .get(key)
            .cloned()
            .ok_or_else(|| format!("missing {key}"))
    };
    let expected = Expected {
        tag: get("--tag")?,
        sha: get("--sha")?,
        manifest: get("--manifest")?,
        correlation: get("--correlation")?,
        since: get("--since")?,
    };
    if !valid_hex(&expected.sha, 40)
        || !valid_hex(&expected.manifest, 64)
        || expected.tag.is_empty()
        || expected.correlation.is_empty()
        || expected.correlation.len() > 128
        || !expected
            .correlation
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
        || !valid_utc_timestamp(&expected.since)
    {
        return Err("invalid release identity or UTC timestamp".into());
    }
    Ok(expected)
}

fn eq_str(value: &Json, expected: &str, what: &str) -> Result<(), String> {
    if value.as_str() == Some(expected) {
        Ok(())
    } else {
        Err(format!("packaging receipt {what} mismatch"))
    }
}

fn validate_receipt(
    value: &Json,
    expected: &Expected,
    run: u64,
    attempt: u64,
) -> Result<(), String> {
    eq_str(&value["schema"], "mesh-packaging-readiness-v1", "schema")?;
    eq_str(&value["status"], "success", "status")?;
    eq_str(
        &value["correlation_id"],
        &expected.correlation,
        "correlation",
    )?;
    if value["packaging_run_id"].as_u64() != Some(run)
        || value["packaging_run_attempt"].as_u64() != Some(attempt)
    {
        return Err("packaging receipt run or attempt mismatch".into());
    }
    let upstream = &value["upstream"];
    eq_str(&upstream["repository"], "Mesh-LLM/mesh-llm", "repository")?;
    eq_str(&upstream["ref"], &expected.tag, "tag")?;
    eq_str(&upstream["sha"], &expected.sha, "tag commit")?;
    eq_str(
        &upstream["manifest_sha256"],
        &expected.manifest,
        "manifest digest",
    )?;
    for key in ["publish_images", "publish_release_assets", "publish_npm"] {
        eq_str(&value["requested"][key], "true", key)?;
    }
    for key in [
        "plan",
        "upstream",
        "package_images",
        "homebrew",
        "release_assembly",
        "npm_addons",
        "npm_package",
        "npm_preflight",
        "publish_npm",
        "image_index",
        "promote_images",
        "publish_release_assets",
    ] {
        eq_str(&value["results"][key], "success", key)?;
    }
    Ok(())
}

fn gh(args: &[String]) -> DynResult<Vec<u8>> {
    let cwd = std::env::current_dir()?.canonicalize()?;
    let executable = std::env::split_paths(&std::env::var_os("PATH").ok_or("PATH is missing")?)
        .map(|p| p.join("gh"))
        .find(|p| p.is_file())
        .ok_or("gh is missing")?;
    let env = std::env::vars_os()
        .map(|(k, v)| {
            let secret = matches!(
                k.to_str(),
                Some(
                    "GH_TOKEN" | "GITHUB_TOKEN" | "GH_ENTERPRISE_TOKEN" | "GITHUB_ENTERPRISE_TOKEN"
                )
            );
            (
                k,
                if secret && !v.is_empty() {
                    Value::Secret(v)
                } else {
                    Value::Public(v)
                },
            )
        })
        .collect::<BTreeMap<OsString, Value>>();
    gh_from(args, executable, cwd, env)
}

fn gh_from(
    args: &[String],
    executable: PathBuf,
    cwd: PathBuf,
    env: BTreeMap<OsString, Value>,
) -> DynResult<Vec<u8>> {
    let spec = ProcessSpec {
        executable,
        arguments: args
            .iter()
            .map(|a| Value::Public(a.clone().into()))
            .collect(),
        cwd,
        environment: env,
    };
    let limits = Limits {
        execution: Duration::from_secs(60),
        graceful_shutdown: Duration::from_secs(2),
        forced_shutdown: Duration::from_secs(5),
        retained_bytes_per_stream: LIMIT,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let result = process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(LIMIT),
            stderr: NonZeroUsize::new(LIMIT),
        },
    )?;
    if !result.process.cleanup.complete
        || !result.process.success()
        || result.process.failure.is_some()
    {
        return Err("GitHub API request failed or timed out".into());
    }
    Ok(result
        .stdout
        .ok_or("GitHub response exceeds size limit")?
        .as_bytes()
        .to_vec())
}

fn api(path: &str, fields: &[String]) -> DynResult<Vec<u8>> {
    let mut args = vec![
        "api".into(),
        "--allow-escape-sequences".into(),
        "--method".into(),
        "GET".into(),
        path.into(),
    ];
    for field in fields {
        args.extend(["-f".into(), field.clone()]);
    }
    gh(&args)
}

fn api_json(path: &str, fields: &[String]) -> DynResult<Json> {
    Ok(serde_json::from_slice(&api(path, fields)?)?)
}

type Run = (u64, u64, String, String);

fn admit_runs(rows: &[Json], title: &str, found: &mut Option<Run>) -> DynResult<()> {
    for row in rows {
        if row["display_title"].as_str() != Some(title) {
            continue;
        }
        if found.is_some() {
            return Err("ambiguous packaging dispatch runs".into());
        }
        let id = row["id"]
            .as_u64()
            .filter(|id| *id > 0)
            .ok_or("invalid run ID")?;
        let attempt = row["run_attempt"]
            .as_u64()
            .filter(|id| *id > 0)
            .ok_or("invalid run attempt")?;
        let status = row["status"]
            .as_str()
            .ok_or("missing run status")?
            .to_string();
        let conclusion = row["conclusion"].as_str().unwrap_or("").to_string();
        *found = Some((id, attempt, status, conclusion));
    }
    Ok(())
}

fn runs(expected: &Expected) -> DynResult<Option<Run>> {
    let mut found = None;
    let title = format!("Packaging · {}", expected.correlation);
    for page in 1..=MAX_PAGES {
        let json = api_json(
            &format!("repos/{REPO}/actions/workflows/images-release.yml/runs"),
            &[
                "event=repository_dispatch".into(),
                format!("created=>={}", expected.since),
                "per_page=100".into(),
                format!("page={page}"),
            ],
        )?;
        let rows = json["workflow_runs"]
            .as_array()
            .ok_or("invalid workflow run list")?;
        admit_runs(rows, &title, &mut found)?;
        if rows.len() < 100 {
            return Ok(found);
        }
    }
    Err("packaging run discovery exceeds page limit".into())
}

fn admit_artifacts(rows: &[Json], run: u64, ids: &mut Vec<u64>) -> DynResult<()> {
    for row in rows {
        if row["name"].as_str() != Some("packaging-readiness") {
            continue;
        }
        if row["workflow_run"]["id"].as_u64() != Some(run) {
            return Err("packaging artifact run mismatch".into());
        }
        if row["expired"].as_bool() != Some(false) {
            return Err("packaging readiness artifact expired".into());
        }
        if row["size_in_bytes"]
            .as_u64()
            .is_none_or(|size| size == 0 || size > LIMIT as u64)
        {
            return Err("invalid packaging readiness artifact size".into());
        }
        ids.push(
            row["id"]
                .as_u64()
                .filter(|id| *id > 0)
                .ok_or("invalid artifact ID")?,
        );
    }
    Ok(())
}

fn artifacts(run: u64) -> DynResult<Vec<u64>> {
    let mut ids = Vec::new();
    for page in 1..=MAX_PAGES {
        let json = api_json(
            &format!("repos/{REPO}/actions/runs/{run}/artifacts"),
            &["per_page=100".into(), format!("page={page}")],
        )?;
        let rows = json["artifacts"]
            .as_array()
            .ok_or("invalid artifact list")?;
        admit_artifacts(rows, run, &mut ids)?;
        if rows.len() < 100 {
            return Ok(ids);
        }
    }
    Err("packaging artifact discovery exceeds page limit".into())
}

fn completed(status: &str, conclusion: &str) -> DynResult<bool> {
    if status == "completed" {
        if conclusion != "success" {
            return Err(format!("packaging run concluded {conclusion}").into());
        }
        return Ok(true);
    }
    if ["queued", "in_progress", "waiting", "pending", "requested"].contains(&status) {
        return Ok(false);
    }
    Err(format!("unknown packaging run status: {status}").into())
}

fn admit_receipt(
    receipt: &Json,
    expected: &Expected,
    run: u64,
    attempt: u64,
    matched: &mut bool,
) -> DynResult<()> {
    if receipt["packaging_run_id"].as_u64() != Some(run) {
        return Err("packaging receipt run mismatch".into());
    }
    let candidate = receipt["packaging_run_attempt"]
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or("invalid receipt attempt")?;
    if candidate > attempt {
        return Err("packaging receipt is from a future attempt".into());
    }
    if candidate < attempt {
        return Ok(());
    }
    if *matched {
        return Err("ambiguous current-attempt packaging receipts".into());
    }
    validate_receipt(receipt, expected, run, attempt)?;
    *matched = true;
    Ok(())
}

fn verify_receipts(
    ids: &[u64],
    expected: &Expected,
    run: u64,
    attempt: u64,
    mut download: impl FnMut(u64) -> DynResult<Vec<u8>>,
) -> DynResult<()> {
    if ids.is_empty() || ids.len() > MAX_RECEIPTS {
        return Err("invalid number of packaging readiness artifacts".into());
    }
    let mut matched = false;
    for id in ids {
        let bytes = crate::artifact::packaging_receipt_zip(download(*id)?)?;
        let receipt: Json = serde_json::from_slice(&bytes)?;
        admit_receipt(&receipt, expected, run, attempt, &mut matched)?;
    }
    if matched {
        Ok(())
    } else {
        Err("missing current-attempt packaging receipt".into())
    }
}

fn verify_final_run(value: &Json, run: u64, attempt: u64, correlation: &str) -> DynResult<()> {
    let title = format!("Packaging · {correlation}");
    if value["id"].as_u64() != Some(run)
        || value["run_attempt"].as_u64() != Some(attempt)
        || value["display_title"].as_str() != Some(title.as_str())
        || value["event"].as_str() != Some("repository_dispatch")
        || value["status"].as_str() != Some("completed")
        || value["conclusion"].as_str() != Some("success")
    {
        return Err("packaging run changed during receipt verification".into());
    }
    Ok(())
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let expected = parse(args)?;
    let deadline = Instant::now() + Duration::from_secs(7200);
    loop {
        if let Some((id, attempt, status, conclusion)) = runs(&expected)?
            && completed(&status, &conclusion)?
        {
            let ids = artifacts(id)?;
            verify_receipts(&ids, &expected, id, attempt, |artifact| {
                api(
                    &format!("repos/{REPO}/actions/artifacts/{artifact}/zip"),
                    &[],
                )
            })?;
            let final_run = api_json(&format!("repos/{REPO}/actions/runs/{id}"), &[])?;
            verify_final_run(&final_run, id, attempt, &expected.correlation)?;
            println!(
                "packaging run {id} attempt {attempt} succeeded with matching readiness receipt"
            );
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err("timed out waiting for packaging readiness".into());
        }
        std::thread::sleep(Duration::from_secs(15));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn expected() -> Expected {
        Expected {
            tag: "v1.2.3".into(),
            sha: "a".repeat(40),
            manifest: "b".repeat(64),
            correlation: "mesh-123-2".into(),
            since: "2026-11-13T00:00:00Z".into(),
        }
    }

    fn receipt() -> Json {
        let expected = expected();
        json!({
            "schema":"mesh-packaging-readiness-v1", "status":"success", "correlation_id":expected.correlation,
            "packaging_run_id":1234, "packaging_run_attempt":2,
            "upstream":{"repository":"Mesh-LLM/mesh-llm","ref":expected.tag,"sha":expected.sha,"manifest_sha256":expected.manifest},
            "requested":{"publish_images":"true","publish_release_assets":"true","publish_npm":"true"},
            "results":{"plan":"success","upstream":"success","package_images":"success","homebrew":"success","release_assembly":"success","npm_addons":"success","npm_package":"success","npm_preflight":"success","publish_npm":"success","image_index":"success","promote_images":"success","publish_release_assets":"success"}
        })
    }

    #[test]
    fn accepts_only_complete_matching_receipts() {
        let good = receipt();
        assert!(validate_receipt(&good, &expected(), 1234, 2).is_ok());
        assert!(validate_receipt(&good, &expected(), 1234, 1).is_err());
        assert!(validate_receipt(&good, &expected(), 9, 2).is_err());
        for (section, key, bad) in [
            ("upstream", "sha", "c"),
            ("upstream", "manifest_sha256", "d"),
            ("upstream", "ref", "v1.2.4"),
            ("requested", "publish_npm", "false"),
            ("results", "promote_images", "skipped"),
            ("results", "npm_preflight", "failure"),
        ] {
            let mut altered = good.clone();
            altered[section][key] = json!(bad);
            assert!(
                validate_receipt(&altered, &expected(), 1234, 2).is_err(),
                "{section}.{key}"
            );
        }
        let mut altered = good;
        altered["correlation_id"] = json!("mesh-123-1");
        assert!(validate_receipt(&altered, &expected(), 1234, 2).is_err());
    }

    #[test]
    fn rejects_ambiguous_runs_and_bad_artifacts() {
        let row = json!({"display_title":"Packaging · mesh-123-2","id":4,"run_attempt":1,"status":"completed","conclusion":"success"});
        let mut found = None;
        let legacy = json!({"display_title":"Packaging · refs/heads/main","id":3,"run_attempt":1,"status":"completed","conclusion":"success"});
        admit_runs(&[legacy], "Packaging · mesh-123-2", &mut found).unwrap();
        assert!(found.is_none());
        admit_runs(
            std::slice::from_ref(&row),
            "Packaging · mesh-123-2",
            &mut found,
        )
        .unwrap();
        assert!(admit_runs(&[row], "Packaging · mesh-123-2", &mut found).is_err());
        let artifact = json!({"name":"packaging-readiness","id":5,"expired":false,"size_in_bytes":500,"workflow_run":{"id":4}});
        let mut ids = Vec::new();
        admit_artifacts(std::slice::from_ref(&artifact), 4, &mut ids).unwrap();
        assert_eq!(ids, [5]);
        let mut expired = artifact.clone();
        expired["expired"] = json!(true);
        assert!(admit_artifacts(&[expired], 4, &mut Vec::new()).is_err());
        assert!(admit_artifacts(std::slice::from_ref(&artifact), 3, &mut Vec::new()).is_err());
        let mut oversized = artifact;
        oversized["size_in_bytes"] = json!(LIMIT + 1);
        assert!(admit_artifacts(&[oversized], 4, &mut Vec::new()).is_err());
        assert!(valid_utc_timestamp("2026-11-13T00:00:00Z"));
        assert!(!valid_utc_timestamp("2026-11-13T00:00:00+"));
        assert!(completed("completed", "failure").is_err());
        assert!(completed("completed", "cancelled").is_err());
        assert!(!completed("in_progress", "").unwrap());
        assert!(completed("completed", "success").unwrap());
    }

    #[test]
    fn selects_latest_receipt_and_rechecks_the_run_attempt() {
        let mut old = receipt();
        old["packaging_run_attempt"] = json!(1);
        old["status"] = json!("failure");
        let mut matched = false;
        admit_receipt(&old, &expected(), 1234, 2, &mut matched).unwrap();
        assert!(!matched);
        admit_receipt(&receipt(), &expected(), 1234, 2, &mut matched).unwrap();
        assert!(matched);
        assert!(admit_receipt(&receipt(), &expected(), 1234, 2, &mut matched).is_err());
        assert!(verify_receipts(&[], &expected(), 1234, 2, |_| unreachable!()).is_err());
        let mut run = json!({"id":1234,"run_attempt":2,"event":"repository_dispatch","display_title":"Packaging · mesh-123-2","status":"completed","conclusion":"success"});
        assert!(verify_final_run(&run, 1234, 2, "mesh-123-2").is_ok());
        run["run_attempt"] = json!(3);
        assert!(verify_final_run(&run, 1234, 2, "mesh-123-2").is_err());
        run["run_attempt"] = json!(2);
        run["status"] = json!("in_progress");
        assert!(verify_final_run(&run, 1234, 2, "mesh-123-2").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn supervised_transport_keeps_binary_zip_bytes() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let zip = include_bytes!("../../tests/fixtures/packaging-readiness.zip");
        let archive = dir.path().join("receipt.zip");
        std::fs::write(&archive, zip).unwrap();
        let command = dir.path().join("gh");
        std::fs::write(&command, "#!/usr/bin/env sh\ncat \"$FIXTURE\"\n").unwrap();
        std::fs::set_permissions(&command, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut env = BTreeMap::new();
        env.insert(
            "PATH".into(),
            Value::Public(std::env::var_os("PATH").unwrap()),
        );
        env.insert("FIXTURE".into(), Value::Public(archive.into_os_string()));
        let bytes = gh_from(&[], command, dir.path().canonicalize().unwrap(), env).unwrap();
        assert_eq!(bytes, zip);
        assert!(crate::artifact::packaging_receipt_zip(bytes).is_ok());
    }
}

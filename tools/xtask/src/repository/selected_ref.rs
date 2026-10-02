//! Freeze an explicitly trusted same-origin branch-reachable MeshLLM revision.
mod git;
use crate::{
    automation::command_interrupt::Interrupt, command::DynResult, repository::check_args::Grammar,
};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};
const PINS: [&str; 2] = [
    "third_party/llama.cpp/upstream.txt",
    "skippy/third_party/llama.cpp/upstream.txt",
];
const GRAMMAR: Grammar = Grammar {
    usage: "repository selected-ref --ref REF --expected-origin URL --event workflow_dispatch [--repository PATH] [--upstream SHA] [--github-output PATH --summary PATH] [--timeout-secs SECONDS]",
    values: &[
        "--ref",
        "--expected-origin",
        "--event",
        "--repository",
        "--upstream",
        "--github-output",
        "--summary",
        "--timeout-secs",
    ],
    flags: &[],
};
#[derive(serde::Serialize)]
struct Selection {
    source: String,
    mesh_source: String,
    upstream: String,
    changed: &'static str,
    mode: &'static str,
    certify: &'static str,
}
fn sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(crate) fn run(
    args: &[String],
    default_root: impl FnOnce() -> DynResult<PathBuf>,
) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if !parsed.positionals.is_empty() {
        return Err("selected-ref accepts named inputs only".into());
    }
    let reference = parsed.last("--ref").ok_or("selected ref missing")?;
    let expected = parsed
        .last("--expected-origin")
        .ok_or("expected same-repository origin missing")?;
    if parsed.last("--event") != Some("workflow_dispatch") {
        return Err("selected source requires an explicit manual dispatch".into());
    }
    if !parsed.last("--upstream").unwrap_or("").is_empty() {
        return Err("selected ref cannot override its existing upstream pin".into());
    }
    if reference.is_empty()
        || reference.trim() != reference
        || expected.is_empty()
        || expected.trim() != expected
    {
        return Err("selected ref and expected origin must be nonempty canonical inputs".into());
    }
    let timeout = parsed
        .last("--timeout-secs")
        .unwrap_or("120")
        .parse::<u64>()?;
    if !(1..=120).contains(&timeout) {
        return Err("selected ref budget must be 1..120 seconds".into());
    }
    let root = match parsed.last("--repository") {
        Some(path) => std::fs::canonicalize(path)?,
        None => default_root()?,
    };
    let interrupt = Interrupt::install()?;
    let transaction = git::Transaction::new(
        &root,
        Duration::from_secs(timeout),
        interrupt.cancellation(),
    )?;
    let result = resolve(&transaction, reference, expected);
    interrupt.finish()?;
    let selected = result?;
    publish(
        &selected,
        parsed.last("--github-output"),
        parsed.last("--summary"),
    )?;
    println!("{}", serde_json::to_string(&selected)?);
    Ok(())
}
fn resolve(git: &git::Transaction<'_>, reference: &str, expected: &str) -> DynResult<Selection> {
    let branch = reference.strip_prefix("refs/heads/").unwrap_or(reference);
    if !sha(reference) {
        if reference.starts_with("refs/") && !reference.starts_with("refs/heads/") {
            return Err("only branches or full lowercase commit SHAs are admitted".into());
        }
        git.checked(&["check-ref-format", &format!("refs/heads/{branch}")])?;
    }
    let origin = git.checked(&["remote", "get-url", "origin"])?;
    if origin_identity(&origin)? != origin_identity(expected)? {
        return Err(
            "origin differs from the controller's expected same-repository authority".into(),
        );
    }
    let controller = git.checked(&["rev-parse", "--verify", "HEAD^{commit}"])?;
    if !sha(&controller) {
        return Err("controller commit is not a lowercase SHA".into());
    }
    let shallow = git.checked(&["rev-parse", "--is-shallow-repository"])?;
    let mut fetch = vec!["fetch", "--no-tags", "--prune"];
    match shallow.as_str() {
        "true" => fetch.push("--unshallow"),
        "false" => (),
        _ => return Err("invalid Git shallow status".into()),
    }
    fetch.extend(["origin", "+refs/heads/*:refs/canary-mesh/*"]);
    git.checked(&fetch)?;
    let target = if sha(reference) {
        reference.to_owned()
    } else {
        format!("refs/canary-mesh/{branch}")
    };
    let source = git.checked(&["rev-parse", "--verify", &format!("{target}^{{commit}}")])?;
    if !sha(&source) {
        return Err("selected commit is not a lowercase SHA".into());
    }
    let reachable = git.checked(&[
        "for-each-ref",
        "--format=%(refname)",
        &format!("--contains={source}"),
        "refs/canary-mesh/",
    ])?;
    if reachable.is_empty() {
        return Err("selected commit is not reachable from a same-repository branch".into());
    }
    let pin = selected_pin(git, &source)?;
    Ok(Selection {
        source: controller,
        mesh_source: source,
        upstream: pin,
        changed: "false",
        mode: "pinned-build",
        certify: "true",
    })
}
fn selected_pin(git: &git::Transaction<'_>, source: &str) -> DynResult<String> {
    let tree = git.checked(&["ls-tree", "-z", source, "--", PINS[0], PINS[1]])?;
    let entries: Vec<_> = tree.split('\0').filter(|entry| !entry.is_empty()).collect();
    let [entry] = entries.as_slice() else {
        return Err("selected revision must have exactly one supported llama.cpp pin".into());
    };
    let (metadata, path) = entry
        .split_once('\t')
        .ok_or("invalid selected pin tree entry")?;
    let fields: Vec<_> = metadata.split_whitespace().collect();
    if !PINS.contains(&path)
        || !matches!(fields.as_slice(), ["100644" | "100755", "blob", object] if sha(object))
    {
        return Err("selected pin must be a regular Git blob".into());
    }
    let pin = git.checked(&["show", &format!("{source}:{path}")])?;
    if !sha(&pin) {
        return Err("selected revision has an invalid llama.cpp pin".into());
    }
    Ok(pin)
}
fn append(path: &str, bytes: &[u8]) -> DynResult<()> {
    let path = Path::new(path);
    if std::fs::symlink_metadata(path).is_ok_and(|meta| !meta.file_type().is_file()) {
        return Err("GitHub evidence destination must be a regular file".into());
    }
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut output = options.open(path)?;
    if !output.metadata()?.is_file() {
        return Err("GitHub evidence destination must be regular".into());
    }
    output.write_all(bytes)?;
    Ok(())
}
fn publish(selected: &Selection, output: Option<&str>, summary: Option<&str>) -> DynResult<()> {
    match (output, summary) {
        (Some(output), Some(summary)) => {
            let values = format!(
                "source={}\nmesh_source={}\nupstream={}\nchanged=false\nmode=pinned-build\ncertify=true\n",
                selected.source, selected.mesh_source, selected.upstream
            );
            let receipt = format!(
                "Certify-only MeshLLM revision: `{}`\n\nExisting llama.cpp pin: `{}`\n",
                selected.mesh_source, selected.upstream
            );
            append(output, values.as_bytes())?;
            append(summary, receipt.as_bytes())?;
        }
        (None, None) => (),
        _ => return Err("GitHub output and summary must be provided together".into()),
    }
    Ok(())
}

fn origin_identity(value: &str) -> DynResult<String> {
    if let Some(path) = value.strip_prefix("file://") {
        if !path.starts_with('/') || path.contains(['\n', '\r', '\0']) {
            return Err("invalid local origin authority".into());
        }
        return Ok(format!("file://{path}"));
    }
    let remote = value
        .strip_prefix("https://")
        .or_else(|| value.strip_prefix("ssh://git@"));
    let (host, path) = match remote {
        Some(remote) => remote
            .split_once('/')
            .ok_or("origin repository path missing")?,
        None => value
            .strip_prefix("git@")
            .and_then(|remote| remote.split_once(':'))
            .ok_or("unsupported origin authority form")?,
    };
    let path = path.strip_suffix(".git").unwrap_or(path);
    if host.is_empty()
        || host.contains(['@', '?', '#', '\\'])
        || path.is_empty()
        || path.contains(['?', '#', '\\', '\n', '\r', '\0'])
        || host.chars().any(char::is_whitespace)
        || path.chars().any(char::is_whitespace)
    {
        return Err("invalid origin repository authority".into());
    }
    Ok(format!(
        "{}/{}",
        host.to_ascii_lowercase(),
        path.to_ascii_lowercase()
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn origin_equivalent_transports_keep_repository_authority() {
        let expected = origin_identity("https://github.com/Mesh-LLM/mesh-llm").unwrap();
        for origin in [
            "https://github.com/Mesh-LLM/mesh-llm.git",
            "git@github.com:Mesh-LLM/mesh-llm.git",
            "ssh://git@github.com/Mesh-LLM/mesh-llm.git",
        ] {
            assert_eq!(origin_identity(origin).unwrap(), expected);
        }
        for origin in [
            "https://github.com/fork/mesh-llm.git",
            "git@foreign.invalid:Mesh-LLM/mesh-llm.git",
            "https://github.com:8443/Mesh-LLM/mesh-llm.git",
            "ssh://git@github.com:2222/Mesh-LLM/mesh-llm.git",
            "https://github.com/Mesh-LLM/other.git",
            "https://github.com/Mesh-LLM/../Mesh-LLM/mesh-llm.git",
        ] {
            assert_ne!(origin_identity(origin).unwrap(), expected);
        }
        for origin in [
            "https://token@github.com/Mesh-LLM/mesh-llm",
            "https://github.com/Mesh-LLM/mesh-llm?ref=main",
            "ssh://attacker@github.com/Mesh-LLM/mesh-llm.git",
        ] {
            assert!(origin_identity(origin).is_err());
        }
        assert_ne!(
            origin_identity("file:///tmp/Case").unwrap(),
            origin_identity("file:///tmp/case").unwrap()
        );
    }
}

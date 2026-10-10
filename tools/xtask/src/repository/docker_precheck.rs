//! Source-only Docker policy. This command never builds an image or executes source.
use super::{check_args::Grammar, check_report::CheckReport};
use crate::command::DynResult;
use std::{
    fs,
    io::Read as _,
    path::{Component, Path, PathBuf},
};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool repository docker-precheck --dockerfile PATH --shared-core true|false --fly-ui-builder true|false --entrypoint-modes true|false --workflow-no-qemu true|false",
    values: &[
        "--dockerfile",
        "--shared-core",
        "--fly-ui-builder",
        "--entrypoint-modes",
        "--workflow-no-qemu",
    ],
    flags: &["--help"],
};

pub(super) fn run(args: &[String], root: impl FnOnce() -> DynResult<PathBuf>) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(value) => value,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let boolean = |name| -> DynResult<bool> {
        match parsed.all(name).as_slice() {
            ["true"] => Ok(true),
            ["false"] => Ok(false),
            _ => Err(format!("{name} requires exactly one true or false").into()),
        }
    };
    let policy = Policy {
        shared: boolean("--shared-core")?,
        fly: boolean("--fly-ui-builder")?,
        entrypoint: boolean("--entrypoint-modes")?,
        no_qemu: boolean("--workflow-no-qemu")?,
    };
    let path = match parsed.all("--dockerfile").as_slice() {
        [path] => (*path).to_owned(),
        _ => return Err("exactly one --dockerfile required".into()),
    };
    let root = root()?;
    check(
        &read(&root, &path)?,
        policy,
        policy
            .entrypoint
            .then(|| read(&root, "mesh/deploy/docker/entrypoint.sh"))
            .transpose()?
            .as_deref(),
        policy
            .no_qemu
            .then(|| read(&root, ".github/workflows/docker.yml"))
            .transpose()?
            .as_deref(),
    )?;
    CheckReport::success("Docker source precheck passed\n".into()).emit()
}

fn read(root: &Path, relative: &str) -> DynResult<String> {
    let path = Path::new(relative);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err("Docker source must be a contained relative path".into());
    }
    let mut location = root.to_path_buf();
    for part in path.components() {
        location.push(part.as_os_str());
        if fs::symlink_metadata(&location)?.file_type().is_symlink() {
            return Err("Docker source symlink refused".into());
        }
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(location)?;
    if !file.metadata()?.is_file() {
        return Err("Docker source must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(1048577).read_to_end(&mut bytes)?;
    if bytes.len() > 1048576 {
        return Err("Docker source exceeds 1 MiB".into());
    }
    Ok(String::from_utf8(bytes)?)
}

#[derive(Clone, Copy)]
struct Policy {
    shared: bool,
    fly: bool,
    entrypoint: bool,
    no_qemu: bool,
}

fn require(condition: bool, reason: &str) -> DynResult<()> {
    if condition {
        Ok(())
    } else {
        Err(reason.to_owned().into())
    }
}

fn check(
    source: &str,
    policy: Policy,
    entrypoint: Option<&str>,
    workflow: Option<&str>,
) -> DynResult<()> {
    let lines: Vec<_> = source
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .collect();
    if policy.shared || policy.fly {
        let ui = lines
            .iter()
            .position(|line| line.starts_with("COPY --from=ui-builder "));
        let cargo = lines.iter().position(|line| line.contains("cargo build"));
        require(
            matches!((ui,cargo), (Some(ui),Some(cargo)) if ui < cargo),
            "UI dist must be copied before cargo build",
        )?;
    }
    if policy.shared {
        shared(&lines)?;
    }
    if policy.fly {
        require(
            lines.iter().any(|line| {
                line.split_whitespace()
                    .collect::<Vec<_>>()
                    .ends_with(&["AS", "ui-builder"])
            }),
            "Fly UI builder stage missing",
        )?;
    }
    if policy.entrypoint {
        entrypoint_modes(entrypoint.ok_or("entrypoint source missing")?)?;
    }
    if policy.no_qemu {
        require(
            !has_qemu_token(workflow.ok_or("Docker workflow missing")?),
            "QEMU tooling reference found in Docker workflow",
        )?;
    }
    Ok(())
}

fn has_qemu_token(source: &str) -> bool {
    source
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .any(|token| token.eq_ignore_ascii_case("qemu"))
}

fn shared(lines: &[&str]) -> DynResult<()> {
    let copies: Vec<Vec<_>> = lines
        .iter()
        .map(|line| line.split_whitespace().collect())
        .filter(|line: &Vec<&str>| line.first() == Some(&"COPY"))
        .collect();
    require(
        !lines
            .iter()
            .map(|line| context_blanket_copy(line))
            .collect::<DynResult<Vec<_>>>()?
            .contains(&true),
        "blanket COPY . . refused",
    )?;
    require(
        copies.iter().any(|row| {
            row.as_slice() == ["COPY", "Cargo.toml", "Cargo.lock", "./"]
                || row.as_slice() == ["COPY", "Cargo.toml", "Cargo.lock", "."]
        }),
        "workspace manifest copy missing",
    )?;
    for path in [
        "mesh/crates/",
        "skippy/crates/",
        "tools/xtask/",
        "scripts/",
        "mesh/scripts/",
        "skippy/scripts/",
    ] {
        require(
            copies
                .iter()
                .any(|row| row.as_slice() == ["COPY", path, path]),
            &format!("workspace tree copy missing: {path}"),
        )?;
    }
    for required in [
        "scripts/prepare-llama.sh pinned",
        "scripts/build-llama.sh",
        "skippy/llama_cpp/patches",
        "ca-certificates",
        "libgomp1",
        "libdbus-1-3",
    ] {
        require(
            lines.iter().any(|line| line.contains(required)),
            &format!("required Docker source contract missing: {required}"),
        )?;
    }
    Ok(())
}

fn entrypoint_modes(source: &str) -> DynResult<()> {
    let modes = source
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| line.split_once(')').map(|(arm, _)| arm))
        .flat_map(|arm| arm.trim_start_matches('(').split('|'))
        .map(|pattern| pattern.trim().trim_matches(['\'', '"']))
        .collect::<Vec<_>>();
    require(
        ["console", "worker", "*"]
            .iter()
            .all(|mode| modes.contains(mode)),
        "entrypoint console/worker/default modes missing",
    )?;
    require(
        !modes.contains(&"api"),
        "api must not be a separate entrypoint mode",
    )
}

fn context_blanket_copy(line: &str) -> DynResult<bool> {
    let Some((instruction, rest)) = line.split_once(char::is_whitespace) else {
        return Ok(false);
    };
    if instruction != "COPY" {
        return Ok(false);
    }
    let mut rest = rest.trim_start();
    while rest.starts_with("--") {
        let (flag, tail) = rest
            .split_once(char::is_whitespace)
            .ok_or("COPY flag lacks operands")?;
        if flag == "--from" {
            require(!tail.trim().is_empty(), "COPY stage source missing")?;
            return Ok(false);
        }
        if let Some(stage) = flag.strip_prefix("--from=") {
            require(!stage.is_empty(), "COPY stage source missing")?;
            return Ok(false);
        }
        rest = tail.trim_start();
    }
    let operands: Vec<String> = if rest.starts_with('[') {
        serde_json::from_str(rest)?
    } else {
        rest.split_whitespace()
            .take_while(|token| !token.starts_with('#'))
            .map(str::to_owned)
            .collect()
    };
    Ok(operands.len() >= 2
        && operands[..operands.len() - 1]
            .iter()
            .any(|source| source == "." || source == "./")
        && operands
            .last()
            .is_some_and(|destination| destination == "." || destination == "./"))
}

#[cfg(test)]
mod tests;

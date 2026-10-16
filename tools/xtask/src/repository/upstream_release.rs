//! Select the newest published llama.cpp release, including prereleases.
mod discovery;
mod github;
use crate::{automation::command_interrupt::Interrupt, command::DynResult};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs::OpenOptions, io::Write};

const REPOSITORY: &str = "repos/ggml-org/llama.cpp";
const GRAMMAR: super::check_args::Grammar = super::check_args::Grammar {
    usage: "repository llama-upstream-release [--upstream SHA] [--github-output PATH --summary PATH]",
    values: &["--upstream", "--github-output", "--summary"],
    flags: &[],
};

#[derive(Deserialize)]
struct Release {
    id: u64,
    tag_name: String,
    draft: bool,
    prerelease: bool,
    created_at: Option<String>,
    published_at: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct Object {
    sha: String,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct Reference {
    #[serde(rename = "ref")]
    name: String,
    object: Object,
}

#[derive(Deserialize)]
struct Tag {
    sha: String,
    object: Object,
}

#[derive(Debug, Serialize)]
struct Selection {
    sha: String,
    release_tag: Option<String>,
    release_id: Option<u64>,
    prerelease: Option<bool>,
    published_at: Option<String>,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if !parsed.positionals.is_empty() {
        return Err("upstream release selection accepts named inputs only".into());
    }
    let destinations = match (parsed.last("--github-output"), parsed.last("--summary")) {
        (Some(output), Some(summary)) => Some((output, summary)),
        (None, None) => None,
        _ => return Err("GitHub output and summary must be provided together".into()),
    };
    let interrupt = Interrupt::install()?;
    let mut client = github::Client::new(interrupt.cancellation());
    let result = select(parsed.last("--upstream").unwrap_or(""), |endpoint| {
        client.read(endpoint)
    });
    interrupt.finish()?;
    let selection = result?;
    if let Some((output, summary)) = destinations {
        append(summary, &selection.summary())?;
        append(output, &format!("sha={}\n", selection.sha))?;
    }
    println!("{}", serde_json::to_string(&selection)?);
    Ok(())
}

fn select(
    requested: &str,
    mut read: impl FnMut(&github::Request) -> DynResult<serde_json::Value>,
) -> DynResult<Selection> {
    if !matches!(requested, "" | "latest") {
        require_sha(requested)?;
        return Ok(Selection {
            sha: requested.into(),
            release_tag: None,
            release_id: None,
            prerelease: None,
            published_at: None,
        });
    }
    let release = discovery::newest(&mut read)?;
    if release.tag_name.is_empty()
        || release.tag_name.len() > 256
        || release.tag_name.contains("..")
        || !release
            .tag_name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.+".contains(&b))
    {
        return Err("upstream release tag is invalid".into());
    }
    // target_commitish is not authoritative: it can be a mutable branch name.
    let endpoint = github::Request::Rest(format!("{REPOSITORY}/git/ref/tags/{}", release.tag_name));
    let reference: Reference = serde_json::from_value(read(&endpoint)?)?;
    if reference.name != format!("refs/tags/{}", release.tag_name) {
        return Err("upstream release tag reference mismatch".into());
    }
    let sha = peel(reference.object.clone(), &mut read)?;
    let current: Reference = serde_json::from_value(read(&endpoint)?)?;
    if current.name != reference.name || current.object != reference.object {
        return Err("upstream release tag moved during resolution".into());
    }
    Ok(Selection {
        sha,
        release_tag: Some(release.tag_name),
        release_id: Some(release.id),
        prerelease: Some(release.prerelease),
        published_at: release.published_at,
    })
}

fn peel(
    mut object: Object,
    read: &mut impl FnMut(&github::Request) -> DynResult<serde_json::Value>,
) -> DynResult<String> {
    let mut seen = BTreeSet::new();
    for _ in 0..8 {
        require_sha(&object.sha)?;
        if !seen.insert(object.sha.clone()) {
            return Err("upstream release tag cycle".into());
        }
        match object.kind.as_str() {
            "commit" => return Ok(object.sha),
            "tag" => {
                let tag: Tag = serde_json::from_value(read(&github::Request::Rest(format!(
                    "{REPOSITORY}/git/tags/{}",
                    object.sha
                )))?)?;
                if tag.sha != object.sha {
                    return Err("upstream tag object identity mismatch".into());
                }
                object = tag.object;
            }
            _ => return Err("upstream release tag must resolve to a commit".into()),
        }
    }
    Err("upstream release tag nesting exceeds resolution limit".into())
}

fn require_sha(value: &str) -> DynResult<()> {
    if value.len() == 40
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err("upstream commit must be a full lowercase 40-character SHA".into())
    }
}

impl Selection {
    fn summary(&self) -> String {
        let source = match &self.release_tag {
            Some(tag) => format!(
                "Newest published llama.cpp {}: [{tag}](https://github.com/ggml-org/llama.cpp/releases/tag/{tag})\n\nPublished: `{}`",
                if self.prerelease == Some(true) {
                    "prerelease"
                } else {
                    "release"
                },
                self.published_at.as_deref().unwrap_or_default(),
            ),
            None => "Explicit upstream commit override (release selection bypassed)".into(),
        };
        format!(
            "{source}\n\nFrozen upstream commit: `{}`\n\nMeshLLM family certification is still required before upgrade publication.\n",
            self.sha
        )
    }
}

fn append(path: &str, value: &str) -> DynResult<()> {
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut output = options.open(path)?;
    if !output.metadata()?.is_file() {
        return Err("GitHub evidence destination must be a regular file".into());
    }
    output.write_all(value.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests;

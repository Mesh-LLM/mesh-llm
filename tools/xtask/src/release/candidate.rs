//! `release candidate`: decide whether a published candidate may be promoted.
//! The workflow reads GitHub and passes the answers in as files; this command
//! owns the rules and prints the release id that Promote edits.

use crate::repository::check_report::CheckReport;
use serde::Deserialize;
use std::collections::BTreeSet;

const USAGE: &str = "cargo xtool release candidate --tag <vX.Y.Z> --release <release.json> --assets <assets.json> --manifest <native-runtimes.json>";

#[derive(Deserialize)]
struct Release {
    id: u64,
    tag_name: String,
    draft: bool,
    prerelease: bool,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    state: String,
}

#[derive(Deserialize)]
struct Manifest {
    artifacts: Vec<Artifact>,
}

#[derive(Deserialize)]
struct Artifact {
    url: String,
}

/// Applies the admission rules to one release, its assets (one JSON object
/// per asset, as `gh api --paginate --jq '.[]'` prints them) and its native
/// runtime manifest.
fn admit(tag: &str, release: &str, assets: &str, manifest: &str) -> Result<u64, String> {
    let release: Release = serde_json::from_str(release)
        .map_err(|error| format!("Unable to resolve the release for {tag}: {error}"))?;
    if release.tag_name != tag {
        return Err(format!(
            "Unable to resolve the release for {tag}: GitHub returned {}",
            release.tag_name
        ));
    }
    if release.draft {
        return Err(format!(
            "{tag} is still a draft: its upload did not finish."
        ));
    }
    if !release.prerelease {
        return Err(format!(
            "{tag} is already promoted. Re-run the failed jobs of its Promote run instead of promoting it again."
        ));
    }
    let assets = serde_json::Deserializer::from_str(assets)
        .into_iter::<Asset>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Unable to read the release files of {tag}: {error}"))?;
    if assets.is_empty() {
        return Err(format!("{tag} has no release files."));
    }
    let incomplete: Vec<&str> = assets
        .iter()
        .filter(|asset| asset.state != "uploaded")
        .map(|asset| asset.name.as_str())
        .collect();
    if !incomplete.is_empty() {
        return Err(format!(
            "{tag} has incomplete uploads:\n{}",
            incomplete.join("\n")
        ));
    }
    let uploaded: BTreeSet<&str> = assets.iter().map(|asset| asset.name.as_str()).collect();
    let manifest: Manifest = serde_json::from_str(manifest)
        .map_err(|error| format!("Unable to read the runtime manifest of {tag}: {error}"))?;
    if manifest.artifacts.is_empty() {
        return Err(format!("{tag} lists no native runtimes."));
    }
    let prefix = format!("/releases/download/{tag}/");
    let missing: Vec<&str> = manifest
        .artifacts
        .iter()
        .map(|artifact| artifact.url.as_str())
        .filter(|url| {
            url.rsplit_once(&prefix)
                .is_none_or(|(_, name)| !uploaded.contains(name))
        })
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{tag} lists native runtimes that are not files of this release:\n{}",
            missing.join("\n")
        ));
    }
    Ok(release.id)
}

pub(crate) fn run(args: &[String]) -> CheckReport {
    let [
        tag_flag,
        tag,
        release_flag,
        release,
        assets_flag,
        assets,
        manifest_flag,
        manifest,
    ] = args
    else {
        return CheckReport::usage(USAGE, "expected four options");
    };
    if (tag_flag.as_str(), release_flag.as_str()) != ("--tag", "--release")
        || (assets_flag.as_str(), manifest_flag.as_str()) != ("--assets", "--manifest")
    {
        return CheckReport::usage(USAGE, "options must be given in the documented order");
    }
    let read = |path: &String| {
        std::fs::read_to_string(path).map_err(|error| format!("Unable to read {path}: {error}"))
    };
    let admitted = read(release).and_then(|release| {
        let (assets, manifest) = (read(assets)?, read(manifest)?);
        admit(tag, &release, &assets, &manifest)
    });
    match admitted {
        Ok(id) => CheckReport::success(format!("{id}\n")),
        Err(message) => CheckReport::failure(String::new(), format!("{message}\n")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAG: &str = "v0.81.0";
    const RELEASE: &str = r#"{"id":42,"tag_name":"v0.81.0","draft":false,"prerelease":true}"#;
    const ASSETS: &str = concat!(
        r#"{"name":"native-runtimes.json","state":"uploaded"}"#,
        "\n",
        r#"{"name":"runtime-metal.tar.gz","state":"uploaded"}"#,
        "\n",
        r#"{"name":"runtime-cpu.tar.gz","state":"uploaded"}"#,
        "\n",
    );
    const MANIFEST: &str = r#"{"artifacts":[
        {"url":"https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.81.0/runtime-metal.tar.gz"},
        {"url":"https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.81.0/runtime-cpu.tar.gz"}]}"#;

    fn refusal(release: &str, assets: &str, manifest: &str) -> String {
        admit(TAG, release, assets, manifest).unwrap_err()
    }

    #[test]
    fn release_candidate_admits_a_complete_unpromoted_candidate() {
        assert_eq!(admit(TAG, RELEASE, ASSETS, MANIFEST), Ok(42));
    }

    #[test]
    fn release_candidate_refuses_drafts_promoted_releases_and_other_tags() {
        let draft = RELEASE.replace(r#""draft":false"#, r#""draft":true"#);
        assert_eq!(
            refusal(&draft, ASSETS, MANIFEST),
            "v0.81.0 is still a draft: its upload did not finish."
        );
        let promoted = RELEASE.replace(r#""prerelease":true"#, r#""prerelease":false"#);
        assert!(refusal(&promoted, ASSETS, MANIFEST).starts_with("v0.81.0 is already promoted."));
        let other = RELEASE.replace("v0.81.0", "v0.80.0");
        assert!(refusal(&other, ASSETS, MANIFEST).ends_with("GitHub returned v0.80.0"));
        assert!(refusal("{}", ASSETS, MANIFEST).starts_with("Unable to resolve the release"));
    }

    #[test]
    fn release_candidate_refuses_missing_and_unfinished_files() {
        assert_eq!(
            refusal(RELEASE, "", MANIFEST),
            "v0.81.0 has no release files."
        );
        let unfinished = ASSETS.replacen("uploaded", "starter", 1);
        assert_eq!(
            refusal(RELEASE, &unfinished, MANIFEST),
            "v0.81.0 has incomplete uploads:\nnative-runtimes.json"
        );
        assert!(refusal(RELEASE, "[", MANIFEST).starts_with("Unable to read the release files"));
    }

    #[test]
    fn release_candidate_refuses_a_manifest_runtime_that_is_not_a_release_file() {
        let removed = ASSETS.replace("runtime-cpu.tar.gz", "unrelated.txt");
        assert_eq!(
            refusal(RELEASE, &removed, MANIFEST),
            "v0.81.0 lists native runtimes that are not files of this release:\nhttps://github.com/Mesh-LLM/mesh-llm/releases/download/v0.81.0/runtime-cpu.tar.gz"
        );
        let other_tag = MANIFEST.replacen("v0.81.0/runtime-metal", "v0.80.0/runtime-metal", 1);
        assert!(refusal(RELEASE, ASSETS, &other_tag).ends_with("v0.80.0/runtime-metal.tar.gz"));
        assert_eq!(
            refusal(RELEASE, ASSETS, r#"{"artifacts":[]}"#),
            "v0.81.0 lists no native runtimes."
        );
    }

    #[test]
    fn release_candidate_command_reads_files_and_prints_the_release_id() {
        let directory = tempfile::tempdir().unwrap();
        let path = |name: &str, text: &str| {
            let path = directory.path().join(name);
            std::fs::write(&path, text).unwrap();
            path.to_str().unwrap().to_owned()
        };
        let mut args: Vec<String> = ["--tag", TAG, "--release"].map(str::to_owned).into();
        args.extend([path("release.json", RELEASE), "--assets".to_owned()]);
        args.extend([path("assets.json", ASSETS), "--manifest".to_owned()]);
        args.push(path("native-runtimes.json", MANIFEST));
        assert_eq!(run(&args), CheckReport::success("42\n".to_owned()));
        args[7] = directory.path().join("absent").to_str().unwrap().to_owned();
        assert_eq!(run(&args).code, 1);
        assert_eq!(run(&args[..6]).code, 2);
    }
}

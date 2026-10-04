use crate::command::DynResult;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    workspace_members: Vec<String>,
}
#[derive(Deserialize)]
struct Package {
    id: String,
    name: String,
    manifest_path: PathBuf,
    publish: Option<Vec<String>>,
    dependencies: Vec<Dependency>,
}
#[derive(Deserialize)]
struct Dependency {
    kind: Option<String>,
    path: Option<PathBuf>,
}

fn projection(input: &[u8], pairs: bool) -> DynResult<String> {
    let metadata: Metadata = serde_json::from_slice(input)?;
    let members: BTreeSet<_> = metadata.workspace_members.into_iter().collect();
    let packages: Vec<_> = metadata
        .packages
        .into_iter()
        .filter(|package| {
            members.contains(&package.id)
                && package
                    .publish
                    .as_ref()
                    .is_none_or(|registries| !registries.is_empty())
        })
        .collect();
    let directories: BTreeMap<_, _> = packages
        .iter()
        .filter_map(|package| {
            package
                .manifest_path
                .parent()
                .map(|path| (path.to_path_buf(), package.name.clone()))
        })
        .collect();
    let mut graph: BTreeMap<String, BTreeSet<String>> = packages
        .iter()
        .map(|package| {
            let dependencies = package
                .dependencies
                .iter()
                .filter(|dependency| dependency.kind.as_deref() != Some("dev"))
                .filter_map(|dependency| {
                    dependency
                        .path
                        .as_ref()
                        .and_then(|path| directories.get(path))
                })
                .filter(|name| **name != package.name)
                .cloned()
                .collect();
            (package.name.clone(), dependencies)
        })
        .collect();
    let output = if pairs {
        graph
            .iter()
            .flat_map(|(name, dependencies)| {
                dependencies
                    .iter()
                    .map(move |dependency| format!("{name} {dependency}\n"))
            })
            .collect()
    } else {
        let mut output = String::new();
        while !graph.is_empty() {
            let next = graph
                .iter()
                .find(|(_, dependencies)| dependencies.is_empty())
                .map(|(name, _)| name.clone())
                .ok_or("publishable workspace dependency cycle")?;
            graph.remove(&next);
            for dependencies in graph.values_mut() {
                dependencies.remove(&next);
            }
            output.push_str(&next);
            output.push('\n');
        }
        output
    };
    Ok(output)
}

fn selected_projection(input: &[u8], script: &str) -> DynResult<String> {
    let metadata: Metadata = serde_json::from_slice(input)?;
    let roster = super::publish_roster::parse(script)?;
    let members: BTreeSet<_> = metadata.workspace_members.iter().collect();
    if members.len() != metadata.workspace_members.len() {
        return Err("duplicate workspace member identity".into());
    }
    let mut identities = BTreeSet::new();
    let mut packages = BTreeMap::new();
    let mut directories = BTreeMap::new();
    for package in &metadata.packages {
        if !identities.insert(&package.id) {
            return Err("duplicate Cargo package identity".into());
        }
        if !members.contains(&package.id) {
            continue;
        }
        let directory = package
            .manifest_path
            .parent()
            .ok_or("manifest has no parent")?;
        if packages.insert(&package.name, package).is_some()
            || directories.insert(directory, &package.name).is_some()
        {
            return Err("ambiguous workspace package name or directory".into());
        }
    }
    if !members.iter().all(|member| identities.contains(*member)) {
        return Err("workspace member missing from Cargo metadata".into());
    }
    let positions: BTreeMap<_, _> = roster
        .iter()
        .enumerate()
        .map(|(i, name)| (name, i))
        .collect();
    for (position, name) in roster.iter().enumerate() {
        let package = packages
            .get(name)
            .ok_or_else(|| format!("selected crate `{name}` is not a workspace member"))?;
        if package
            .publish
            .as_ref()
            .is_some_and(|registries| !registries.iter().any(|registry| registry == "crates-io"))
        {
            return Err(format!("selected crate `{name}` cannot publish to crates.io").into());
        }
        for dependency in package
            .dependencies
            .iter()
            .filter(|dep| dep.kind.as_deref() != Some("dev"))
        {
            let Some(path) = &dependency.path else {
                continue;
            };
            let target = directories
                .get(path.as_path())
                .ok_or_else(|| format!("{name}: path dependency is outside the workspace"))?;
            let target_position = positions.get(*target).ok_or_else(|| {
                format!("{name}: dependency `{target}` is absent from selected roster")
            })?;
            if *target_position >= position {
                return Err(format!("{name}: dependency `{target}` must publish first").into());
            }
        }
    }
    Ok(roster.into_iter().map(|name| format!("{name}\n")).collect())
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    if let [flag, path] = args
        && flag == "--selected-script"
    {
        let script = std::fs::File::open(path)?;
        let mut contents = String::new();
        script.take(1_048_577).read_to_string(&mut contents)?;
        if contents.len() > 1_048_576 {
            return Err("publish script exceeds 1 MiB".into());
        }
        let input = read_metadata()?;
        let output = selected_projection(&input, &contents)?;
        return crate::repository::check_report::CheckReport::success(output).emit();
    }
    let pairs = match args {
        [] => false,
        [flag] if flag == "--dependency-pairs" => true,
        _ => {
            return Err(
                "usage: repository publish-order [--dependency-pairs | --selected-script PATH] < cargo-metadata.json".into(),
            );
        }
    };
    let input = read_metadata()?;
    let output = projection(&input, pairs)?;
    crate::repository::check_report::CheckReport::success(output).emit()
}

fn read_metadata() -> DynResult<Vec<u8>> {
    const LIMIT: u64 = 64 * 1024 * 1024;
    let mut input = Vec::new();
    std::io::stdin().take(LIMIT + 1).read_to_end(&mut input)?;
    if input.len() as u64 > LIMIT {
        return Err("Cargo metadata exceeds 64 MiB".into());
    }
    Ok(input)
}

#[cfg(test)]
mod tests {
    fn metadata(cycle: bool) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "workspace_members":["consumer-id","provider-id","private-id"],
            "packages":[
                {"id":"consumer-id","name":"consumer","manifest_path":"/repo/consumer/Cargo.toml","publish":null,"dependencies":[{"kind":null,"path":"/repo/provider","optional":true},{"kind":"dev","path":"/repo/private"}]},
                {"id":"provider-id","name":"provider","manifest_path":"/repo/provider/Cargo.toml","publish":null,"dependencies": if cycle { serde_json::json!([{"kind":"build","path":"/repo/consumer"}]) } else { serde_json::json!([]) }},
                {"id":"private-id","name":"private","manifest_path":"/repo/private/Cargo.toml","publish":[],"dependencies":[]}
            ]
        })).unwrap()
    }

    #[test]
    fn optional_dependencies_publish_before_consumers() {
        assert_eq!(
            super::projection(&metadata(false), false).unwrap(),
            "provider\nconsumer\n"
        );
        assert_eq!(
            super::projection(&metadata(false), true).unwrap(),
            "consumer provider\n"
        );
    }

    #[test]
    fn cycles_fail_instead_of_emitting_partial_order() {
        assert!(super::projection(&metadata(true), false).is_err());
    }

    #[test]
    fn selected_release_roster_preserves_order_and_rejects_invalid_dependencies() {
        let script = "publish_crates=(\n provider\n consumer\n)\n";
        assert_eq!(
            super::selected_projection(&metadata(false), script).unwrap(),
            "provider\nconsumer\n"
        );
        for invalid in [
            "publish_crates=(\n consumer\n provider\n)",
            "publish_crates=(\n consumer\n)",
            "publish_crates=(\n provider\n controller-only\n)",
            "publish_crates=(\n private\n)",
        ] {
            assert!(super::selected_projection(&metadata(false), invalid).is_err());
        }
        assert!(super::selected_projection(&metadata(true), script).is_err());
    }

    #[test]
    fn selected_release_rejects_ambiguous_identity_and_private_path_dependencies() {
        let script = "publish_crates=(\n provider\n consumer\n)\n";
        let original: serde_json::Value = serde_json::from_slice(&metadata(false)).unwrap();
        for mutation in 0..6 {
            let mut value = original.clone();
            match mutation {
                0 => value["workspace_members"][1] = value["workspace_members"][0].clone(),
                1 => value["packages"][1]["id"] = value["packages"][0]["id"].clone(),
                2 => value["packages"][1]["name"] = value["packages"][0]["name"].clone(),
                3 => value["packages"][0]["dependencies"][1]["kind"] = serde_json::Value::Null,
                4 => value["packages"][1]["publish"] = serde_json::json!(["private-registry"]),
                5 => value["workspace_members"][1] = serde_json::json!("missing-id"),
                _ => unreachable!(),
            }
            assert!(
                super::selected_projection(&serde_json::to_vec(&value).unwrap(), script).is_err(),
                "accepted mutation {mutation}"
            );
        }
    }
}

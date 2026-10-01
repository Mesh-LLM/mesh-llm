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

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let pairs = match args {
        [] => false,
        [flag] if flag == "--dependency-pairs" => true,
        _ => {
            return Err(
                "usage: repository publish-order [--dependency-pairs] < cargo-metadata.json".into(),
            );
        }
    };
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input)?;
    let output = projection(&input, pairs)?;
    crate::repository::check_report::CheckReport::success(output).emit()
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
}

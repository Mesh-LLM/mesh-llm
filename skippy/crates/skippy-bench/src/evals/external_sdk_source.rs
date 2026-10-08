//! Exact external research SDK leaves; no interpreter source is generated in Mesh.
use super::*;

const MAX_LEAF_BYTES: u64 = 1048576;
const LEAVES: &[(&str, &str)] = &[
    (
        "speed-project/pyproject.toml",
        "15c7f2a51ecd86c11c456a1954fa59c66acf96ffc006ae8ae348aaa9c72b76ec",
    ),
    (
        "speed-project/uv.lock",
        "008fad1e5ea48efde545ef3b05fa9c45700b32d0fc7db1584c9f3487f82634b2",
    ),
    (
        "speed-prepare-dataset.py",
        "1c0016902dbde196305d72a46830866d54f5ef5b51d9a5805615ea80f061ce14",
    ),
    (
        "swe-project/pyproject.toml",
        "6d6600df827e367958a35ee56c8813d8c247b37e9dc2db655e4ce7ad2e855b61",
    ),
    (
        "swe-project/uv.lock",
        "85aceb40793727186c4af3a36a614026af56e906f3e36948869b5aa39a92a7a8",
    ),
    (
        "mcp-import-probe.py",
        "d5d5fba746854a16485aefc6ee67a0845fa08b7c0f1fa190a35e6b8ce0b3ae18",
    ),
    (
        "speed-bench-auth.py",
        "fe838b99dc1824f3629d920eca6277cd7813fc5cd4ff5a0fbdf675ba34e64e71",
    ),
    (
        "swe-evaluate.py",
        "04a7f6cb96972ae456d199121660224dc92cc8e9119cefc8ee6565c2c53dc5a5",
    ),
    (
        "swe-expert-instances.py",
        "70323222f5703430e09d7d879b4f1fcb003f0dd33fd27dc8bcd067d85976e9d3",
    ),
    (
        "swe-generate-instances.py",
        "134464e52e287ed9d402c1cfec79afe239d56a51c852d42b870a209c6ae5ae48",
    ),
    (
        "swe-import-probe.py",
        "c66c5603c5dc7758877361bb9a0b09fd07b2723800ed1aa37acf41e57978a60f",
    ),
    (
        "swerex-modal/deployment/config.py",
        "a19643cd219c3fa15de001cb520cf087165c64b47c264688dfa4c26385b02e45",
    ),
    (
        "swerex-modal/deployment/modal.py",
        "a827f2b4c90cc56aeffb152b40a24a941af75360735613ee7a8abb457daddc41",
    ),
    (
        "swerex-modal/runtime/remote.py",
        "7c0dd7a42dfcc99b2dc51f961cb0fe2aee81c09caae08a256694b4e37b4da532",
    ),
    (
        "swerex-sdk-client.py",
        "81bea2daff30ff7d9abcb59d560ad02569552f1e1b63202f0d3857aea1513825",
    ),
];

pub(super) fn leaf(name: &str) -> Result<PathBuf> {
    let pin = LEAVES
        .iter()
        .find(|(path, _)| *path == name)
        .map(|(_, pin)| *pin)
        .context("unknown external SDK leaf")?;
    let root = PathBuf::from(
        env::var_os("MESH_PYTHON_RESEARCH_SOURCE")
            .context("MESH_PYTHON_RESEARCH_SOURCE must name the admitted research checkout")?,
    );
    if !root.is_absolute()
        || root
            .to_str()
            .is_none_or(|v| v.chars().any(char::is_control))
    {
        bail!("external research source needs an absolute control-free UTF-8 root");
    }
    admit(&root.canonicalize()?.join("benchmark-adapters"), name, pin)
}

fn admit(root: &Path, name: &str, pin: &str) -> Result<PathBuf> {
    if !Path::new(name)
        .components()
        .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        bail!("external SDK leaf must have a relative normal path");
    }
    let path = root.join(name);
    let bytes = sdk_environment::read(&path, MAX_LEAF_BYTES)?;
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(root.canonicalize()?) || sdk_environment::hash_bytes(&bytes) != pin {
        bail!("external SDK leaf identity or containment changed");
    }
    Ok(canonical)
}

pub(super) fn read(name: &str) -> Result<Vec<u8>> {
    let path = leaf(name)?;
    let bytes = sdk_environment::read(&path, MAX_LEAF_BYTES)?;
    let pin = LEAVES
        .iter()
        .find(|(p, _)| *p == name)
        .context("unknown external SDK leaf")?
        .1;
    if sdk_environment::hash_bytes(&bytes) != pin {
        bail!("external SDK leaf changed during read");
    }
    Ok(bytes)
}

pub(super) fn admit_run(id: EvalId) -> Result<()> {
    let names: &[&str] = match id {
        EvalId::SpeedBench => &[
            "speed-bench-auth.py",
            "speed-prepare-dataset.py",
            "speed-project/pyproject.toml",
            "speed-project/uv.lock",
        ],
        EvalId::McpAtlas => &["mcp-import-probe.py"],
        EvalId::SweBenchPro => &[
            "swe-import-probe.py",
            "swe-generate-instances.py",
            "swe-expert-instances.py",
            "swe-evaluate.py",
        ],
        _ => &[],
    };
    for name in names {
        leaf(name)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_leaf_bytes_and_mutation_refusal() {
        let fixture = tempfile::tempdir().unwrap();
        let path = fixture.path().join("leaf");
        fs::write(&path, b"inert SDK source fixture").unwrap();
        let pin = sdk_environment::hash_bytes(b"inert SDK source fixture");
        assert_eq!(
            admit(fixture.path(), "leaf", &pin).unwrap(),
            path.canonicalize().unwrap()
        );
        fs::write(&path, b"changed SDK source fixture").unwrap();
        assert!(admit(fixture.path(), "leaf", &pin).is_err());
        assert!(admit(fixture.path(), "missing", &pin).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn link_and_outside_parent_refused() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("leaf"), b"inert").unwrap();
        std::os::unix::fs::symlink(outside.path().join("leaf"), root.path().join("leaf")).unwrap();
        let pin = sdk_environment::hash_bytes(b"inert");
        assert!(admit(root.path(), "leaf", &pin).is_err());
        assert!(admit(root.path(), "../missing", &pin).is_err());
    }
}

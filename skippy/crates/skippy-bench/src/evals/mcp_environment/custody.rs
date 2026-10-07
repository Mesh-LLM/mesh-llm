use super::super::sdk_environment;
use super::*;
pub(super) use sdk_environment::{capture_tools, hash, read};
const SOURCE_PINS: &[(&str, &str)] = &[
    (
        "uv.lock",
        "d66f1df735362c8bba61159485763e023f8d55ff9df89fab75618457e3e03aa9",
    ),
    (
        "pyproject.toml",
        "ce32787701e018aff6b6dad1f286b88b2a4d97925bbf6ae266f9770984d45c59",
    ),
    (
        "mcp_completion_script.py",
        "3f6fbc679612c355734138bf81dd3aaeb9dccda856de34da101f577747619309",
    ),
    (
        "mcp_evals_scores.py",
        "17eba75dff9d71ac1b2835a1b2f80990c4ace8f03294bb235177a00a596a88ab",
    ),
    (
        "mcp_completion/__init__.py",
        "6a808c118f4296e39719be191b2cfe9bfdfd3074df75418f3b2064e75f7971ba",
    ),
    (
        "mcp_completion/agent_eval.py",
        "e65679e3195d641b8b1404f0ade7ec2b7a678ddba94bb442d90e30e1fa6ba2e0",
    ),
    (
        "mcp_completion/config.py",
        "21c4fa5d37c338ceb5fca7fc6b3c0b93ab12a8ca7da599c77d12f877d2c86376",
    ),
    (
        "mcp_completion/errors.py",
        "77ec191469b0d4d88925c7daf476ef777588a86476ee9596b7b9c06793500d67",
    ),
    (
        "mcp_completion/llm.py",
        "eee1cb2625157a063357cdfc093f3cd3c841040f52f36a623db0a657df9d2e89",
    ),
    (
        "mcp_completion/main.py",
        "0e3f15453e410998ea82c07542dca56ade3f8714166d5d4720bd92f2159b6b0d",
    ),
    (
        "mcp_completion/mcp_client/__init__.py",
        "bf155ff2ebd6be1dd9137dff9c9fe3c7baf9d019040ec1d820f6fa3971c1ff6d",
    ),
    (
        "mcp_completion/mcp_client/base_client.py",
        "6187575fac439e7c3eec9b4c30f4e7691740fd9a2a148c8e0101b917b060bd41",
    ),
    (
        "mcp_completion/mcp_client/sandbox_client.py",
        "430c613db9bf1385c2228b43ce73f476a4fb8b50ddd12dd7600237eee1060c0e",
    ),
    (
        "mcp_completion/schema.py",
        "d8a91dc5fe98ff151ad676a1c693627c1c03cb78da6bcd9fd91232b5ee436222",
    ),
];
pub(super) fn source(project: &Path) -> Result<()> {
    source_expected(project, SOURCE_PINS)
}
pub(super) fn source_expected(project: &Path, expected: &[(&str, &str)]) -> Result<()> {
    for (path, pin) in expected {
        if hash(&project.join(path))? != *pin {
            bail!("MCP source/lock changed: {path}");
        }
    }
    let actual = sdk_environment::package_roster(project, &project.join("mcp_completion"))?;
    let wanted: BTreeMap<PathBuf, String> = expected
        .iter()
        .filter(|(p, _)| p.starts_with("mcp_completion/"))
        .map(|(p, h)| (PathBuf::from(*p), format!("file:{h}")))
        .collect();
    if actual != wanted {
        bail!("MCP package source contains unadmitted files or bytecode");
    }
    Ok(())
}
pub(super) fn tools(receipt: &Receipt) -> Result<()> {
    if capture_tools(&receipt.uv, &receipt.python)? != receipt.tool_pins {
        bail!("MCP prepared tools changed");
    }
    Ok(())
}
pub(super) fn environment(root: &Path, python: &Path) -> Result<BTreeMap<PathBuf, String>> {
    sdk_environment::environment(root, python, sdk_environment::PythonProfile::Mcp312)
}

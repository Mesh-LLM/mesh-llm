use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct OtherShard {
    pub(super) schema_version: u32,
    pub(super) groups: Vec<Group<OtherMember>>,
    pub(super) python_implementation_edges: Vec<Group<OtherMember>>,
    pub(super) outside_scanner_source_calls: Vec<OutsideCall>,
    #[serde(default)]
    pub(super) manual_tsv_commands: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub(super) manual_tsv_rows: Vec<ManualTsvRow>,
}

pub(super) type ManualTsvRow = (usize, String, String, String, String);

pub(super) type OtherMember = (usize, String, usize, OtherDisposition, String, String);

#[derive(Deserialize)]
pub(super) struct OutsideCall {
    pub(super) file: String,
    pub(super) line: usize,
    pub(super) source_block: String,
    pub(super) target: String,
    pub(super) boundary: String,
    pub(super) kind: OutsideKind,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum OutsideKind {
    Instruction,
    Provisioning,
    Data,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum OtherDisposition {
    Instruction,
    Provisioning,
    Selection,
    Prose,
    Data,
    Execution,
    Conditional,
}

#[derive(Deserialize)]
pub(super) struct Shard<G> {
    pub(super) schema_version: u32,
    pub(super) groups: Vec<G>,
}

#[derive(Deserialize)]
pub(super) struct Group<M> {
    pub(super) file: String,
    pub(super) members: Vec<M>,
}

pub(super) type GithubMember = (usize, String, usize, GithubKind, String);
pub(super) type ScriptMember = (usize, String, usize, ScriptKind, String, String);
#[derive(Deserialize)]
pub(super) struct ScriptShard {
    pub(super) schema_version: u32,
    pub(super) groups: Vec<Group<ScriptMember>>,
    #[serde(default)]
    pub(super) python_implementation_groups: Vec<PythonGroup>,
    #[serde(default)]
    pub(super) python_implementation_edges: Vec<PythonEdge>,
    #[serde(default)]
    pub(super) python_implementation_sources: Vec<(String, String, usize)>,
    #[serde(default)]
    pub(super) source_calls: Vec<ScriptSourceCall>,
}

#[derive(Deserialize)]
pub(super) struct ScriptSourceCall {
    pub(super) id: String,
    pub(super) context: String,
    pub(super) target: String,
    pub(super) boundary: String,
    pub(super) owner: String,
    pub(super) replacement: String,
    pub(super) deletion_condition: String,
}

#[derive(Deserialize)]
pub(super) struct PythonGroup {
    pub(super) file: String,
    pub(super) root: String,
    pub(super) boundary: String,
    pub(super) members: Vec<PythonMember>,
}

pub(super) type PythonMember = (
    usize,
    String,
    usize,
    TestKind,
    PythonDisposition,
    String,
    String,
);

#[derive(Deserialize)]
pub(super) struct PythonEdge {
    pub(super) id: String,
    pub(super) source: String,
    pub(super) source_block: String,
    pub(super) disposition: PythonDisposition,
    pub(super) caller_contract: String,
    pub(super) target: String,
    pub(super) root: String,
    pub(super) boundary: String,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum PythonDisposition {
    Execution,
    Indirect,
    Selection,
    Data,
}
pub(super) type TestMember = (
    usize,
    String,
    usize,
    TestKind,
    TestDisposition,
    String,
    String,
);

#[derive(Deserialize)]
pub(super) struct TestShard {
    pub(super) schema_version: u32,
    pub(super) groups: Vec<Group<TestMember>>,
    #[serde(default)]
    pub(super) additional_groups: Vec<(String, Vec<AdditionalTestCase>)>,
    #[serde(default)]
    pub(super) observed_source_sha256: String,
}

pub(super) type AdditionalTestCase = (TestDisposition, String, String, Vec<usize>);

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum GithubKind {
    Script,
    Inline,
    Installer,
    Provision,
    Metadata,
    Probe,
    Selector,
    Comment,
}

impl GithubKind {
    const fn executable(&self) -> bool {
        matches!(self, Self::Script | Self::Inline | Self::Installer)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum ScriptKind {
    Script,
    Inline,
    Selection,
    Data,
    Indirect,
    Environment,
}

impl ScriptKind {
    const fn executable(&self) -> bool {
        matches!(self, Self::Script | Self::Inline | Self::Environment)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum TestKind {
    DynamicImport,
    SubprocessOrInterpreter,
}

impl TestKind {
    pub(super) const fn label(&self) -> &'static str {
        match self {
            Self::DynamicImport => "dynamic-import",
            Self::SubprocessOrInterpreter => "subprocess-or-interpreter",
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum TestDisposition {
    Import,
    Launch,
    Conditional,
    Assembly,
    Fixture,
}

impl TestDisposition {
    const fn executable(&self) -> bool {
        matches!(
            self,
            Self::Import | Self::Launch | Self::Conditional | Self::Fixture
        )
    }
}

pub(super) struct Member {
    pub(super) line: usize,
    pub(super) hash: String,
    pub(super) occurrence: usize,
    pub(super) kind: &'static str,
    pub(super) executable: bool,
    pub(super) details: Vec<String>,
}

impl From<GithubMember> for Member {
    fn from((line, hash, occurrence, kind, detail): GithubMember) -> Self {
        Self {
            line,
            hash,
            occurrence,
            executable: kind.executable(),
            kind: "candidate",
            details: vec![detail],
        }
    }
}

impl From<ScriptMember> for Member {
    fn from((line, hash, occurrence, kind, target, contract): ScriptMember) -> Self {
        Self {
            line,
            hash,
            occurrence,
            executable: kind.executable(),
            kind: "candidate",
            details: vec![target, contract],
        }
    }
}

impl From<TestMember> for Member {
    fn from((line, hash, occurrence, kind, disposition, target, reason): TestMember) -> Self {
        Self {
            line,
            hash,
            occurrence,
            executable: disposition.executable(),
            kind: kind.label(),
            details: vec![target, reason],
        }
    }
}

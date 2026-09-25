use crate::command::{DynResult, run_command, trimmed_stderr_or_stdout};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;

const LEDGER_DIR: &str = "ci/automation-migration";

#[derive(Deserialize)]
pub(super) struct PythonFile {
    pub(super) path: String,
    pub(super) classification: String,
    pub(super) replacement_owner: String,
    pub(super) deletion_condition: String,
}

#[derive(Deserialize)]
pub(super) struct Inventory {
    pub(super) schema_version: u32,
    pub(super) files: Vec<PythonFile>,
}

#[derive(Clone, Deserialize)]
pub(super) struct VerifiedEdge {
    pub(super) id: String,
    pub(super) owner: String,
    pub(super) reason: String,
    pub(super) target: String,
    pub(super) replacement_task: u32,
}

#[derive(Deserialize)]
pub(super) struct GithubEdge {
    pub(super) id: String,
    pub(super) source_block: String,
    pub(super) owner: String,
    pub(super) disposition: String,
    pub(super) argv: String,
    pub(super) status_output_effects: String,
    pub(super) transitive_boundary: String,
    pub(super) replacement: String,
    pub(super) deletion_condition: String,
    pub(super) reason: String,
}

#[derive(Deserialize)]
pub(super) struct CandidateCensus {
    pub(super) acceptance: bool,
}

#[derive(Deserialize)]
pub(super) struct InvocationLedger {
    pub(super) schema_version: u32,
    pub(super) source_verified_edges: Vec<VerifiedEdge>,
    pub(super) github_source_records: Vec<GithubEdge>,
    #[serde(default)]
    pub(super) selected_process_calls: Vec<SelectedProcessCall>,
    #[serde(default)]
    pub(super) runner_image_planner_loader: Option<PlannerLoader>,
    #[serde(default)]
    pub(super) family_canary_loaders: Vec<FamilyLoader>,
    pub(super) reproducible_candidate_census: CandidateCensus,
}

#[derive(Deserialize)]
pub(super) struct BoundaryLedger {
    #[serde(default)]
    pub(super) boundary_records: Vec<BoundaryRecord>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum BoundaryDisposition {
    ExternalTrustBoundary,
    PlatformConditional,
    RuntimeSelectedBoundary,
    BoundedSelector,
    FiniteTarget,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SourceEvidence {
    pub(super) path: String,
    pub(super) line: usize,
    pub(super) text: String,
}

/// Binds by caller, line, exact source block and physical occurrence; when the
/// scanner observed that occurrence, `candidate_id` must equal its validated ID.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BoundaryRecord {
    pub(super) id: String,
    pub(super) disposition: BoundaryDisposition,
    pub(super) caller: String,
    pub(super) line: usize,
    pub(super) source_block: String,
    pub(super) occurrence: usize,
    pub(super) candidate_id: Option<String>,
    pub(super) child: Option<String>,
    #[serde(default)]
    pub(super) bound_child: Option<String>,
    pub(super) selector: String,
    pub(super) reachable_bytes: String,
    pub(super) replacement_owner: String,
    pub(super) deletion_condition: String,
    pub(super) rationale: String,
    pub(super) evidence: Vec<SourceEvidence>,
}

#[derive(Clone, Deserialize)]
pub(super) struct PlannerLoader {
    pub(super) caller: String,
    pub(super) source_block: String,
    pub(super) target: String,
    pub(super) target_sha256: String,
    pub(super) invocation: String,
    pub(super) status_streams_effects: String,
    pub(super) replacement_owner: String,
    pub(super) deletion_phase: u32,
    pub(super) deletion_condition: String,
}

#[derive(Clone, Deserialize)]
pub(super) struct FamilyLoader {
    pub(super) edge_id: String,
    pub(super) caller: String,
    pub(super) source_block: String,
    pub(super) target: String,
    pub(super) target_sha256: String,
    pub(super) resolution: String,
    pub(super) invocation: String,
    pub(super) status_streams_effects: String,
    pub(super) descendant_processes: String,
    pub(super) replacement_owner: String,
    pub(super) deletion_phase: u32,
    pub(super) deletion_condition: String,
}

#[derive(Clone, Deserialize)]
pub(super) struct SelectedProcessCall {
    pub(super) caller: String,
    pub(super) line: usize,
    pub(super) source_block: String,
    pub(super) child: String,
    pub(super) argv: String,
    pub(super) status_streams_effects: String,
    pub(super) replacement_owner: String,
    pub(super) child_source_known: bool,
}

#[derive(Deserialize)]
pub(super) struct InstructionAsset {
    pub(super) path: String,
    pub(super) classification: String,
    pub(super) owner: String,
}

#[derive(Deserialize)]
pub(super) struct InstructionLedger {
    pub(super) schema_version: u32,
    pub(super) assets: Vec<InstructionAsset>,
}

#[derive(Deserialize)]
pub(super) struct ExceptionLedger {
    pub(super) schema_version: u32,
    pub(super) exceptions: Vec<ExceptionEntry>,
}

#[derive(Deserialize)]
pub(super) struct ExceptionEntry {
    pub(super) path: String,
    pub(super) status: String,
    pub(super) callers: Option<Vec<String>>,
    pub(super) local_dependency_files: Option<Vec<String>>,
    pub(super) cadence: Option<String>,
    pub(super) purpose: Option<String>,
    pub(super) isolation_test: Option<String>,
}

pub(super) struct MigrationLedgers {
    pub(super) inventory: Inventory,
    pub(super) invocations: InvocationLedger,
    pub(super) instructions: InstructionLedger,
    pub(super) exceptions: ExceptionLedger,
}

impl MigrationLedgers {
    pub(super) fn load(root: &Path) -> DynResult<Self> {
        let load = |name: &str| -> DynResult<Vec<u8>> {
            let path = root.join(LEDGER_DIR).join(name);
            fs::read(&path).map_err(|error| format!("{}: {error}", path.display()).into())
        };
        Ok(Self {
            inventory: serde_json::from_slice(&load("python-inventory.json")?)?,
            invocations: serde_json::from_slice(&load("invocations.json")?)?,
            instructions: serde_json::from_slice(&load("instructions.json")?)?,
            exceptions: serde_json::from_slice(&load("python-exceptions.json")?)?,
        })
    }

    pub(super) fn versions(&self) -> DynResult<()> {
        if [
            self.inventory.schema_version,
            self.invocations.schema_version,
            self.instructions.schema_version,
            self.exceptions.schema_version,
        ] != [1; 4]
        {
            return Err("automation migration: unsupported ledger schema version".into());
        }
        Ok(())
    }
}

pub(super) fn tracked_paths(root: &Path) -> DynResult<Vec<String>> {
    let mut git = Command::new("git");
    git.current_dir(root).env("GIT_MASTER", "1").args([
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]);
    let output = run_command(&mut git)?;
    if !output.status.success() {
        return Err(format!("git ls-files failed: {}", trimmed_stderr_or_stdout(&output)).into());
    }
    Ok(String::from_utf8(output.stdout)?
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect())
}

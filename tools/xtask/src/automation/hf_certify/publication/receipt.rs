use super::contract::{PublisherInput, hex};
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::{Value, json};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Object {
    oid: String,
    size: u64,
    mutation_attempted: bool,
    uploaded_parts: usize,
    object_present: bool,
    source_custody_verified: bool,
    completed: bool,
    error: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Publication {
    schema_version: u32,
    repo: String,
    parent_commit: String,
    ordered_paths: Vec<String>,
    objects: Vec<Object>,
    object_attempted_paths: Vec<String>,
    commit_attempted: bool,
    commit_oid: Option<String>,
    remote_verified_paths: Vec<String>,
    final_source_custody_verified: bool,
    completed: bool,
    error: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Receipt {
    schema_version: u32,
    request_sha256: String,
    status: String,
    publication: Option<Publication>,
    source_custody_verified: bool,
    error: Option<String>,
}
impl Receipt {
    pub(super) fn observe(
        &self,
        input: &PublisherInput,
        hash: &str,
        progress: bool,
    ) -> DynResult<Value> {
        if self.schema_version != 1
            || self.request_sha256 != hash
            || (progress && self.status != "IN_PROGRESS")
            || (!progress && !["FAILED", "PUBLISHED"].contains(&self.status.as_str()))
            || self.error.as_ref().is_some_and(|e| e.len() > 1024)
        {
            return Err("publication receipt correlation refused".into());
        }
        let observed = match &self.publication {
            Some(p) => p.observe(input)?,
            None => Value::Null,
        };
        if (progress && self.publication.as_ref().is_some_and(|p| p.completed))
            || (self.status == "PUBLISHED"
                && (self.error.is_some()
                    || !self.source_custody_verified
                    || self.publication.as_ref().is_none_or(|p| !p.completed)))
        {
            return Err("publication receipt completion incoherent".into());
        }
        Ok(
            json!({"schema_version":1,"request_sha256":hash,"status":self.status,"publication":observed,
            "source_custody_verified":self.source_custody_verified,"error_present":self.error.is_some()}),
        )
    }
    pub(super) fn accepted(&self) -> bool {
        self.status == "PUBLISHED"
            && self.error.is_none()
            && self.source_custody_verified
            && self.publication.as_ref().is_some_and(|p| {
                p.completed && p.error.is_none() && p.objects.iter().all(|o| o.error.is_none())
            })
    }
}
impl Publication {
    fn observe(&self, input: &PublisherInput) -> DynResult<Value> {
        let paths = input
            .shards
            .iter()
            .chain(&input.sidecars)
            .map(|a| a.path_in_repo.clone())
            .collect::<Vec<_>>();
        let attempted = input
            .shards
            .iter()
            .map(|a| a.path_in_repo.clone())
            .collect::<Vec<_>>();
        if self.schema_version != 1
            || self.repo != input.repo
            || self.parent_commit != input.parent_commit
            || self.ordered_paths != paths
            || !attempted.starts_with(&self.object_attempted_paths)
            || !paths.starts_with(&self.remote_verified_paths)
            || self.objects.len() > input.shards.len()
            || self.objects.len() > self.object_attempted_paths.len()
            || self.commit_oid.as_ref().is_some_and(|oid| !hex(oid, 40))
            || (self.commit_oid.is_some() && !self.commit_attempted)
            || (!self.remote_verified_paths.is_empty() && self.commit_oid.is_none())
            || self.error.as_ref().is_some_and(|e| e.len() > 1024)
        {
            return Err("publication ordered identity/partial roster refused".into());
        }
        let mut objects = Vec::new();
        for (object, artifact) in self.objects.iter().zip(&input.shards) {
            if object.oid != artifact.sha256
                || object.size != artifact.byte_size
                || object.uploaded_parts > 10000
                || object.error.as_ref().is_some_and(|e| e.len() > 1024)
            {
                return Err("publication object correlation refused".into());
            }
            objects.push(json!({"oid":object.oid,"size":object.size,"mutation_attempted":object.mutation_attempted,
                "uploaded_parts":object.uploaded_parts,"object_present":object.object_present,
                "source_custody_verified":object.source_custody_verified,"completed":object.completed,"error_present":object.error.is_some()}));
        }
        if self.completed
            && (self.objects.len() != input.shards.len()
                || !self.objects.iter().all(|o| {
                    o.completed
                        && o.source_custody_verified
                        && o.object_present
                        && o.error.is_none()
                })
                || self.remote_verified_paths != paths
                || self.commit_oid.is_none()
                || !self.final_source_custody_verified
                || self.error.is_some())
        {
            return Err("publication complete receipt lacks immutable custody".into());
        }
        Ok(
            json!({"schema_version":1,"repo":self.repo,"parent_commit":self.parent_commit,"ordered_paths":self.ordered_paths,
            "objects":objects,"object_attempted_paths":self.object_attempted_paths,"commit_attempted":self.commit_attempted,
            "commit_oid":self.commit_oid,"remote_verified_paths":self.remote_verified_paths,"final_source_custody_verified":self.final_source_custody_verified,
            "completed":self.completed,"error_present":self.error.is_some()}),
        )
    }
}

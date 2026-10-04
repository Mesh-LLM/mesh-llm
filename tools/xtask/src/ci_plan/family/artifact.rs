use super::document::Json;
use super::fields::{PlanResult, exact, hex_sha, object, string, strings};
use super::integer::Integer;
use super::projection::{ToJson, object as json_object};
use super::text::FamilyString;
use std::collections::BTreeMap;

pub(super) struct Integrity {
    size_bytes: Integer,
    blob_id: FamilyString,
}

pub(super) struct Artifact {
    repo: FamilyString,
    revision: FamilyString,
    pub(super) files: Vec<FamilyString>,
    file_integrity: BTreeMap<FamilyString, Integrity>,
    selector: FamilyString,
}

impl ToJson for Integrity {
    fn to_json(&self) -> Json {
        json_object([
            ("size_bytes", self.size_bytes.to_json()),
            ("blob_id", self.blob_id.to_json()),
        ])
    }
}

impl ToJson for Artifact {
    fn to_json(&self) -> Json {
        json_object([
            ("repo", self.repo.to_json()),
            ("revision", self.revision.to_json()),
            ("files", self.files.to_json()),
            ("file_integrity", self.file_integrity.to_json()),
            ("selector", self.selector.to_json()),
        ])
    }
}

pub(super) fn parse(value: Option<&Json>, field: &str) -> PlanResult<Artifact> {
    let artifact = object(value, field)?;
    exact(
        artifact,
        &["repo", "revision", "files", "file_integrity", "selector"],
        field,
    )?;
    let repo = string(artifact.get("repo"), &format!("{field}.repo"))?;
    if repo.codes().iter().filter(|code| **code == 0x2f).count() != 1
        || repo.codes().first() == Some(&0x2f)
        || repo.codes().last() == Some(&0x2f)
    {
        return Err(format!(
            "{field}.repo must be an owner/repository coordinate"
        ));
    }
    let revision = string(artifact.get("revision"), &format!("{field}.revision"))?;
    if !revision
        .scalar_text()
        .is_some_and(|text| hex_sha(&text, 40, 64))
    {
        return Err(format!(
            "{field}.revision must be a lowercase immutable SHA"
        ));
    }
    let files = strings(artifact.get("files"), &format!("{field}.files"))?;
    if files.is_empty() {
        return Err(format!("{field}.files must not be empty"));
    }
    for file in &files {
        if file.codes().first() == Some(&0x2f)
            || file.codes().last() == Some(&0x2f)
            || file
                .codes()
                .split(|code| *code == 0x2f)
                .any(|part| part == [0x2e, 0x2e])
        {
            return Err(format!(
                "{field}.files contains an unsafe path: {}",
                file.diagnostic()
            ));
        }
    }
    let integrity_field = format!("{field}.file_integrity");
    let integrity = object(artifact.get("file_integrity"), &integrity_field)?;
    let entries = integrity
        .as_object()
        .ok_or_else(|| format!("{integrity_field} must be an object"))?;
    if entries.len() != files.len() || entries.iter().any(|(key, _)| !files.contains(key)) {
        return Err(format!(
            "{integrity_field} must exactly cover {field}.files"
        ));
    }
    let mut file_integrity = BTreeMap::new();
    for file in &files {
        let record_field = format!("{integrity_field}[{}]", file.diagnostic());
        let record = object(integrity.get_key(file), &record_field)?;
        exact(record, &["size_bytes", "blob_id"], &record_field)?;
        let size_bytes = Integer::parse(
            record.get("size_bytes"),
            &format!("{record_field}.size_bytes"),
            1,
        )?;
        let blob_id = string(record.get("blob_id"), &format!("{record_field}.blob_id"))?;
        if !blob_id
            .scalar_text()
            .is_some_and(|text| hex_sha(&text, 64, 64))
        {
            return Err(format!(
                "{record_field}.blob_id must be a lowercase SHA-256"
            ));
        }
        file_integrity.insert(
            file.clone(),
            Integrity {
                size_bytes,
                blob_id,
            },
        );
    }
    let selector = string(artifact.get("selector"), &format!("{field}.selector"))?;
    Ok(Artifact {
        repo,
        revision,
        files,
        file_integrity,
        selector,
    })
}

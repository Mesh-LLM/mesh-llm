//! Bind the standalone Skippy qualification receipt record into a composed
//! product manifest. The record's provenance is established by `qualify
//! host-receipts`; this structural re-check (owned by
//! `crate::qualification::host_receipts::validate_record`) rejects a truncated,
//! swapped, or foreign record before it reaches a composed product.

use super::digest::file_sha256;
use super::pure_path::PurePath;
use crate::ci_plan::document::Json;
use crate::qualification::contract;
use crate::qualification::host_receipts;
use std::path::Path;

/// Filename the record takes inside a composed bundle, discovered
/// automatically so a recomposition check binds the same record.
pub(crate) const RECORD_NAME: &str = host_receipts::RECORD_NAME;

fn text(value: &str) -> Json {
    Json::String(value.to_owned())
}

pub(crate) fn bind(bundle: &PurePath, record_path: &Path) -> Result<Json, String> {
    if !record_path.is_file() {
        return Err(format!(
            "qualification receipt record is missing: {}",
            record_path.display()
        ));
    }
    let record = contract::load_json(record_path)
        .map_err(|error| format!("qualification receipt record is not JSON: {error}"))?;
    let record = host_receipts::validate_record(&record)?;
    let relative = PurePath::new(record_path.to_string_lossy().as_ref()).relative_to(bundle)?;
    let record_sha256 = file_sha256(record_path).map_err(|failure| failure.error.to_string())?;
    Ok(Json::Object(vec![
        ("schema_version".to_owned(), Json::Number(1.into())),
        ("record_path".to_owned(), text(&relative)),
        ("record_sha256".to_owned(), text(&record_sha256)),
        ("source_sha".to_owned(), text(&record.source_sha)),
        ("plan_digest".to_owned(), text(&record.plan_digest)),
        ("platform".to_owned(), text(&record.platform)),
        ("required_row".to_owned(), text(&record.required_row)),
        (
            "receipts".to_owned(),
            Json::Array(
                record
                    .receipts
                    .iter()
                    .map(|(row_id, status, digest)| {
                        Json::Object(vec![
                            ("row_id".to_owned(), text(row_id)),
                            ("status".to_owned(), text(status)),
                            ("receipt_sha256".to_owned(), text(digest)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qualification::contract;

    fn record(platform: &str, required: &str, rows: &[(&str, &str)]) -> Json {
        contract::parse_strict(
            serde_json::json!({
                "schema_version": 1,
                "source_sha": "a".repeat(40),
                "plan_digest": "b".repeat(64),
                "platform": platform,
                "required_row": required,
                "receipts": rows
                    .iter()
                    .map(|(row, status)| serde_json::json!({
                        "row_id": row,
                        "status": status,
                        "receipt_sha256": "c".repeat(64),
                    }))
                    .collect::<Vec<_>>(),
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap()
    }

    fn bundle() -> (tempfile::TempDir, PurePath) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().to_string_lossy().replace('\\', "/");
        (temp, PurePath::new(&root))
    }

    #[test]
    fn binds_a_qualified_record_into_the_manifest() {
        let (temp, bundle) = bundle();
        let path = temp.path().join(RECORD_NAME);
        let record = record(
            "linux",
            "linux-cpu",
            &[
                ("linux-cpu", "qualified"),
                ("linux-cuda", "hardware-unavailable"),
            ],
        );
        std::fs::write(&path, contract::render(&record.to_value()).unwrap()).unwrap();
        let bound = bind(&bundle, &path).expect("valid record");
        assert_eq!(
            bound.get("record_path").and_then(Json::as_str),
            Some(RECORD_NAME)
        );
        assert_eq!(
            bound.get("required_row").and_then(Json::as_str),
            Some("linux-cpu")
        );
        let receipts = bound.get("receipts").and_then(Json::as_array).unwrap();
        assert_eq!(receipts.len(), 2);
        // Sorted by row, not input order.
        assert_eq!(
            receipts[0].get("row_id").and_then(Json::as_str),
            Some("linux-cpu")
        );
        assert_eq!(
            receipts[1].get("row_id").and_then(Json::as_str),
            Some("linux-cuda")
        );
    }

    #[test]
    fn rejects_missing_foreign_and_unqualified_records() {
        let (temp, bundle) = bundle();
        let missing = temp.path().join(RECORD_NAME);
        assert!(bind(&bundle, &missing).unwrap_err().contains("missing"));

        let path = temp.path().join(RECORD_NAME);
        let write = |record: Json| {
            std::fs::write(&path, contract::render(&record.to_value()).unwrap()).unwrap();
        };
        write(record("plan9", "linux-cpu", &[("linux-cpu", "qualified")]));
        assert!(
            bind(&bundle, &path)
                .unwrap_err()
                .contains("unknown platform")
        );
        write(record(
            "linux",
            "linux-cpu",
            &[("windows-cpu", "qualified")],
        ));
        assert!(bind(&bundle, &path).unwrap_err().contains("foreign row"));
        write(record(
            "linux",
            "linux-cpu",
            &[("linux-cpu", "hardware-unavailable")],
        ));
        assert!(bind(&bundle, &path).unwrap_err().contains("not qualified"));
        write(record(
            "linux",
            "linux-cpu",
            &[("linux-cpu", "qualified"), ("linux-cpu", "qualified")],
        ));
        assert!(bind(&bundle, &path).unwrap_err().contains("repeats"));
        write(record("linux", "linux-cpu", &[("linux-cpu", "green")]));
        assert!(bind(&bundle, &path).unwrap_err().contains("unknown status"));
    }
}

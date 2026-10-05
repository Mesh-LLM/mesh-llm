//! Versioned process boundary between the portable tool and optional reader.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{
    DynResult,
    selection::{Selection, Trajectory},
};

pub const SCHEMA_VERSION: u8 = 1;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u8,
    pub dataset_file: PathBuf,
    pub selection: Selection,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub schema_version: u8,
    pub rows: Vec<SelectedTrajectory>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SelectedTrajectory {
    #[serde(flatten)]
    pub metadata: Trajectory,
    pub messages_json: String,
}

impl From<Trajectory> for SelectedTrajectory {
    fn from(mut row: Trajectory) -> Self {
        let messages_json = std::mem::take(&mut row.messages_json);
        Self {
            metadata: row,
            messages_json,
        }
    }
}

impl Response {
    pub fn into_rows(self, expected: usize) -> DynResult<Vec<Trajectory>> {
        if self.schema_version != SCHEMA_VERSION || self.rows.len() != expected {
            return Err("trajectory reader response schema or row count differs".into());
        }
        Ok(self
            .rows
            .into_iter()
            .map(|row| {
                let mut metadata = row.metadata;
                metadata.messages_json = row.messages_json;
                metadata
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_wire_preserves_message_bodies_while_public_metadata_excludes_them() {
        let row = Trajectory {
            session_id: "session".into(),
            source_dataset: "source".into(),
            messages_json: "[{\"role\":\"user\",\"content\":\"é\"}]".into(),
            n_turns: 20,
            max_isl: 9000,
            total_tokens: 9100,
        };
        assert!(
            serde_json::to_value(&row)
                .unwrap()
                .get("messages_json")
                .is_none()
        );
        let expected = row.clone();
        let response = Response {
            schema_version: SCHEMA_VERSION,
            rows: vec![row.into()],
        };
        let bytes = serde_json::to_vec(&response).unwrap();
        let decoded: Response = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.into_rows(1).unwrap(), vec![expected]);
        let wrong: Response = serde_json::from_slice(&bytes).unwrap();
        assert!(wrong.into_rows(2).is_err());
    }
}

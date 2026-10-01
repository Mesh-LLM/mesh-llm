use super::{GithubMatrix, MatrixRow, Shard};
use crate::ci_plan::family::document::Json;
use crate::ci_plan::family::projection::{ToJson, object};

impl ToJson for Shard {
    fn to_json(&self) -> Json {
        object([
            ("id", self.id.to_json()),
            ("shard_index", self.shard_index.to_json()),
            ("families", self.families.to_json()),
            ("estimated_work_bytes", self.estimated_work_bytes.to_json()),
        ])
    }
}

impl ToJson for MatrixRow {
    fn to_json(&self) -> Json {
        object([
            ("id", self.id.to_json()),
            ("shard_index", self.shard_index.to_json()),
            ("families", self.families.to_json()),
            ("estimated_work_bytes", self.estimated_work_bytes.to_json()),
        ])
    }
}

impl ToJson for GithubMatrix {
    fn to_json(&self) -> Json {
        object([("include", self.include.to_json())])
    }
}

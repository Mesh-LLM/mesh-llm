use super::{Evidence, Json, Model};
use crate::ci_plan::family::projection::{ToJson, object};

impl ToJson for Evidence {
    fn to_json(&self) -> Json {
        object([
            ("fixture", self.fixture.to_json()),
            ("comparison", self.comparison.to_json()),
        ])
    }
}

impl ToJson for Model {
    fn to_json(&self) -> Json {
        let mut entries = vec![
            ("family".into(), self.family.to_json()),
            ("class".into(), self.model_class.to_json()),
            ("architecture".into(), self.architecture.to_json()),
            ("profile".into(), self.profile.to_json()),
            (
                "certification_status".into(),
                self.certification_status.to_json(),
            ),
            ("oracle".into(), self.oracle.to_json()),
            (
                "certification_lanes".into(),
                self.certification_lanes.to_json(),
            ),
            ("artifact".into(), self.artifact.to_json()),
            ("draft_artifact".into(), self.draft_artifact.to_json()),
            ("mmproj_artifact".into(), self.mmproj_artifact.to_json()),
            ("execution".into(), self.execution.to_json()),
            ("resources".into(), self.resources.to_json()),
            ("notes".into(), self.notes.to_json()),
            ("manifest_index".into(), self.manifest_index.to_json()),
        ];
        if let Some(evidence) = &self.evidence {
            entries.push(("evidence".into(), evidence.to_json()));
        }
        Json::Object(entries)
    }
}

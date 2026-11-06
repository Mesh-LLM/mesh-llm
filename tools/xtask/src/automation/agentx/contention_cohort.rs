//! Bounded two-turn sessions for concurrency diagnosis.
use super::Prepared;
use crate::command::DynResult;
use serde_json::Value;

pub(super) const SESSIONS: usize = 16;
pub(super) fn configuration(input: &Prepared, port: u16, concurrency: u32) -> DynResult<Value> {
    super::agentic_config::configuration(
        input,
        port,
        concurrency,
        SESSIONS,
        "mesh-agentx-contention-v2",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_runs_all_sessions_without_scenario_warmup_or_duration() {
        let models: Vec<super::super::Model> = serde_json::from_str(include_str!(
            "../../../../../skippy/evals/kv-validation/models.json"
        ))
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dataset = dir.path().join("traces");
        std::fs::create_dir(&dataset).unwrap();
        std::fs::write(dir.path().join("cohort.json"), b"bounded source cohort").unwrap();
        let input = Prepared {
            tools: None,
            root: "/repo".into(),
            output: "/results".into(),
            weights: "/weights/model.gguf".into(),
            dataset,
            model: models[1].clone(),
            profile: super::super::Profile::Contention,
        };
        let config = configuration(&input, 8080, 16).unwrap();
        let benchmark = &config["benchmark"];
        assert_eq!(benchmark["dataset"]["entries"], SESSIONS);
        assert_eq!(benchmark["phases"][0]["sessions"], SESSIONS);
        assert_eq!(benchmark["phases"][0]["concurrency"], 16);
        assert!(benchmark["phases"][0]["duration"].is_null());
        assert!(benchmark["phases"][0]["warmup_requests_per_lane"].is_null());
        assert!(benchmark["scenario"].is_null());
    }
}

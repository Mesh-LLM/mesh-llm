use super::document::Json;
use super::fields::{PlanResult, choice, exact, number, object};
use super::integer::WorkBytes;
use super::projection::ToJson;

pub(super) struct Resources {
    runner_role: String,
    cache_policy: String,
    pub(super) estimated_model_bytes: WorkBytes,
    startup_timeout_secs: Option<u64>,
    minimum_runner_memory_gib: Option<u64>,
}

impl ToJson for Resources {
    fn to_json(&self) -> Json {
        let mut entries = vec![
            ("runner_role".into(), self.runner_role.to_json()),
            ("cache_policy".into(), self.cache_policy.to_json()),
            (
                "estimated_model_bytes".into(),
                self.estimated_model_bytes.to_json(),
            ),
            (
                "startup_timeout_secs".into(),
                self.startup_timeout_secs.to_json(),
            ),
        ];
        if let Some(memory) = self.minimum_runner_memory_gib {
            entries.push(("minimum_runner_memory_gib".into(), memory.to_json()));
        }
        Json::Object(entries)
    }
}

pub(super) fn parse(value: Option<&Json>, field: &str) -> PlanResult<Resources> {
    let field = format!("{field}.resources");
    let row = object(value, &field)?;
    exact(
        row,
        &[
            "runner_role",
            "cache_policy",
            "estimated_model_bytes",
            "minimum_runner_memory_gib",
            "startup_timeout_secs",
        ],
        &field,
    )?;
    let runner_role = choice(
        row.get("runner_role"),
        &format!("{field}.runner_role"),
        &["family-certify"],
    )?;
    let cache_policy = choice(
        row.get("cache_policy"),
        &format!("{field}.cache_policy"),
        &["immutable-local"],
    )?;
    let estimated_model_bytes = WorkBytes::parse(
        row.get("estimated_model_bytes"),
        &format!("{field}.estimated_model_bytes"),
    )?;
    let startup_timeout_secs = row
        .get("startup_timeout_secs")
        .map(|value| {
            number(
                Some(value),
                &format!("{field}.startup_timeout_secs"),
                180..=1800,
            )
        })
        .transpose()?;
    let minimum_runner_memory_gib = row
        .get("minimum_runner_memory_gib")
        .map(|value| {
            number(
                Some(value),
                &format!("{field}.minimum_runner_memory_gib"),
                128..=256,
            )
        })
        .transpose()?;
    if minimum_runner_memory_gib.is_some_and(|size| size != 128 && size != 256) {
        return Err(format!(
            "{field}.minimum_runner_memory_gib must be 128 or 256"
        ));
    }
    Ok(Resources {
        runner_role,
        cache_policy,
        estimated_model_bytes,
        startup_timeout_secs,
        minimum_runner_memory_gib,
    })
}

use super::integer::{Integer, WorkBytes};
mod projection;

pub(super) struct SelectedFamily<'a> {
    pub(super) family: &'a str,
    pub(super) manifest_index: usize,
    pub(super) estimated_model_bytes: WorkBytes,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ShardError {
    InvalidCount,
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCount => formatter.write_str("--shard-count must be positive"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Shard {
    id: String,
    shard_index: usize,
    families: Vec<String>,
    estimated_work_bytes: Integer,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct MatrixRow {
    id: String,
    shard_index: usize,
    families: String,
    estimated_work_bytes: Integer,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct GithubMatrix {
    pub(super) include: Vec<MatrixRow>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct FamilyProjection {
    pub(super) shards: Vec<Shard>,
    pub(super) github_matrix: GithubMatrix,
}

pub(super) fn shard_families(
    models: &[SelectedFamily<'_>],
    requested_count: usize,
) -> Result<FamilyProjection, ShardError> {
    if requested_count == 0 {
        return Err(ShardError::InvalidCount);
    }
    let count = requested_count.min(models.len());
    let mut ordered = models.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .estimated_model_bytes
            .cmp(&left.estimated_model_bytes)
            .then(left.family.cmp(right.family))
    });
    let mut buckets = (0..count)
        .map(|index| (index, Integer::default(), Vec::<&SelectedFamily<'_>>::new()))
        .collect::<Vec<_>>();
    for model in ordered {
        let bucket = buckets
            .iter_mut()
            .min_by(|left, right| left.1.cmp(&right.1).then(left.0.cmp(&right.0)))
            .ok_or(ShardError::InvalidCount)?;
        bucket.1 = bucket.1.sum(model.estimated_model_bytes.integer());
        bucket.2.push(model);
    }
    let shards = buckets
        .into_iter()
        .map(|(index, total, mut members)| {
            members.sort_by_key(|model| model.manifest_index);
            Shard {
                id: format!("family-battery-{:02}", index + 1),
                shard_index: index,
                families: members
                    .iter()
                    .map(|model| model.family.to_owned())
                    .collect(),
                estimated_work_bytes: total,
            }
        })
        .collect::<Vec<_>>();
    let mut scheduled = shards.iter().collect::<Vec<_>>();
    scheduled.sort_by(|left, right| {
        left.estimated_work_bytes
            .cmp(&right.estimated_work_bytes)
            .then(left.families.cmp(&right.families))
    });
    let github_matrix = GithubMatrix {
        include: scheduled
            .into_iter()
            .map(|shard| MatrixRow {
                id: shard.id.clone(),
                shard_index: shard.shard_index,
                families: shard.families.join(","),
                estimated_work_bytes: shard.estimated_work_bytes.clone(),
            })
            .collect(),
    };
    Ok(FamilyProjection {
        shards,
        github_matrix,
    })
}

#[cfg(test)]
mod tests {
    use super::{SelectedFamily, WorkBytes, shard_families};
    use crate::ci_plan::family::document;
    use crate::ci_plan::family::{output, projection::ToJson};
    use serde::Deserialize;
    use serde_json::Value;
    use std::path::Path;

    #[derive(Deserialize)]
    struct FixtureResources {
        estimated_model_bytes: u64,
    }

    #[derive(Deserialize)]
    struct FixtureModel {
        family: String,
        manifest_index: usize,
        resources: FixtureResources,
    }

    #[derive(Deserialize)]
    struct FixturePlan {
        selected_models: Vec<FixtureModel>,
        shards: Value,
        github_matrix: Value,
    }

    fn fixture(name: &str) -> FixturePlan {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/family_evidence")
            .join(format!("{name}.stdout"));
        serde_json::from_slice(&std::fs::read(path).expect("frozen plan exists"))
            .expect("frozen plan is valid JSON")
    }

    fn weight(raw: &str) -> WorkBytes {
        WorkBytes::parse(Some(&document::parse(raw).expect("integer")), "weight").expect("positive")
    }

    fn assert_projection(name: &str, count: usize) {
        let plan = fixture(name);
        let families = plan
            .selected_models
            .iter()
            .map(|model| SelectedFamily {
                family: &model.family,
                manifest_index: model.manifest_index,
                estimated_model_bytes: weight(&model.resources.estimated_model_bytes.to_string()),
            })
            .collect::<Vec<_>>();

        let projection = shard_families(&families, count).expect("positive shard count");

        for (actual, expected) in [
            (projection.shards.to_json(), plan.shards),
            (projection.github_matrix.to_json(), plan.github_matrix),
        ] {
            let encoded = output::compact(&actual).expect("projection");
            let actual: Value = serde_json::from_str(&encoded).expect("small fixture numbers");
            assert_eq!(actual, expected, "fixture {name}");
        }
    }

    #[test]
    fn single_shard_matches_frozen_manifest_order_and_total() {
        assert_projection("real-1", 1);
    }

    #[test]
    fn four_shards_match_frozen_balanced_assignment() {
        assert_projection("real-4", 4);
    }

    #[test]
    fn capped_shards_match_frozen_family_first_matrix() {
        assert_projection("real-256", 256);
    }

    #[test]
    fn reversed_family_filter_matches_frozen_manifest_order() {
        assert_projection("real-reversed", 1);
    }

    #[test]
    fn equal_weights_match_frozen_alphabetic_assignment() {
        assert_projection("synthetic-equal", 2);
    }

    #[test]
    fn uneven_weights_match_frozen_matrix_array_order() {
        assert_projection("synthetic-uneven", 3);
    }

    #[test]
    fn zero_shard_count_rejects_without_projection() {
        let plan = fixture("real-1");
        let families = plan
            .selected_models
            .iter()
            .map(|model| SelectedFamily {
                family: &model.family,
                manifest_index: model.manifest_index,
                estimated_model_bytes: weight(&model.resources.estimated_model_bytes.to_string()),
            })
            .collect::<Vec<_>>();

        let error = shard_families(&families, 0).expect_err("zero is invalid");

        assert_eq!(error.to_string(), "--shard-count must be positive");
    }

    #[test]
    fn repeated_projection_is_deterministic_without_mutating_input() {
        let plan = fixture("synthetic-uneven");
        let families = plan
            .selected_models
            .iter()
            .map(|model| SelectedFamily {
                family: &model.family,
                manifest_index: model.manifest_index,
                estimated_model_bytes: weight(&model.resources.estimated_model_bytes.to_string()),
            })
            .collect::<Vec<_>>();
        let original = families
            .iter()
            .map(|model| model.family)
            .collect::<Vec<_>>();

        let first = shard_families(&families, 3).expect("valid request");
        let second = shard_families(&families, 3).expect("valid request");

        assert_eq!(first, second);
        assert_eq!(
            families
                .iter()
                .map(|model| model.family)
                .collect::<Vec<_>>(),
            original
        );
    }

    #[test]
    fn total_is_exact_when_multiple_u64_inputs_exceed_u64_maximum() {
        let families = [
            SelectedFamily {
                family: "large",
                manifest_index: 0,
                estimated_model_bytes: weight("18446744073709551615"),
            },
            SelectedFamily {
                family: "small",
                manifest_index: 1,
                estimated_model_bytes: weight("1"),
            },
        ];

        let result = shard_families(&families, 1).expect("unbounded sum");

        assert_eq!(
            output::compact(&result.github_matrix.to_json()).expect("JSON"),
            r#"{"include":[{"id":"family-battery-01","shard_index":0,"families":"large,small","estimated_work_bytes":18446744073709551616}]}"#
        );
    }
}

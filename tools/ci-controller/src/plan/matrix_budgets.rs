//! Per-matrix `max-parallel` allowances. `<os>_max_parallel` limits one lane
//! matrix; `<os>_max_workers` caps the summed allowances of every set of
//! selected lane matrices that can run at the same time in one plan, where
//! matrices ordered by lane `needs` never overlap. Lanes consume only the
//! emitted allowance, so adding a matrix cannot multiply a run's workers.
//! Row selection is never changed: a plan that cannot fit fails instead.

use crate::plan::diagnostics::{PlanResult, fail};
use crate::plan::request::Profile;
use serde_json::{Map, Value, json};

const SWIFT_FULL_TARGETS: u64 = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Platform {
    Linux,
    Macos,
    Windows,
}

enum Rows {
    All(&'static str),
    OnPlatform(&'static str, &'static str),
    Row(&'static str, &'static str),
    SwiftTargets,
}

struct LaneMatrix {
    key: &'static str,
    platform: Platform,
    after: &'static [&'static str],
    /// A literal `max_parallel: 1` lane call: counted, never emitted.
    singleton: bool,
    rows: Rows,
}

use Platform::{Linux, Macos, Windows};

const fn matrix(key: &'static str, platform: Platform, rows: Rows) -> LaneMatrix {
    LaneMatrix {
        key,
        platform,
        after: &[],
        singleton: false,
        rows,
    }
}

const fn linux_backend(
    key: &'static str,
    after: &'static [&'static str],
    id: &'static str,
) -> LaneMatrix {
    LaneMatrix {
        key,
        platform: Linux,
        after,
        singleton: true,
        rows: Rows::Row("runtime_products", id),
    }
}

const LANE_MATRICES: &[LaneMatrix] = &[
    matrix("linux_clippy", Linux, Rows::All("clippy")),
    matrix("linux_rust_tests", Linux, Rows::All("rust_tests")),
    matrix("linux_hosts", Linux, Rows::OnPlatform("hosts", "linux")),
    linux_backend("linux_runtime_cpu", &[], "linux-cpu"),
    linux_backend("linux_runtime_cuda", &[], "linux-cuda"),
    linux_backend("linux_runtime_rocm", &[], "linux-rocm"),
    linux_backend("linux_runtime_vulkan", &[], "linux-vulkan"),
    linux_backend(
        "linux_product_cpu",
        &["linux_hosts", "linux_runtime_cpu"],
        "linux-cpu",
    ),
    linux_backend(
        "linux_product_cuda",
        &["linux_hosts", "linux_runtime_cuda"],
        "linux-cuda",
    ),
    linux_backend(
        "linux_product_rocm",
        &["linux_hosts", "linux_runtime_rocm"],
        "linux-rocm",
    ),
    linux_backend(
        "linux_product_vulkan",
        &["linux_hosts", "linux_runtime_vulkan"],
        "linux-vulkan",
    ),
    matrix("macos_hosts", Macos, Rows::OnPlatform("hosts", "macos")),
    matrix(
        "macos_runtimes",
        Macos,
        Rows::OnPlatform("runtime_products", "macos"),
    ),
    LaneMatrix {
        after: &["macos_hosts", "macos_runtimes"],
        ..matrix(
            "macos_products",
            Macos,
            Rows::OnPlatform("runtime_products", "macos"),
        )
    },
    matrix(
        "macos_platform_checks",
        Macos,
        Rows::OnPlatform("platform_checks", "macos"),
    ),
    matrix("macos_swift_sdk", Macos, Rows::SwiftTargets),
    matrix(
        "windows_hosts",
        Windows,
        Rows::OnPlatform("hosts", "windows"),
    ),
    matrix(
        "windows_runtimes",
        Windows,
        Rows::OnPlatform("runtime_products", "windows"),
    ),
    LaneMatrix {
        after: &["windows_hosts", "windows_runtimes"],
        ..matrix(
            "windows_products",
            Windows,
            Rows::OnPlatform("runtime_products", "windows"),
        )
    },
    matrix(
        "windows_platform_checks",
        Windows,
        Rows::OnPlatform("platform_checks", "windows"),
    ),
];

fn demand(source: &Rows, matrices: &Map<String, Value>, profile: Profile) -> u64 {
    let matrix = |name: &str| {
        matrices
            .get(name)
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
    };
    let count = |rows: usize| u64::try_from(rows).unwrap_or(u64::MAX);
    match source {
        Rows::All(name) => count(matrix(name).len()),
        Rows::OnPlatform(name, platform) => count(
            matrix(name)
                .iter()
                .filter(|row| row["platform"] == *platform)
                .count(),
        ),
        Rows::Row(name, id) => count(matrix(name).iter().filter(|row| row["id"] == *id).count()),
        // Ready/draft PRs build the Swift SDK in host-only mode, without the matrix.
        Rows::SwiftTargets => {
            let swift = matrix("sdk").iter().any(|row| row["id"] == "swift");
            if swift && profile.exhaustive() {
                SWIFT_FULL_TARGETS
            } else {
                0
            }
        }
    }
}

fn ordered(lane: &[&LaneMatrix], first: usize, second: usize) -> bool {
    lane[first].after.contains(&lane[second].key) || lane[second].after.contains(&lane[first].key)
}

/// The largest summed allowance of selected matrices that can all overlap,
/// found exhaustively over the (at most eleven) matrices of one platform.
fn widest_overlap(lane: &[&LaneMatrix], allowance: &[u64]) -> (u64, Vec<usize>) {
    let selected = (0..lane.len())
        .filter(|&index| allowance[index] > 0)
        .collect::<Vec<_>>();
    let mut widest = (0, Vec::new());
    for mask in 1_u32..(1 << selected.len()) {
        let members = selected
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, index)| *index)
            .collect::<Vec<_>>();
        let overlapping = members.iter().enumerate().all(|(at, first)| {
            members[at + 1..]
                .iter()
                .all(|second| !ordered(lane, *first, *second))
        });
        let total = members.iter().map(|index| allowance[*index]).sum::<u64>();
        if overlapping && total > widest.0 {
            widest = (total, members);
        }
    }
    widest
}

fn platform_allowances(
    lane: &[&LaneMatrix],
    matrices: &Map<String, Value>,
    profile: Profile,
    (matrix_limit, workers): (u64, u64),
) -> PlanResult<Vec<u64>> {
    let mut allowance = lane
        .iter()
        .map(|entry| {
            let cap = if entry.singleton { 1 } else { matrix_limit };
            demand(&entry.rows, matrices, profile).min(cap)
        })
        .collect::<Vec<_>>();
    loop {
        let (total, members) = widest_overlap(lane, &allowance);
        if total <= workers {
            return Ok(allowance);
        }
        let reducible = members
            .iter()
            .copied()
            .filter(|&index| !lane[index].singleton && allowance[index] > 1)
            .max_by_key(|&index| (allowance[index], std::cmp::Reverse(index)));
        let Some(index) = reducible else {
            return fail(format!(
                "profile {} worker budget {workers} cannot run {} overlapping selected matrices",
                profile.name(),
                members.len()
            ));
        };
        allowance[index] -= 1;
    }
}

/// `budgets.matrix_max_parallel`. Limits and worker ceilings are in Linux,
/// macOS, Windows order. An unselected matrix is skipped by its lane and
/// reports the minimum valid `max-parallel` of 1.
pub(super) fn allowances(
    matrices: &Map<String, Value>,
    profile: Profile,
    matrix_limits: [u64; 3],
    workers: [u64; 3],
) -> PlanResult<Map<String, Value>> {
    let mut emitted = Map::new();
    let platforms = [Linux, Macos, Windows].into_iter().zip(matrix_limits);
    for ((platform, limit), ceiling) in platforms.zip(workers) {
        let lane = LANE_MATRICES
            .iter()
            .filter(|entry| entry.platform == platform)
            .collect::<Vec<_>>();
        let allowance = platform_allowances(&lane, matrices, profile, (limit, ceiling))?;
        for (entry, value) in lane.iter().zip(allowance) {
            if !entry.singleton {
                emitted.insert(entry.key.to_owned(), json!(value.max(1)));
            }
        }
    }
    Ok(emitted)
}

/// The summed allowance of the widest overlapping set of selected matrices
/// per platform, in Linux, macOS, Windows order, for plan-level fixtures.
pub fn peak_workers(plan: &Value) -> [u64; 3] {
    let empty = Map::new();
    let matrices = plan["matrices"].as_object().unwrap_or(&empty);
    let profile = plan["profile"]
        .as_str()
        .and_then(Profile::parse)
        .unwrap_or(Profile::Main);
    let emitted = &plan["budgets"]["matrix_max_parallel"];
    [Linux, Macos, Windows].map(|platform| {
        let lane = LANE_MATRICES
            .iter()
            .filter(|entry| entry.platform == platform)
            .collect::<Vec<_>>();
        let allowance = lane
            .iter()
            .map(|entry| {
                let rows = demand(&entry.rows, matrices, profile);
                let cap = emitted[entry.key].as_u64().unwrap_or(1);
                rows.min(cap)
            })
            .collect::<Vec<_>>();
        widest_overlap(&lane, &allowance).0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows_of(platform: &str, count: usize) -> Value {
        Value::Array(
            (0..count)
                .map(|index| json!({"id": format!("{platform}-{index}"), "platform": platform}))
                .collect(),
        )
    }

    fn windows_rows() -> Map<String, Value> {
        let mut matrices = Map::new();
        matrices.insert("hosts".into(), rows_of("windows", 1));
        matrices.insert("runtime_products".into(), rows_of("windows", 4));
        matrices.insert("platform_checks".into(), rows_of("windows", 4));
        matrices
    }

    #[test]
    fn overlapping_matrices_share_one_worker_ceiling() {
        let emitted =
            allowances(&windows_rows(), Profile::Main, [12, 4, 4], [12, 8, 5]).expect("fits");
        let get = |key: &str| emitted[key].as_u64().expect("allowance");
        assert!(
            get("windows_hosts") + get("windows_runtimes") + get("windows_platform_checks") <= 5
        );
        assert!(get("windows_products") + get("windows_platform_checks") <= 5);
        assert_eq!(
            [
                get("windows_hosts"),
                get("windows_runtimes"),
                get("windows_products"),
                get("windows_platform_checks")
            ],
            [1, 2, 3, 2]
        );
    }

    #[test]
    fn matrix_limits_apply_when_the_ceiling_has_room() {
        let emitted =
            allowances(&windows_rows(), Profile::Main, [12, 4, 2], [12, 8, 5]).expect("fits");
        assert_eq!(emitted["windows_runtimes"], 2);
        assert_eq!(emitted["windows_products"], 2);
        assert_eq!(emitted["macos_swift_sdk"], 1);
    }

    #[test]
    fn a_ceiling_below_the_overlapping_matrix_count_refuses_the_plan() {
        let error = allowances(&windows_rows(), Profile::PrReady, [7, 2, 1], [12, 4, 2])
            .expect_err("three overlapping matrices exceed two workers");
        assert!(error.0.contains("cannot run 3 overlapping"), "{}", error.0);
    }
}

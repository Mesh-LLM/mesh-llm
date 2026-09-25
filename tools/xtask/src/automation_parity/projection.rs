//! The Rust statement of what `.github/actions/plan-ci/action.yml` derives
//! from a plan: every `$GITHUB_OUTPUT` line, including the digest and the
//! five lane projections, rendered the way `jq -c`/`jq -r` render them.

use super::jq_render::{compact, object, raw};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SIGNALS: [&str; 9] = [
    "rust_changed",
    "ui_changed",
    "website_changed",
    "website_docs_changed",
    "plugin_exemplars_changed",
    "cli_surface_changed",
    "docs_only",
    "backend_changed",
    "runner_contract_required",
];
const BUDGETS: [&str; 4] = [
    "linux_max_parallel",
    "macos_max_parallel",
    "windows_max_parallel",
    "total_max_workers",
];
const PLATFORMS: [&str; 3] = ["linux", "macos", "windows"];

/// Where a lane projection places `required`, and what decides it.
#[derive(Clone, Copy)]
enum Required {
    /// Given before the common fields.
    Leading(bool),
    /// Appended last: any nonempty matrix, or `also`.
    Trailing { also: bool },
}

struct Plan<'a>(&'a Value);

impl Plan<'_> {
    fn rows(&self, matrix: &str) -> Result<&[Value], String> {
        self.0["matrices"][matrix]
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| format!("matrices.{matrix} is not an array"))
    }

    fn select(&self, matrix: &str, keep: impl Fn(&Value) -> bool) -> Result<Vec<Value>, String> {
        Ok(self
            .rows(matrix)?
            .iter()
            .filter(|row| keep(row))
            .cloned()
            .collect())
    }

    fn on(&self, matrix: &str, platform: &str) -> Result<Vec<Value>, String> {
        self.select(matrix, |row| row["platform"] == platform)
    }

    fn slices(&self) -> Vec<&str> {
        self.0["required_slices"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect()
    }

    /// `profile, domains, required_slices, signals, budgets` in that order.
    fn common(&self) -> Vec<(&'static str, String)> {
        [
            "profile",
            "domains",
            "required_slices",
            "signals",
            "budgets",
        ]
        .into_iter()
        .map(|key| (key, compact(&self.0[key])))
        .collect()
    }

    fn lane(&self, lane: &str, required: Required, matrices: &[(&str, Vec<Value>)]) -> String {
        let mut pairs = vec![("lane", compact(&Value::from(lane)))];
        if let Required::Leading(required) = required {
            pairs.push(("required", required.to_string()));
        }
        pairs.extend(self.common());
        let rows = matrices
            .iter()
            .map(|(name, rows)| (*name, compact(&Value::Array(rows.clone()))))
            .collect::<Vec<_>>();
        pairs.push(("matrices", object(&rows)));
        if let Required::Trailing { also } = required {
            let nonempty = matrices.iter().any(|(_, rows)| !rows.is_empty());
            pairs.push(("required", (nonempty || also).to_string()));
        }
        object(&pairs)
    }

    fn quality(&self) -> Result<String, String> {
        let slices = self.slices();
        let required = slices.contains(&"quality") || slices.contains(&"runner-contract");
        let clippy = self.rows("clippy")?.to_vec();
        Ok(self.lane(
            "quality",
            Required::Leading(required),
            &[("clippy", clippy)],
        ))
    }

    fn website(&self) -> String {
        self.lane(
            "website",
            Required::Leading(self.slices().contains(&"web")),
            &[],
        )
    }

    fn linux(&self) -> Result<String, String> {
        let matrices = [
            ("rust_tests", self.rows("rust_tests")?.to_vec()),
            ("hosts", self.on("hosts", "linux")?),
            ("runtime_products", self.on("runtime_products", "linux")?),
            (
                "smoke",
                self.select("smoke", |row| row["id"] != "metal-model-load")?,
            ),
            ("sdk", self.on("sdk", "linux")?),
        ];
        // Linux is also required by a selected static-abi slice alone.
        let also = self.slices().contains(&"static-abi");
        Ok(self.lane("linux", Required::Trailing { also }, &matrices))
    }

    fn macos(&self) -> Result<String, String> {
        let matrices = [
            ("hosts", self.on("hosts", "macos")?),
            ("runtime_products", self.on("runtime_products", "macos")?),
            ("platform_checks", self.on("platform_checks", "macos")?),
            (
                "smoke",
                self.select("smoke", |row| row["id"] == "metal-model-load")?,
            ),
            ("sdk", self.on("sdk", "macos")?),
        ];
        Ok(self.lane("macos", Required::Trailing { also: false }, &matrices))
    }

    fn windows(&self) -> Result<String, String> {
        let matrices = [
            ("hosts", self.on("hosts", "windows")?),
            ("runtime_products", self.on("runtime_products", "windows")?),
            ("platform_checks", self.on("platform_checks", "windows")?),
        ];
        Ok(self.lane("windows", Required::Trailing { also: false }, &matrices))
    }
}

/// The `$GITHUB_OUTPUT` text the action writes for this planner stdout.
pub(super) fn action_outputs(stdout: &[u8]) -> Result<String, String> {
    let value: Value = serde_json::from_slice(stdout).map_err(|error| error.to_string())?;
    if value["schema_version"] != 1 {
        return Err("schema_version is not 1".to_owned());
    }
    let plan = Plan(&value);
    let plan_json = compact(&value);
    let digest = hex::encode(Sha256::digest(plan_json.as_bytes()));
    let mut lines = vec![
        "plan_path=ci-plan.json".to_owned(),
        format!("plan_json={plan_json}"),
        format!("plan_digest={digest}"),
        format!("profile={}", raw(&value["profile"])),
    ];
    for key in ["required_slices", "domains", "affected_crates"] {
        lines.push(format!("{key}={}", compact(&value[key])));
    }
    let whole = [
        "clippy",
        "rust_tests",
        "hosts",
        "runtime_products",
        "platform_checks",
        "smoke",
        "sdk",
    ];
    for matrix in whole {
        lines.push(format!(
            "{matrix}_matrix={}",
            compact(&Value::Array(plan.rows(matrix)?.to_vec()))
        ));
    }
    for matrix in ["hosts", "runtime_products"] {
        for platform in PLATFORMS {
            let rows = Value::Array(plan.on(matrix, platform)?);
            lines.push(format!("{platform}_{matrix}_matrix={}", compact(&rows)));
        }
    }
    lines.extend(
        SIGNALS
            .iter()
            .map(|name| format!("{name}={}", raw(&value["signals"][name]))),
    );
    lines.extend(
        BUDGETS
            .iter()
            .map(|name| format!("{name}={}", raw(&value["budgets"][name]))),
    );
    lines.push(format!("quality_lane_plan={}", plan.quality()?));
    lines.push(format!("website_lane_plan={}", plan.website()));
    lines.push(format!("linux_lane_plan={}", plan.linux()?));
    lines.push(format!("macos_lane_plan={}", plan.macos()?));
    lines.push(format!("windows_lane_plan={}", plan.windows()?));
    Ok(lines.join("\n") + "\n")
}

/// Compares two output documents and names the first differing output.
pub(super) fn outputs_difference(expected: &str, actual: &str) -> Option<String> {
    if expected == actual {
        return None;
    }
    let name = |line: &str| {
        line.split_once('=')
            .map_or(line, |(name, _)| name)
            .to_owned()
    };
    let mismatch = expected
        .lines()
        .zip(actual.lines())
        .find(|(left, right)| left != right)
        .map(|(left, _)| name(left));
    let first = super::process::first_difference(expected.as_bytes(), actual.as_bytes());
    Some(format!(
        "output {}: {}",
        mismatch.unwrap_or_else(|| "<line count>".to_owned()),
        first.unwrap_or_default()
    ))
}

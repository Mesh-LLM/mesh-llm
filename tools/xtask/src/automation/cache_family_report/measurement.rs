use super::input::Row;
use crate::command::DynResult;

pub(super) fn hit(row: &Row) -> Option<f64> {
    if let (Some(imports), Some(decodes)) = (
        &row.skippy.cache_hit_import_ms,
        &row.skippy.cache_hit_decode_ms,
    ) {
        let mut values = imports
            .iter()
            .zip(decodes)
            .filter_map(|(a, b)| Some(a.as_ref()? + b.as_ref()?))
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        return match values.len() {
            0 => None,
            length if length % 2 == 1 => Some(values[middle]),
            _ => Some(values[middle - 1] / 2.0 + values[middle] / 2.0),
        };
    }
    row.skippy.cache_hit_total_ms
}

pub(super) struct Baseline {
    pub value: f64,
    pub label: &'static str,
}

pub(super) fn baseline(row: &Row, package: bool) -> Option<Baseline> {
    if row.llama_server.status.as_deref() == Some("ok") {
        if let Some(value) = row.llama_server.warm_median_ms {
            return Some(Baseline {
                value,
                label: "llama-server warm median",
            });
        }
        if let Some(value) = row.llama_server.warm_mean_ms {
            return Some(Baseline {
                value,
                label: "llama-server warm mean",
            });
        }
    }
    if package {
        row.skippy.recompute_total_ms.map(|value| Baseline {
            value,
            label: "Skippy stage recompute",
        })
    } else {
        None
    }
}

pub(super) fn storage(row: &Row) -> DynResult<(Option<u64>, &'static str)> {
    if let Some(value) = row.skippy.cache_storage_bytes {
        return Ok((Some(value), "measured"));
    }
    if row.payload.as_deref() == Some("resident-kv")
        && let (Some(bytes), Some(tokens)) =
            (row.case.resident_kv_bytes_per_token, row.prefix_tokens)
    {
        return Ok((
            Some(
                bytes
                    .checked_mul(tokens)
                    .ok_or("cache storage byte count overflow")?,
            ),
            "metadata-derived",
        ));
    }
    Ok((None, "n/a"))
}

pub(super) fn ms(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".into(), |v| format!("{v:.1}"))
}

pub(super) fn bytes(value: Option<u64>) -> String {
    match value {
        None => "n/a".into(),
        Some(0) => "0".into(),
        Some(value) if value < 1024 * 1024 => format!("{:.1} KiB", value as f64 / 1024.0),
        Some(value) => format!("{:.1} MiB", value as f64 / (1024.0 * 1024.0)),
    }
}

pub(super) fn win(row: &Row, package: bool, bold: bool) -> String {
    // A displayed failed correctness row must never become a performance claim.
    if row.skippy.status.as_deref() != Some("pass") {
        return "n/a".into();
    }
    let value = baseline(row, package)
        .zip(hit(row))
        .filter(|(_, hit)| *hit > 0.0)
        .map(|(base, hit)| base.value / hit)
        .filter(|v| v.is_finite());
    value.map_or_else(
        || "n/a".into(),
        |value| {
            if bold {
                format!("**{value:.2}x faster**")
            } else {
                format!("{value:.2}x")
            }
        },
    )
}

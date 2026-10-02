use super::{
    input::{Corpus, FAMILIES, Row, USE_CASES, order},
    measurement as m,
};
use crate::command::DynResult;
use std::collections::BTreeMap;

fn cell(text: &str) -> String {
    text.replace('|', "/")
        .replace(['\n', '\r'], " ")
        .replace('`', "'")
}
fn number(value: Option<u64>) -> String {
    value.map_or_else(|| "n/a".into(), |v| v.to_string())
}
fn payload(row: &Row) -> String {
    cell(match row.payload.as_deref() {
        Some("resident-kv") => "ResidentKv",
        Some("kv-recurrent") => "KvRecurrent",
        Some("full-state") => "FullState",
        Some("recurrent-only") => "RecurrentOnly",
        Some(value) => value,
        None => "n/a",
    })
}

fn table(rows: &[&Row], package: bool) -> DynResult<String> {
    let base_title = if package {
        "Baseline"
    } else {
        "llama-server warm median ms"
    };
    let base_align = if package { "---" } else { "---:" };
    let mut lines = vec![
        format!(
            "| Family | Representative model ref | Production payload | Correctness | Prefix tokens | Prompt tokens | {base_title} | Skippy hit median ms | Skippy win | Cache bytes | Size method | Notes |"
        ),
        format!(
            "| --- | --- | --- | --- | ---: | ---: | {base_align} | ---: | ---: | ---: | --- | --- |"
        ),
    ];
    for row in rows {
        let (storage, method) = m::storage(row)?;
        let baseline = if package {
            m::baseline(row, true).map_or_else(|| "n/a".into(), |base| base.label.into())
        } else {
            m::ms(m::baseline(row, false).map(|base| base.value))
        };
        lines.push(format!(
            "| {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            cell(&row.family),
            cell(&row.model_id),
            payload(row),
            cell(row.skippy.status.as_deref().unwrap_or("missing")),
            number(row.prefix_tokens),
            number(row.benchmark_prompt_token_count),
            baseline,
            m::ms(m::hit(row)),
            m::win(row, package, true),
            m::bytes(storage),
            method,
            cell(&row.notes)
        ));
    }
    Ok(lines.join("\n"))
}

fn matrix(rows: &[&Row]) -> String {
    let mut families = rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<Vec<_>>();
    families.sort_by(|a, b| order(a, FAMILIES).cmp(&order(b, FAMILIES)));
    families.dedup();
    let mut cases = rows
        .iter()
        .filter_map(|row| row.use_case.as_deref())
        .collect::<Vec<_>>();
    cases.sort_by(|a, b| order(a, USE_CASES).cmp(&order(b, USE_CASES)));
    cases.dedup();
    let mut by_key = BTreeMap::new();
    let mut labels = BTreeMap::new();
    for row in rows {
        let key = row.use_case.as_deref().unwrap_or("");
        by_key.insert((key, row.family.as_str()), *row);
        labels.insert(key, row.use_case_label.as_deref().unwrap_or(key));
    }
    let mut lines = vec![
        format!(
            "| Use case | {} |",
            families
                .iter()
                .map(|v| cell(v))
                .collect::<Vec<_>>()
                .join(" | ")
        ),
        format!("| --- | {} |", vec!["---:"; families.len()].join(" | ")),
    ];
    for key in cases {
        let mut cells = vec![cell(labels.get(key).copied().unwrap_or(key))];
        cells.extend(families.iter().map(|family| {
            by_key
                .get(&(key, *family))
                .map_or_else(|| "n/a".into(), |row| m::win(row, false, false))
        }));
        lines.push(format!("| {} |", cells.join(" | ")));
    }
    lines.join("\n")
}

fn sources(corpus: &Corpus) -> String {
    let mut rows = corpus.use_cases.iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| order(&a.key, USE_CASES).cmp(&order(&b.key, USE_CASES)));
    let mut lines = vec![
        "| Use case | Dataset | Config | Split | Row |".into(),
        "| --- | --- | --- | --- | ---: |".into(),
    ];
    for row in rows {
        let source = &row.source;
        lines.push(format!(
            "| {} | `{}` | `{}` | `{}` | {} |",
            cell(row.label.as_deref().unwrap_or(&row.key)),
            cell(&source.dataset),
            cell(&source.config),
            cell(&source.split),
            source.row_idx.map_or_else(String::new, |v| v.to_string())
        ));
    }
    lines.join("\n")
}

pub(super) fn report(rows: &[Row], corpus: &Corpus) -> DynResult<String> {
    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| order(&a.family, FAMILIES).cmp(&order(&b.family, FAMILIES)));
    let has_case = |row: &&Row| row.use_case.as_deref().is_some_and(|s| !s.is_empty());
    let baseline_ok = |row: &&Row| {
        row.stage_load_mode == "runtime-slice" && row.llama_server.status.as_deref() == Some("ok")
    };
    let full = ordered
        .iter()
        .copied()
        .filter(|row| !has_case(row) && baseline_ok(row))
        .collect::<Vec<_>>();
    let package = ordered
        .iter()
        .copied()
        .filter(|row| !has_case(row) && row.stage_load_mode != "runtime-slice")
        .collect::<Vec<_>>();
    let use_cases = ordered
        .iter()
        .copied()
        .filter(|row| has_case(row) && baseline_ok(row))
        .collect::<Vec<_>>();
    Ok(vec![
        "### Full-GGUF llama-server vs Skippy".into(),
        String::new(),
        "Rows are ordered so related runtime/cache families appear next to each other.".into(),
        String::new(),
        table(&full, false)?,
        String::new(),
        "### Use-Case Benchmark Matrix".into(),
        String::new(),
        "This matrix uses one Hugging Face-sourced representative prompt per use case,".into(),
        "the same requested prefix tokens, one generated token, Skippy".into(),
        "`--runtime-lane-count 1`, llama-server `--parallel 1`, and the same full-GGUF".into(),
        "family set as the table above. Values are Skippy warm-hit latency speedup over".into(),
        "llama-server warm-cache latency. DeepSeek3 stays in the package-only section".into(),
        "because there is no practical local full-GGUF llama-server baseline for that".into(),
        "artifact.".into(),
        String::new(),
        matrix(&use_cases),
        String::new(),
        "Prompt sources are checked in at `evals/skippy-usecase-corpus.json` with source".into(),
        "dataset metadata:".into(),
        String::new(),
        sources(corpus),
        String::new(),
        "### Package-Only Giant Models".into(),
        String::new(),
        "These rows validate cache strategy for models where a full llama-server".into(),
        "baseline is not operationally useful because monolithic residency is too large.".into(),
        String::new(),
        table(&package, true)?,
        String::new(),
    ]
    .join("\n"))
}

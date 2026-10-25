use super::report_escape::xml;
use super::report_input::Build;
use super::report_row::Row;
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::io::Write;

pub(super) enum Metric {
    Decode,
    Output,
    Ttft,
}

impl Metric {
    pub(super) fn artifact(&self) -> &'static str {
        match self {
            Self::Decode => "decode-throughput.svg",
            Self::Output => "workload-output-throughput.svg",
            Self::Ttft => "ttft-p50.svg",
        }
    }

    fn labels(&self) -> (&'static str, &'static str) {
        match self {
            Self::Decode => (
                "Decode throughput by arm",
                "Generated tokens / decode second",
            ),
            Self::Output => (
                "End-to-end workload output throughput by arm",
                "Generated tokens / wall-clock second",
            ),
            Self::Ttft => (
                "Median time to first token by arm",
                "Seconds (lower is better)",
            ),
        }
    }

    fn value(&self, row: &Row) -> Option<f64> {
        match self {
            Self::Decode => row.decode_tokens_per_second,
            Self::Output => row.workload_output_tokens_per_second,
            Self::Ttft => row.ttft_p50_seconds,
        }
    }
}

pub(super) fn render(metric: &Metric, data: (&[Row], &[Build])) -> DynResult<Vec<u8>> {
    let (rows, builds) = data;
    let concurrency = rows
        .iter()
        .map(|row| row.concurrency)
        .collect::<BTreeSet<_>>();
    let maximum = rows
        .iter()
        .filter_map(|row| metric.value(row))
        .max_by(f64::total_cmp);
    let ceiling = maximum.map_or(1.0, |value| (value * 1.12).max(1e-9));
    let denominator = f64::from(u32::try_from(concurrency.len().saturating_sub(1).max(1))?);
    let mut output = Vec::new();
    let (title, axis) = metric.labels();
    write!(
        output,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"540\" viewBox=\"0 0 960 540\"><rect width=\"100%\" height=\"100%\" fill=\"#fff\"/><text x=\"480\" y=\"36\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"22\" font-weight=\"700\">{}</text>",
        xml(title)
    )?;
    for tick in 0..6 {
        let ordinate = 430.0 - 350.0 * f64::from(tick) / 5.0;
        let value = ceiling * f64::from(tick) / 5.0;
        write!(
            output,
            "<line x1=\"100\" y1=\"{ordinate:.1}\" x2=\"890\" y2=\"{ordinate:.1}\" stroke=\"#e2e8f0\"/><text x=\"90\" y=\"{:.1}\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">{value:.1}</text>",
            ordinate + 4.0
        )?;
    }
    for (index, value) in concurrency.iter().enumerate() {
        let abscissa = 100.0 + 790.0 * f64::from(u32::try_from(index)?) / denominator;
        write!(
            output,
            "<text x=\"{abscissa:.1}\" y=\"454\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">{value}</text>"
        )?;
    }
    const COLORS: &[&str] = &[
        "#0284c7", "#dc2626", "#16a34a", "#7c3aed", "#ea580c", "#0891b2",
    ];
    for (index, build) in builds.iter().enumerate() {
        let color = COLORS[index % COLORS.len()];
        let mut points = Vec::new();
        for (position, offered) in concurrency.iter().enumerate() {
            if let Some(value) = rows
                .iter()
                .rev()
                .find(|row| row.label == build.label && row.concurrency == *offered)
                .and_then(|row| metric.value(row))
            {
                points.push((
                    100.0 + 790.0 * f64::from(u32::try_from(position)?) / denominator,
                    430.0 - 350.0 * value / ceiling,
                ));
            }
        }
        let coordinates = points
            .iter()
            .map(|(horizontal, vertical)| format!("{horizontal:.1},{vertical:.1}"))
            .collect::<Vec<_>>()
            .join(" ");
        write!(
            output,
            "<polyline points=\"{coordinates}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"3\"/>"
        )?;
        for (horizontal, vertical) in points {
            write!(
                output,
                "<circle cx=\"{horizontal:.1}\" cy=\"{vertical:.1}\" r=\"4\" fill=\"{color}\"/>"
            )?;
        }
        let legend = 110_u64
            .checked_add(
                u64::try_from(index)?
                    .checked_mul(150)
                    .ok_or("legend overflow")?,
            )
            .ok_or("legend overflow")?;
        write!(
            output,
            "<text x=\"{legend}\" y=\"490\" font-family=\"sans-serif\" font-size=\"13\" fill=\"{color}\">{}</text>",
            xml(&build.label)
        )?;
    }
    write!(
        output,
        "<text x=\"480\" y=\"525\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\">Client concurrency</text><text x=\"22\" y=\"255\" text-anchor=\"middle\" transform=\"rotate(-90 22 255)\" font-family=\"sans-serif\" font-size=\"13\">{}</text></svg>",
        xml(axis)
    )?;
    Ok(output)
}

use super::document::Json;
use super::failure::Failure;
use super::plan::Plan;
use super::projection::ToJson;
use std::fs;
use std::io::Write;
use std::path::Path;

pub(super) fn write(
    plan: &Plan,
    destinations: (Option<&Path>, Option<&Path>),
) -> Result<String, Failure> {
    let (output, github) = destinations;
    let rendered = render(&plan.to_json(), 0).map_err(Failure::Runtime)? + "\n";
    let Some(path) = output else {
        return if github.is_some() {
            Err(Failure::MissingOutput { stdout: rendered })
        } else {
            Ok(rendered)
        };
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(Failure::io)?;
    }
    fs::write(path, rendered).map_err(Failure::io)?;
    if let Some(github) = github {
        let absolute = path.canonicalize().map_err(Failure::io)?;
        let matrix = compact(&plan.github_matrix.to_json()).map_err(Failure::Runtime)?;
        let lines = format!(
            "plan_path={}\nmanifest_sha256={}\nfamily_count={}\nmatrix={}\n",
            absolute.display(),
            plan.manifest_sha256,
            plan.selected_family_count,
            matrix
        );
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(github)
            .map_err(Failure::io)?
            .write_all(lines.as_bytes())
            .map_err(Failure::io)?;
    }
    Ok(String::new())
}

fn render(value: &Json, depth: usize) -> Result<String, String> {
    Ok(match value {
        Json::Null => "null".into(),
        Json::Bool(flag) => flag.to_string(),
        Json::Integer(number) => number.to_string(),
        Json::Float(number) => crate::prepared_input::value_format::float_repr(*number),
        Json::String(text) => {
            let mut output = String::new();
            text.write_json(&mut output);
            output
        }
        Json::Object(entries) if entries.is_empty() => "{}".into(),
        Json::Array(items) if items.is_empty() => "[]".into(),
        Json::Object(entries) => {
            let mut entries = entries.iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let items = entries
                .into_iter()
                .map(|(key, value)| {
                    let mut quoted = String::new();
                    key.write_json(&mut quoted);
                    Ok(format!(
                        "{}{}: {}",
                        "  ".repeat(depth + 1),
                        quoted,
                        render(value, depth + 1)?
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            format!("{{\n{}\n{}}}", items.join(",\n"), "  ".repeat(depth))
        }
        Json::Array(items) => {
            let items = items
                .iter()
                .map(|value| {
                    Ok(format!(
                        "{}{}",
                        "  ".repeat(depth + 1),
                        render(value, depth + 1)?
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            format!("[\n{}\n{}]", items.join(",\n"), "  ".repeat(depth))
        }
    })
}

pub(super) fn compact(value: &Json) -> Result<String, String> {
    Ok(match value {
        Json::Object(entries) => {
            let items = entries
                .iter()
                .map(|(key, value)| {
                    let mut quoted = String::new();
                    key.write_json(&mut quoted);
                    Ok(format!("{quoted}:{}", compact(value)?))
                })
                .collect::<Result<Vec<_>, String>>()?;
            format!("{{{}}}", items.join(","))
        }
        Json::Array(items) => {
            let items = items.iter().map(compact).collect::<Result<Vec<_>, _>>()?;
            format!("[{}]", items.join(","))
        }
        Json::Null | Json::Bool(_) | Json::Integer(_) | Json::Float(_) | Json::String(_) => {
            render(value, 0)?
        }
    })
}

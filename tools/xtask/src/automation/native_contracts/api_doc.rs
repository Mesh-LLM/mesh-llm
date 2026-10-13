mod render;

use crate::command::DynResult;
use std::path::PathBuf;

struct Function {
    name: String,
    declaration: String,
    brief: String,
}

struct Header {
    name: String,
    brief: String,
    declarations: Vec<String>,
    functions: Vec<Function>,
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn brief(comment: &str) -> String {
    let lines = comment
        .lines()
        .map(|line| {
            line.trim()
                .trim_end_matches("*/")
                .trim_start_matches('/')
                .trim_start_matches('*')
                .trim_end_matches("*/")
                .trim()
        })
        .collect::<Vec<_>>();
    let text = lines.join("\n");
    let Some((_, rest)) = text.split_once("@brief") else {
        return String::new();
    };
    let end = rest.find("\n@").unwrap_or(rest.len());
    normalize(&rest[..end])
}

fn parse(name: String, text: &str) -> DynResult<Header> {
    let comments: Vec<_> = text
        .match_indices("/**")
        .filter_map(|(start, _)| text[start..].find("*/").map(|end| (start, start + end + 2)))
        .collect();
    let file = comments
        .iter()
        .find(|(start, end)| text[*start..*end].contains("@file"))
        .ok_or("missing file documentation")?;
    let header_brief = brief(&text[file.0..file.1]);
    if header_brief.is_empty() {
        return Err("missing header @brief".into());
    }
    let mut exports: Vec<_> = ["LLAMA_API", "SKIPPY_COMMON_API"]
        .into_iter()
        .flat_map(|prefix| text.match_indices(prefix).map(|(start, _)| start))
        .collect();
    exports.sort_unstable();
    let mut functions = Vec::new();
    for start in exports {
        let Some(end) = text[start..].find(';') else {
            continue;
        };
        let declaration = normalize(&text[start..start + end + 1]);
        let Some(symbol_start) = declaration
            .match_indices("skippy_")
            .map(|(start, _)| start)
            .find(|start| {
                let rest = &declaration[*start..];
                let end = rest
                    .find(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                    .unwrap_or(rest.len());
                rest[end..].trim_start().starts_with('(')
            })
        else {
            continue;
        };
        let symbol = declaration[symbol_start..]
            .split('(')
            .next()
            .ok_or("missing function name")?
            .trim();
        if !symbol
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            continue;
        }
        let comment = comments
            .iter()
            .rev()
            .find(|(_, end)| *end <= start && text[*end..start].trim().is_empty())
            .ok_or_else(|| format!("missing function documentation: {symbol}"))?;
        let function_brief = brief(&text[comment.0..comment.1]);
        if function_brief.is_empty() {
            return Err(format!("missing @brief: {symbol}").into());
        }
        functions.push(Function {
            name: symbol.to_owned(),
            declaration,
            brief: function_brief,
        });
    }
    let mut declarations = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line
            .strip_prefix("struct ")
            .or_else(|| line.strip_prefix("enum "))
        {
            let symbol = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches(';');
            let tail = rest.strip_prefix(symbol).unwrap_or("").trim_start();
            if symbol.starts_with("skippy_")
                && (rest.trim_end().ends_with(';') || tail.starts_with('{'))
                && !declarations.iter().any(|item| item == symbol)
            {
                declarations.push(symbol.to_owned());
            }
        }
    }
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#define ") {
            let (symbol, value) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
            if symbol.starts_with("SKIPPY_")
                && !symbol.ends_with("_H")
                && symbol
                    .bytes()
                    .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
            {
                let item = if value.trim().is_empty() {
                    symbol.to_owned()
                } else {
                    format!("{symbol} = {}", value.trim())
                };
                if !declarations.contains(&item) {
                    declarations.push(item);
                }
            }
        }
    }
    Ok(Header {
        name,
        brief: header_brief,
        declarations,
        functions,
    })
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let values = super::options(args, &["--include-dir", "--output", "--check"])?;
    let include = values.get("--include-dir").map_or_else(
        || PathBuf::from(".deps/llama.cpp/include/skippy"),
        PathBuf::from,
    );
    let output = values.get("--output").map_or_else(
        || PathBuf::from("website/src/docs/pages/skippy-api.md"),
        PathBuf::from,
    );
    let mut paths = std::fs::read_dir(&include)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| path.extension().is_some_and(|extension| extension == "h"));
    paths.sort();
    if let Some(parent) = include.parent() {
        let umbrella = parent.join("skippy.h");
        if umbrella.is_file() {
            paths.insert(0, umbrella);
        }
    }
    let headers = paths
        .iter()
        .map(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("header name must be UTF-8")?;
            parse(name.to_owned(), &std::fs::read_to_string(path)?)
        })
        .collect::<DynResult<Vec<_>>>()?;
    super::publish(
        &output,
        render::render(&headers).as_bytes(),
        values.contains_key("--check"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_export_libraries_are_documented() {
        let source = "/** @file sample.h\n * @brief Sample.\n */\n/** @brief First. */\nLLAMA_API int skippy_first(void);\n/** @brief Second. */\nSKIPPY_COMMON_API int skippy_second(void);\n";
        let header = parse("sample.h".into(), source).unwrap();
        assert_eq!(
            header
                .functions
                .iter()
                .map(|function| function.name.as_str())
                .collect::<Vec<_>>(),
            ["skippy_first", "skippy_second"]
        );
    }

    #[test]
    fn undocumented_export_is_rejected() {
        let source = "/** @file sample.h\n * @brief Sample.\n */\nstruct skippy_handle;\nLLAMA_API int skippy_first(void);\n";
        assert!(parse("sample.h".into(), source).is_err());
    }
}

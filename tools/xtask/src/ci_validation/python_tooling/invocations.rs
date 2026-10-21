use crate::ci_validation::lane_results::workflow_yaml::{self, Node};
use crate::command::DynResult;
use std::fs;
use std::path::Path;
use syn::visit::Visit;
use syn::{Expr, Item, Meta};

const JAVASCRIPT_SOURCE_LIMIT: usize = 1024 * 1024;

#[derive(Clone)]
pub(super) struct HereDoc {
    marker: String,
    shell_program: bool,
}

pub(super) fn executable_source(
    path: &Path,
    absolute: &Path,
    bytes: &[u8],
) -> DynResult<Option<Vec<(String, String)>>> {
    if is_workflow_or_action_yaml(path) {
        return Ok(Some(yaml_run_lines(std::str::from_utf8(bytes)?)?));
    }
    if (is_shell_or_just_source(path) || is_executable_file(absolute))
        && let Ok(text) = std::str::from_utf8(bytes)
    {
        return Ok(Some(
            text.lines()
                .enumerate()
                .map(|(index, line)| (format!("line {}", index + 1), line.to_owned()))
                .collect(),
        ));
    }
    Ok(None)
}

fn yaml_run_lines(source: &str) -> DynResult<Vec<(String, String)>> {
    let document = workflow_yaml::parse_resolved_aliases(source)
        .map_err(|error| format!("could not parse workflow/action YAML: {error}"))?;
    let mut scripts = Vec::new();
    collect_run_scripts(&document, &mut scripts)?;
    Ok(scripts
        .into_iter()
        .enumerate()
        .flat_map(|(command, script)| {
            script
                .lines()
                .enumerate()
                .map(move |(line, text)| {
                    (
                        format!("run command {} line {}", command + 1, line + 1),
                        text.to_owned(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect())
}

fn collect_run_scripts(node: &Node, scripts: &mut Vec<String>) -> DynResult<()> {
    let Node::Map(entries) = node else {
        return Ok(());
    };
    for (key, value) in entries {
        match key.as_str() {
            "jobs" => collect_workflow_job_steps(value, scripts)?,
            "runs" => collect_composite_action_steps(value, scripts)?,
            _ => {}
        }
    }
    Ok(())
}

fn collect_workflow_job_steps(jobs: &Node, scripts: &mut Vec<String>) -> DynResult<()> {
    let Node::Map(jobs) = jobs else {
        return Err("workflow jobs value must be a mapping".into());
    };
    for (_, job) in jobs {
        let Node::Map(entries) = job else {
            continue;
        };
        for (key, value) in entries {
            if key == "steps" {
                collect_step_scripts(value, scripts)?;
            }
        }
    }
    Ok(())
}

fn collect_composite_action_steps(runs: &Node, scripts: &mut Vec<String>) -> DynResult<()> {
    let Node::Map(entries) = runs else {
        return Ok(());
    };
    for (key, value) in entries {
        if key == "steps" {
            collect_step_scripts(value, scripts)?;
        }
    }
    Ok(())
}

fn collect_step_scripts(steps: &Node, scripts: &mut Vec<String>) -> DynResult<()> {
    let Node::Seq(items) = steps else {
        return Err("workflow/action steps value must be a sequence".into());
    };
    for item in items {
        let Node::Map(entries) = item else {
            if matches!(item, Node::Scalar(value) if is_flow_mapping_run_field(value)) {
                return Err(
                    "flow-map workflow run fields are unsupported by the workflow YAML reader"
                        .into(),
                );
            }
            return Err("workflow/action step value must be a mapping".into());
        };
        for (key, value) in entries {
            if key.trim_start().starts_with('{') {
                return Err(
                    "flow-map workflow run fields are unsupported by the workflow YAML reader"
                        .into(),
                );
            }
            if key == "run" {
                let script = value
                    .text()
                    .ok_or("workflow/action step run value must be a scalar string")?;
                scripts.push(script.to_owned());
            }
        }
    }
    Ok(())
}

fn is_flow_mapping_run_field(value: &str) -> bool {
    let value = value.trim();
    value.starts_with('{') || value.starts_with("run:")
}

pub(super) fn has_python_tool_invocation(line: &str) -> bool {
    let mut heredoc = None;
    has_python_tool_invocation_in_line(line, &mut heredoc)
}

pub(super) fn has_python_tool_invocation_in_line(
    line: &str,
    heredoc: &mut Option<HereDoc>,
) -> bool {
    let normalized = line.trim();
    if let Some(document) = heredoc.as_ref() {
        if normalized == document.marker.as_str() {
            *heredoc = None;
            return false;
        }
        return document.shell_program && has_python_tool_invocation(line);
    }

    for segment in shell_segments(line) {
        let Some(first) = segment.first().map(String::as_str) else {
            continue;
        };
        let first = first.trim_start_matches(['!', '(']);
        if matches!(first, "echo" | "printf" | "printf%" | "cat") {
            continue;
        }
        if segment.iter().any(|word| is_python_tool_word(word)) {
            return true;
        }
    }

    if command_substitution_runs_python(line) {
        return true;
    }
    if let Some(marker) = heredoc_marker(line) {
        let segments = shell_segments(line);
        let command = segments
            .first()
            .and_then(|segment| segment.first())
            .map(|word| word.trim_start_matches(['!', '(']));
        let shell_program = command.is_some_and(is_shell_executable);
        *heredoc = Some(HereDoc {
            marker,
            shell_program,
        });
    }
    false
}

fn shell_segments(line: &str) -> Vec<Vec<String>> {
    let mut segments = Vec::new();
    let mut words = Vec::new();
    let mut token = String::new();
    let mut quote = None;
    let mut escaped = false;
    let mut chars = line.chars().peekable();
    while let Some(character) = chars.next() {
        if escaped {
            token.push(character);
            escaped = false;
            continue;
        }
        if character == '\\' && quote != Some('\'') {
            escaped = true;
            continue;
        }
        if let Some(open) = quote {
            if character == open {
                quote = None;
            } else {
                token.push(character);
            }
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            '#' if token.is_empty() => break,
            ';' | '|' | '&' => {
                if !token.is_empty() {
                    words.push(std::mem::take(&mut token));
                }
                if !words.is_empty() {
                    segments.push(std::mem::take(&mut words));
                }
                while chars.peek().is_some_and(|next| *next == character) {
                    chars.next();
                }
            }
            character if character.is_whitespace() => {
                if !token.is_empty() {
                    words.push(std::mem::take(&mut token));
                }
            }
            _ => token.push(character),
        }
    }
    if !token.is_empty() {
        words.push(token);
    }
    if !words.is_empty() {
        segments.push(words);
    }
    segments
}

fn is_python_tool_word(word: &str) -> bool {
    if word.starts_with("--") {
        return false;
    }
    if word.contains('=') && !word.starts_with('$') {
        return false;
    }
    if matches!(word, "uv" | "uvx" | "pip" | "pip3") {
        return true;
    }
    if is_python_executable(word) {
        return true;
    }
    if is_python_launcher(word) {
        return true;
    }
    let normalized = word.trim_matches(|character| character == '"' || character == '\'');
    normalized.split('$').skip(1).any(|suffix| {
        let name: String = suffix
            .trim_start_matches('{')
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .collect();
        let name = name.to_ascii_uppercase();
        matches!(name.as_str(), "PYTHON" | "PYTHON2" | "PYTHON3" | "PYPY")
            || [
                "_PYTHON",
                "_PYTHON2",
                "_PYTHON3",
                "_PYTHON_BIN",
                "_PYTHON_CMD",
                "_PYTHON_EXECUTABLE",
            ]
            .iter()
            .any(|suffix| name.ends_with(suffix))
    }) || normalized.ends_with("_python3")
}

pub(super) fn javascript_source_python_invocations(source: &str) -> DynResult<Vec<String>> {
    if source.len() > JAVASCRIPT_SOURCE_LIMIT {
        return Err("JavaScript source exceeds the bounded Python invocation scan size".into());
    }
    let bytes = source.as_bytes();
    let mut cursor = 0;
    const PROCESS_CALLS: [&str; 10] = [
        "spawn",
        "spawnSync",
        "exec",
        "execSync",
        "execFile",
        "execFileSync",
        "fork",
        "execa",
        "execaSync",
        "Command",
    ];
    let mut invocations = Vec::new();

    while cursor < bytes.len() {
        if bytes[cursor..].starts_with(b"//") {
            cursor = bytes[cursor..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| cursor + offset + 1);
            continue;
        }
        if bytes[cursor..].starts_with(b"/*") {
            let Some(end) = bytes[cursor + 2..]
                .windows(2)
                .position(|window| window == b"*/")
            else {
                break;
            };
            cursor += end + 4;
            continue;
        }
        if matches!(bytes[cursor], b'\'' | b'"' | b'`') {
            cursor = javascript_string_end(bytes, cursor).unwrap_or(bytes.len());
            continue;
        }
        if is_javascript_identifier_start(bytes[cursor]) {
            let start = cursor;
            cursor += 1;
            while cursor < bytes.len() && is_javascript_identifier_continue(bytes[cursor]) {
                cursor += 1;
            }
            let name = &source[start..cursor];
            if PROCESS_CALLS.contains(&name)
                && let Some(executable) = javascript_first_string_argument(source, cursor)
                && if matches!(name, "exec" | "execSync") {
                    has_python_tool_invocation(executable)
                } else {
                    is_python_tool_word(executable)
                }
            {
                invocations.push(format!("{name}('{executable}')"));
            }
            continue;
        }
        cursor += 1;
    }
    Ok(invocations)
}

fn is_javascript_identifier_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'$')
}

fn is_javascript_identifier_continue(byte: u8) -> bool {
    is_javascript_identifier_start(byte) || byte.is_ascii_digit()
}

fn javascript_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let quote = *bytes.get(start)?;
    let mut cursor = start + 1;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'\\' => cursor = (cursor + 2).min(bytes.len()),
            byte if byte == quote => return Some(cursor + 1),
            _ => cursor += 1,
        }
    }
    None
}

fn javascript_first_string_argument(source: &str, mut cursor: usize) -> Option<&str> {
    let bytes = source.as_bytes();
    cursor = skip_javascript_trivia(bytes, cursor)?;
    if *bytes.get(cursor)? != b'(' {
        return None;
    }
    cursor = skip_javascript_trivia(bytes, cursor + 1)?;
    let quote = *bytes.get(cursor)?;
    if !matches!(quote, b'\'' | b'"' | b'`') {
        return None;
    }
    let content_start = cursor + 1;
    let mut end = content_start;
    while end < bytes.len() {
        match bytes[end] {
            b'\\' => end = (end + 2).min(bytes.len()),
            byte if byte == quote => {
                let content = &source[content_start..end];
                if quote == b'`' && content.contains("${") {
                    return None;
                }
                return Some(content);
            }
            _ => end += 1,
        }
    }
    None
}

fn skip_javascript_trivia(bytes: &[u8], mut cursor: usize) -> Option<usize> {
    loop {
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        if bytes.get(cursor..)?.starts_with(b"//") {
            cursor = bytes[cursor..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| cursor + offset + 1);
            continue;
        }
        if bytes.get(cursor..)?.starts_with(b"/*") {
            let end = bytes[cursor + 2..]
                .windows(2)
                .position(|window| window == b"*/")?;
            cursor += end + 4;
            continue;
        }
        return Some(cursor);
    }
}

pub(super) fn is_javascript_source(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("js" | "cjs" | "mjs" | "ts")
    ) && (path.to_string_lossy().starts_with("scripts/")
        || path.to_string_lossy().starts_with(".github/actions/"))
}

fn command_substitution_runs_python(line: &str) -> bool {
    let compact = line.replace(" ", "");
    compact.contains("$(python")
        || compact.contains("$(/usr/bin/python")
        || compact.contains("$(command-vpython")
}

fn heredoc_marker(line: &str) -> Option<String> {
    let start = line.find("<<")? + 2;
    let tail = line[start..].trim_start_matches('-').trim_start();
    let quoted = tail.strip_prefix('\'').or_else(|| tail.strip_prefix('"'));
    if let Some(quoted) = quoted {
        let quote = tail.chars().next()?;
        return quoted.split_once(quote).map(|(value, _)| value.to_owned());
    }
    let marker: String = tail
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect();
    (!marker.is_empty()).then_some(marker)
}

pub(super) fn check_rust_process_invocations(path: &str, bytes: &[u8]) -> DynResult<()> {
    let source = std::str::from_utf8(bytes)?;
    let syntax = syn::parse_file(source)
        .map_err(|error| format!("cannot parse production Rust source `{path}`: {error}"))?;
    let mut visitor = ProcessInvocationVisitor {
        path,
        violation: None,
    };
    visitor.visit_file(&syntax);
    if let Some(violation) = visitor.violation {
        return Err(violation.into());
    }
    Ok(())
}

struct ProcessInvocationVisitor<'a> {
    path: &'a str,
    violation: Option<String>,
}

impl Visit<'_> for ProcessInvocationVisitor<'_> {
    fn visit_item(&mut self, item: &Item) {
        if item.attrs().iter().any(is_cfg_test) {
            return;
        }
        syn::visit::visit_item(self, item);
    }

    fn visit_expr(&mut self, expression: &Expr) {
        if self.violation.is_none()
            && let Some(executable) = command_executable(expression)
        {
            if is_python_tool_word(&executable) {
                self.violation = Some(format!(
                    "production Rust process invocation runs Python tooling in `{}`",
                    self.path
                ));
            } else if (is_shell_executable(&executable) || is_env_executable(&executable))
                && expression_runs_python(expression)
            {
                self.violation = Some(format!(
                    "production Rust shell or env process embeds a Python command in `{}`",
                    self.path
                ));
            }
        }
        syn::visit::visit_expr(self, expression);
    }
}

fn command_executable(expression: &Expr) -> Option<String> {
    match expression {
        Expr::Call(call) => {
            command_new_path(&call.func).and_then(|_| call.args.first().and_then(string_literal))
        }
        Expr::MethodCall(method) => command_executable(&method.receiver),
        Expr::Paren(paren) => command_executable(&paren.expr),
        Expr::Group(group) => command_executable(&group.expr),
        Expr::Reference(reference) => command_executable(&reference.expr),
        Expr::Try(try_expr) => command_executable(&try_expr.expr),
        _ => None,
    }
}

fn is_cfg_test(attribute: &syn::Attribute) -> bool {
    if !attribute.path().is_ident("cfg") {
        return false;
    }
    let Meta::List(list) = &attribute.meta else {
        return false;
    };
    list.tokens.to_string().trim() == "test"
}

fn command_new_path(expression: &Expr) -> Option<()> {
    let Expr::Path(path) = expression else {
        return None;
    };
    let segments: Vec<_> = path
        .path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    (segments.len() >= 2
        && segments.last().is_some_and(|segment| segment == "new")
        && matches!(
            segments[segments.len() - 2].as_str(),
            "Command" | "ProcessCommand"
        ))
    .then_some(())
}

fn string_literal(expression: &Expr) -> Option<String> {
    match expression {
        Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Str(value) => Some(value.value()),
            _ => None,
        },
        Expr::Paren(paren) => string_literal(&paren.expr),
        Expr::Group(group) => string_literal(&group.expr),
        _ => None,
    }
}

fn expression_runs_python(expression: &Expr) -> bool {
    let mut literals = StringLiteralVisitor::default();
    literals.visit_expr(expression);
    literals
        .values
        .iter()
        .any(|value| has_python_tool_invocation(value))
}

#[derive(Default)]
struct StringLiteralVisitor {
    values: Vec<String>,
}

impl Visit<'_> for StringLiteralVisitor {
    fn visit_lit_str(&mut self, literal: &syn::LitStr) {
        self.values.push(literal.value());
    }
}

fn is_shell_executable(executable: &str) -> bool {
    Path::new(executable)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            matches!(
                name,
                "sh" | "bash" | "zsh" | "dash" | "cmd" | "powershell" | "pwsh"
            )
        })
}

fn is_env_executable(executable: &str) -> bool {
    Path::new(executable)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, "env" | "env.exe"))
}

pub(super) fn is_python_source(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "py" | "pyw" | "pyi" | "ipynb"
            )
        })
}

pub(super) fn is_python_package_manifest(path: &Path, absolute: &Path) -> DynResult<bool> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    let name = name.to_ascii_lowercase();
    if matches!(
        name.as_str(),
        "pyproject.toml"
            | "setup.py"
            | "setup.cfg"
            | "pipfile"
            | "pipfile.lock"
            | "uv.lock"
            | "poetry.lock"
            | "pdm.lock"
            | "tox.ini"
            | ".python-version"
            | "pixi.toml"
            | "pixi.lock"
            | "conda.yml"
            | "conda.yaml"
    ) || (name.starts_with("requirements")
        && matches!(
            Path::new(&name)
                .extension()
                .and_then(|extension| extension.to_str()),
            Some("txt" | "in" | "lock")
        ))
        || (name.starts_with("constraints")
            && matches!(
                Path::new(&name)
                    .extension()
                    .and_then(|extension| extension.to_str()),
                Some("txt" | "in" | "lock")
            ))
    {
        return Ok(true);
    }
    if matches!(name.as_str(), "environment.yml" | "environment.yaml") {
        let text = fs::read_to_string(absolute)?.to_ascii_lowercase();
        return Ok(text.lines().any(|line| {
            let line = line.trim();
            line.starts_with("- python") || line.starts_with("- pip")
        }));
    }
    Ok(false)
}

pub(super) fn has_python_shebang(contents: &[u8]) -> bool {
    let first = contents
        .split(|byte| *byte == b'\n')
        .next()
        .unwrap_or_default();
    let Ok(first) = std::str::from_utf8(first) else {
        return false;
    };
    let Some(interpreter) = first.trim().strip_prefix("#!") else {
        return false;
    };
    let words: Vec<_> = interpreter.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }
    let first_name = Path::new(words[0])
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(words[0]);
    if is_python_executable(first_name) {
        return true;
    }
    first_name == "env"
        && words
            .iter()
            .skip(1)
            .find(|word| !word.starts_with('-') && !is_environment_assignment(word))
            .is_some_and(|word| is_python_executable(word))
}

fn is_environment_assignment(word: &str) -> bool {
    let Some((name, _)) = word.split_once('=') else {
        return false;
    };
    let mut characters = name.bytes();
    characters
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && characters.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn is_python_executable(word: &str) -> bool {
    let basename = Path::new(word)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(word)
        .to_ascii_lowercase();
    let basename = basename.strip_suffix(".exe").unwrap_or(&basename);
    ["python", "pypy"].iter().any(|prefix| {
        basename.strip_prefix(prefix).is_some_and(|suffix| {
            suffix.is_empty()
                || (suffix
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_ascii_digit())
                    && suffix
                        .chars()
                        .all(|character| character.is_ascii_digit() || character == '.'))
        })
    })
}

fn is_python_launcher(word: &str) -> bool {
    let basename = Path::new(word)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(word)
        .to_ascii_lowercase();
    let basename = basename.strip_suffix(".exe").unwrap_or(&basename);
    basename == "py"
        || basename.strip_prefix("py").is_some_and(|suffix| {
            !suffix.is_empty()
                && suffix
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || byte == b'.')
        })
        || basename.strip_prefix("pip").is_some_and(|suffix| {
            suffix.is_empty()
                || suffix
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || byte == b'.')
        })
}

fn is_workflow_or_action_yaml(path: &Path) -> bool {
    let text = path.to_string_lossy();
    (text.starts_with(".github/workflows/") || text.starts_with(".github/actions/"))
        && matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("yml" | "yaml")
        )
}

fn is_shell_or_just_source(path: &Path) -> bool {
    let text = path.to_string_lossy();
    let extension = path.extension().and_then(|extension| extension.to_str());
    if text == "Justfile" || text.starts_with("just/") && extension == Some("just") {
        return true;
    }
    if matches!(
        extension,
        Some("sh" | "bash" | "zsh" | "ps1" | "cmd" | "bat")
    ) {
        // Fixture directories contain inert historical command transcripts and samples.
        return !path.components().any(|component| {
            matches!(
                component.as_os_str().to_str(),
                Some("fixtures" | "testdata")
            )
        });
    }
    extension.is_none() && (text.starts_with("scripts/") || text.starts_with(".github/actions/"))
}

fn is_executable_file(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path)
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

pub(super) fn is_production_rust_source(path: &Path) -> bool {
    path.extension().and_then(|extension| extension.to_str()) == Some("rs")
        && (path.file_name().is_some_and(|name| name == "build.rs")
            || path
                .components()
                .any(|component| component.as_os_str() == "src"))
}

trait ItemAttributes {
    fn attrs(&self) -> &[syn::Attribute];
}

impl ItemAttributes for Item {
    fn attrs(&self) -> &[syn::Attribute] {
        match self {
            Item::Const(item) => &item.attrs,
            Item::Enum(item) => &item.attrs,
            Item::ExternCrate(item) => &item.attrs,
            Item::Fn(item) => &item.attrs,
            Item::ForeignMod(item) => &item.attrs,
            Item::Impl(item) => &item.attrs,
            Item::Macro(item) => &item.attrs,
            Item::Mod(item) => &item.attrs,
            Item::Static(item) => &item.attrs,
            Item::Struct(item) => &item.attrs,
            Item::Trait(item) => &item.attrs,
            Item::TraitAlias(item) => &item.attrs,
            Item::Type(item) => &item.attrs,
            Item::Union(item) => &item.attrs,
            Item::Use(item) => &item.attrs,
            _ => &[],
        }
    }
}

use super::Header;

fn anchor(prefix: &str, name: &str) -> String {
    let mut slug = String::new();
    for character in name.to_lowercase().chars() {
        if character.is_ascii_lowercase() || character.is_ascii_digit() {
            slug.push(character);
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    format!("skippy-{prefix}-{}", slug.trim_matches('-'))
}

fn pretty(declaration: &str) -> String {
    let text = declaration
        .replacen('(', "(\n        ", 1)
        .replace(", ", ",\n        ");
    if let Some(prefix) = text.strip_suffix(");")
        && prefix.ends_with(char::is_whitespace)
    {
        return format!("{}\n);", prefix.trim_end());
    }
    text
}

pub(super) fn render(headers: &[Header]) -> String {
    let count: usize = headers.iter().map(|header| header.functions.len()).sum();
    let mut lines: Vec<String> = include_str!("intro.txt")
        .lines()
        .map(str::to_owned)
        .collect();
    lines.push(String::new());
    lines.push(format!(
        "Current generated surface: **{} headers** and **{count} exported functions**.",
        headers.len()
    ));
    lines.extend(include_str!("navigation.txt").lines().map(str::to_owned));
    for header in headers.iter().filter(|header| !header.functions.is_empty()) {
        let label = if header.functions.len() == 1 {
            "function"
        } else {
            "functions"
        };
        lines.push("    <section class=\"skippy-api-index__group\">".into());
        lines.push(format!("      <a class=\"skippy-api-index__group-title\" href=\"#{}\"><code>{}</code><span>{} {label}</span></a>", anchor("header", &header.name), header.name, header.functions.len()));
        lines.push("      <div class=\"skippy-api-index__functions\">".into());
        for function in &header.functions {
            lines.push(format!(
                "        <a href=\"#{}\"><code>{}</code></a>",
                anchor("fn", &function.name),
                function.name
            ));
        }
        lines.extend(["      </div>".into(), "    </section>".into()]);
    }
    lines.extend(include_str!("include_map.txt").lines().map(str::to_owned));
    for header in headers {
        let path = if header.name == "skippy.h" {
            "include/skippy.h".into()
        } else {
            format!("include/skippy/{}", header.name)
        };
        lines.push(format!("| `{path}` | {} |", header.brief));
    }
    lines.extend(include_str!("conventions.txt").lines().map(str::to_owned));
    lines.push(String::new());
    for header in headers.iter().filter(|header| !header.functions.is_empty()) {
        lines.extend([
            format!("<a id=\"{}\"></a>", anchor("header", &header.name)),
            format!("### `{}`", header.name),
            String::new(),
        ]);
        for function in &header.functions {
            lines.extend([
                format!("<a id=\"{}\"></a>", anchor("fn", &function.name)),
                format!("#### `{}`", function.name),
                String::new(),
                function.brief.clone(),
                String::new(),
                "```cpp".into(),
                pretty(&function.declaration),
                "```".into(),
                String::new(),
            ]);
        }
        lines.extend(["<a class=\"skippy-api-backlink\" href=\"#skippy-function-index\">↩ Back to function index</a>".into(), String::new()]);
    }
    lines.extend([
        "<a id=\"skippy-native-declarations\"></a>".into(),
        "## Native declarations".into(),
        String::new(),
        "The headers also define the following enums, structs, opaque handles, and ABI constants:"
            .into(),
        String::new(),
    ]);
    for header in headers
        .iter()
        .filter(|header| !header.declarations.is_empty())
    {
        let declarations = header
            .declarations
            .iter()
            .map(|item| format!("`{item}`"))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("- `{}`: {declarations}", header.name));
    }
    lines.extend([String::new(), "Source directory: `include/skippy/`. Regenerate this page after changing any public header or exported function.".into(), String::new()]);
    lines.join("\n")
}

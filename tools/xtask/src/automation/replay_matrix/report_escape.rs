pub(super) fn xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub(super) fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

pub(super) fn code(text: &str) -> String {
    format!(
        "<code>{}</code>",
        xml(text)
            .replace('|', "&#124;")
            .replace('`', "&#96;")
            .replace(['\r', '\n'], " ")
    )
}

pub(super) fn number(value: Option<f64>, digits: usize, suffix: &str) -> String {
    value.map_or_else(
        || "\u{2014}".into(),
        |value| format!("{value:.digits$}{suffix}"),
    )
}

pub(super) fn range(values: (Option<f64>, Option<f64>), digits: usize) -> String {
    match values {
        (Some(low), Some(high)) => format!("{low:.digits$}\u{2013}{high:.digits$}"),
        (Some(_) | None, None) | (None, Some(_)) => "\u{2014}".into(),
    }
}

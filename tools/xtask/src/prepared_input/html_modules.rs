//! The `src` of every `<script type="module">` start tag, found the way
//! Python's `html.parser.HTMLParser` tokenizes a document: case-insensitive
//! tag and attribute names, comments and declarations skipped, `<script>`
//! and `<style>` bodies treated as raw text unless the tag self-closes, the
//! last duplicate attribute winning, and quoted or bare values with
//! character references decoded. An unterminated final tag is ignored.

/// One module script's `src`: `Some("")` when absent
/// (`attrs.get("src", "")`), `None` when present without a value.
pub(crate) type ModuleSource = Option<String>;

type Attributes = Vec<(String, Option<String>)>;

pub(crate) fn module_sources(html: &str) -> Vec<ModuleSource> {
    let mut sources = Vec::new();
    let mut rest = html;
    while let Some(start) = rest.find('<') {
        rest = &rest[start..];
        let Some(tag) = start_tag(rest) else {
            rest = skip_markup(rest);
            continue;
        };
        rest = tag.after;
        if tag.name == "script" && attribute(&tag.attrs, "type") == Some(Some("module")) {
            let src = attribute(&tag.attrs, "src").unwrap_or(Some(""));
            sources.push(src.map(str::to_owned));
        }
        if !tag.self_closing && matches!(tag.name.as_str(), "script" | "style") {
            rest = skip_raw_text(rest, &tag.name);
        }
    }
    sources
}

struct StartTag<'a> {
    name: String,
    attrs: Attributes,
    self_closing: bool,
    after: &'a str,
}

/// The last value of `name`: `None` when absent, `Some(None)` when bare.
fn attribute<'a>(attrs: &'a Attributes, name: &str) -> Option<Option<&'a str>> {
    attrs
        .iter()
        .rev()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_deref())
}

/// Skips a comment, declaration, processing instruction, end tag or a
/// literal `<` that does not open a tag.
fn skip_markup(text: &str) -> &str {
    let close = |marker: &str, from: usize| {
        text[from..]
            .find(marker)
            .map_or("", |end| &text[from + end + marker.len()..])
    };
    if text.starts_with("<!--") {
        close("-->", 4)
    } else if text.starts_with("<!") || text.starts_with("<?") || text.starts_with("</") {
        close(">", 2)
    } else {
        &text[1..]
    }
}

/// Everything after the `</name` end tag that closes a raw-text element.
fn skip_raw_text<'a>(text: &'a str, name: &str) -> &'a str {
    let lower = text.to_ascii_lowercase();
    let needle = format!("</{name}");
    let mut from = 0;
    while let Some(found) = lower[from..].find(&needle) {
        let end = from + found + needle.len();
        let boundary = lower[end..].chars().next();
        if boundary.is_none_or(|ch| ch.is_ascii_whitespace() || ch == '/' || ch == '>') {
            return skip_markup(&text[from + found..]);
        }
        from = end;
    }
    ""
}

fn start_tag(text: &str) -> Option<StartTag<'_>> {
    let body = text.strip_prefix('<')?;
    if !body.starts_with(|ch: char| ch.is_ascii_alphabetic()) {
        return None;
    }
    let name_end = body
        .find(|ch: char| ch.is_ascii_whitespace() || matches!(ch, '/' | '>' | '\0'))
        .unwrap_or(body.len());
    let name = body[..name_end].to_ascii_lowercase();
    let mut rest = &body[name_end..];
    let mut attrs = Vec::new();
    loop {
        rest = rest.trim_start_matches(|ch: char| ch.is_ascii_whitespace());
        let (self_closing, after) = match rest.strip_prefix("/>") {
            Some(after) => (true, Some(after)),
            None => (false, rest.strip_prefix('>')),
        };
        if let Some(after) = after {
            return Some(StartTag {
                name,
                attrs,
                self_closing,
                after,
            });
        }
        if let Some(after) = rest.strip_prefix('/') {
            rest = after;
            continue;
        }
        if rest.is_empty() {
            return None;
        }
        let (key, value, after) = attribute_at(rest);
        attrs.push((key, value));
        rest = after;
    }
}

/// One `name[=value]` pair starting at a non-space, non-`/` character.
fn attribute_at(text: &str) -> (String, Option<String>, &str) {
    let first = text.chars().next().map_or(0, char::len_utf8);
    let name_end = text[first..]
        .find(|ch: char| ch.is_ascii_whitespace() || matches!(ch, '/' | '=' | '>'))
        .map_or(text.len(), |end| first + end);
    let key = text[..name_end].to_ascii_lowercase();
    let after_name = &text[name_end..];
    let spaced = after_name.trim_start_matches(|ch: char| ch.is_ascii_whitespace());
    let Some(value_text) = spaced.strip_prefix('=') else {
        return (key, None, after_name);
    };
    let value_text = value_text.trim_start_matches(|ch: char| ch.is_ascii_whitespace());
    for quote in ['"', '\''] {
        if let Some(quoted) = value_text.strip_prefix(quote)
            && let Some(end) = quoted.find(quote)
        {
            return (key, Some(unescape(&quoted[..end])), &quoted[end + 1..]);
        }
    }
    let end = value_text
        .find(|ch: char| ch.is_ascii_whitespace() || ch == '>')
        .unwrap_or(value_text.len());
    (key, Some(unescape(&value_text[..end])), &value_text[end..])
}

/// `html.unescape` for the references that appear in attribute values:
/// the XML entities and decimal or hexadecimal character references.
fn unescape(value: &str) -> String {
    let mut out = String::new();
    let mut rest = value;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let (text, length) =
            reference(rest).map_or(("&".to_owned(), 1), |(ch, length)| (ch.to_string(), length));
        out.push_str(&text);
        rest = &rest[length..];
    }
    out.push_str(rest);
    out
}

fn reference(text: &str) -> Option<(char, usize)> {
    let end = text.find(';')?;
    let ch = match &text[1..end] {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        body => {
            let digits = body.strip_prefix('#')?;
            let code = match digits.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => digits.parse().ok()?,
            };
            char::from_u32(code)?
        }
    };
    Some((ch, end + 1))
}

#[cfg(test)]
mod tests {
    use super::module_sources;

    fn some(text: &str) -> Option<String> {
        Some(text.to_owned())
    }

    /// Expected values are `HTMLParser` observations on Python 3.13.
    #[test]
    fn migration_prepared_inputs_html_modules_follow_html_parser() {
        let html = "<!doctype html><!-- <script type=module src=a.js> -->\
                    <STYLE>x<script type=module src=b.js></style>\
                    <script type=\"module\" src='c&amp;d.js' src=/e.js/>\
                    <script\ttype=module src=f.js></script>\
                    <script type=text/javascript src=g.js></script>\
                    <script type=module>import('/h.js')</script>";
        assert_eq!(module_sources(html), [some("/e.js/"), some("")]);
        let closed = "<script type=module src=a.js />x</script><script type=module src=b.js>";
        assert_eq!(module_sources(closed), [some("a.js"), some("b.js")]);
        let spaced = "<script type = \"module\" src = x&#47;y.js><script type=MODULE src=z.js>";
        assert_eq!(module_sources(spaced), [some("x/y.js")]);
        assert_eq!(
            module_sources("<script type=module src=\"a\" src></script>"),
            [None]
        );
        assert!(module_sources("<script type=module src=\"x.js\"").is_empty());
    }
}

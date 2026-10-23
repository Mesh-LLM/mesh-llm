//! Extract asset and external module URLs from built HTML. This small scanner
//! covers start tags, comments, and script/style raw text used by these inputs.

type Attributes = Vec<(String, Option<String>)>;

pub(crate) fn module_sources(html: &str) -> Result<Vec<String>, String> {
    let mut sources = Vec::new();
    for tag in start_tags(html)? {
        if tag.name != "script"
            || !attribute(&tag.attrs, "type").is_some_and(|value| {
                value.is_some_and(|value| value.eq_ignore_ascii_case("module"))
            })
        {
            continue;
        }
        match attribute(&tag.attrs, "src") {
            None => {}
            Some(None) => return Err("module script src attribute needs a value".into()),
            Some(Some(source)) => sources.push(source.to_owned()),
        }
    }
    Ok(sources)
}

pub(crate) fn asset_references(html: &str) -> Result<Vec<String>, String> {
    let mut references = Vec::new();
    for tag in start_tags(html)? {
        for name in ["src", "href"] {
            if let Some(Some(value)) = attribute(&tag.attrs, name)
                && !value.is_empty()
            {
                references.push(value.to_owned());
            }
        }
    }
    Ok(references)
}

struct StartTag {
    name: String,
    attrs: Attributes,
}

/// Keep the first duplicate attribute, matching browser tokenization.
fn attribute<'a>(attrs: &'a Attributes, name: &str) -> Option<Option<&'a str>> {
    attrs
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_deref())
}

fn start_tags(html: &str) -> Result<Vec<StartTag>, String> {
    let mut tags = Vec::new();
    let mut rest = html;
    while let Some(start) = rest.find('<') {
        rest = &rest[start..];
        let Some((tag, after)) = parse_start_tag(rest)? else {
            rest = skip_markup(rest);
            continue;
        };
        let raw_text = matches!(tag.name.as_str(), "script" | "style");
        let name = tag.name.clone();
        tags.push(tag);
        rest = if raw_text {
            skip_raw_text(after, &name)
        } else {
            after
        };
    }
    Ok(tags)
}

fn parse_start_tag(text: &str) -> Result<Option<(StartTag, &str)>, String> {
    let Some(body) = text.strip_prefix('<') else {
        return Ok(None);
    };
    if !body.starts_with(|ch: char| ch.is_ascii_alphabetic()) {
        return Ok(None);
    }
    let name_end = body
        .find(|ch: char| ch.is_ascii_whitespace() || matches!(ch, '/' | '>' | '\0'))
        .unwrap_or(body.len());
    let name = body[..name_end].to_ascii_lowercase();
    let mut rest = &body[name_end..];
    let mut attrs = Vec::new();
    loop {
        rest = rest.trim_start_matches(|ch: char| ch.is_ascii_whitespace());
        if let Some(after) = rest.strip_prefix('>') {
            return Ok(Some((StartTag { name, attrs }, after)));
        }
        if let Some(after) = rest.strip_prefix("/>") {
            return Ok(Some((StartTag { name, attrs }, after)));
        }
        if let Some(after) = rest.strip_prefix('/') {
            rest = after;
            continue;
        }
        if rest.is_empty() {
            return Ok(None);
        }
        let (key, value, after) = parse_attribute(rest)?;
        attrs.push((key, value));
        rest = after;
    }
}

fn parse_attribute(text: &str) -> Result<(String, Option<String>, &str), String> {
    let first = text.chars().next().map_or(0, char::len_utf8);
    let name_end = text[first..]
        .find(|ch: char| ch.is_ascii_whitespace() || matches!(ch, '/' | '=' | '>'))
        .map_or(text.len(), |end| first + end);
    let key = text[..name_end].to_ascii_lowercase();
    let after_name = &text[name_end..];
    let spaced = after_name.trim_start_matches(|ch: char| ch.is_ascii_whitespace());
    let Some(value_text) = spaced.strip_prefix('=') else {
        return Ok((key, None, after_name));
    };
    let value_text = value_text.trim_start_matches(|ch: char| ch.is_ascii_whitespace());
    for quote in ['"', '\''] {
        if let Some(quoted) = value_text.strip_prefix(quote) {
            let Some(end) = quoted.find(quote) else {
                return Err("unterminated quoted HTML attribute".into());
            };
            let value = decode_attribute(&key, &quoted[..end])?;
            return Ok((key, Some(value), &quoted[end + 1..]));
        }
    }
    let end = value_text
        .find(|ch: char| ch.is_ascii_whitespace() || ch == '>')
        .unwrap_or(value_text.len());
    let value = decode_attribute(&key, &value_text[..end])?;
    Ok((key, Some(value), &value_text[end..]))
}

fn decode_attribute(name: &str, value: &str) -> Result<String, String> {
    if matches!(name, "src" | "href" | "type") {
        decode_references(value)
    } else {
        Ok(value.to_owned())
    }
}

/// Decode common URL-relevant named references and numeric character refs.
/// Reject unknown semicolon-terminated names because browsers may decode them.
fn decode_references(value: &str) -> Result<String, String> {
    let mut output = String::new();
    let mut rest = value;
    while let Some(start) = rest.find('&') {
        output.push_str(&rest[..start]);
        rest = &rest[start..];
        match decode_reference(rest)? {
            Some((character, length)) => {
                output.push(character);
                rest = &rest[length..];
            }
            None => {
                output.push('&');
                rest = &rest[1..];
            }
        }
    }
    output.push_str(rest);
    Ok(output)
}

fn decode_reference(text: &str) -> Result<Option<(char, usize)>, String> {
    if text.starts_with("&#") {
        return decode_numeric_reference(text).map(Some);
    }
    let Some(end) = text.find(';') else {
        return Ok(None);
    };
    let body = &text[1..end];
    let character = match body {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "copy" => '©',
        "reg" => '®',
        "nbsp" => '\u{00a0}',
        "colon" => ':',
        "sol" => '/',
        "period" => '.',
        "quest" => '?',
        "num" => '#',
        "equals" => '=',
        "percnt" => '%',
        "commat" => '@',
        "plus" => '+',
        "semi" => ';',
        "lowbar" => '_',
        "hyphen" => '\u{2010}',
        body if body.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
            && body.bytes().all(|byte| byte.is_ascii_alphanumeric()) =>
        {
            return Err("unsupported named HTML character reference".into());
        }
        _ => return Ok(None),
    };
    Ok(Some((character, end + 1)))
}

/// Numeric references consume the longest decimal or hexadecimal digit run;
/// HTML permits the terminating semicolon to be omitted.
fn decode_numeric_reference(text: &str) -> Result<(char, usize), String> {
    let bytes = text.as_bytes();
    let (radix, digits_start) = match bytes.get(2).copied() {
        Some(b'x' | b'X') => (16, 3),
        _ => (10, 2),
    };
    let is_digit = |byte: &u8| match radix {
        16 => byte.is_ascii_hexdigit(),
        _ => byte.is_ascii_digit(),
    };
    let digits_len = bytes[digits_start..]
        .iter()
        .copied()
        .take_while(is_digit)
        .count();
    if digits_len == 0 {
        return Err("invalid numeric HTML character reference".into());
    }
    let digits_end = digits_start + digits_len;
    let code = u32::from_str_radix(&text[digits_start..digits_end], radix)
        .map_err(|_| "invalid numeric HTML character reference".to_owned())?;
    let character = char::from_u32(code)
        .ok_or_else(|| "invalid numeric HTML character reference".to_owned())?;
    let semicolon = if bytes.get(digits_end) == Some(&b';') {
        1
    } else {
        0
    };
    let consumed = digits_end + semicolon;
    Ok((character, consumed))
}

/// Skip a comment, declaration, processing instruction, end tag, or `<` text.
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

/// Skip script/style bodies through their closing tags so text is not scanned
/// as markup. This deliberately handles only the raw-text elements used here.
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

#[cfg(test)]
mod tests {
    use super::{asset_references, module_sources};

    #[test]
    fn extracts_external_modules_while_skipping_comments_and_raw_text() {
        let html = "<!doctype html><!-- <script type=module src=hidden.js> -->\
                    <style>.x{content:'<script type=module src=hidden.js>'}</style>\
                    <script>const text = '<script type=module src=hidden.js>';</script>\
                    <SCRIPT TYPE=module SRC='assets/app&amp;main.js'></SCRIPT>\
                    <script type=text/javascript src=ignored.js></script>\
                    <script type=module>import('/inline.js')</script>";
        assert_eq!(
            module_sources(html).expect("extract module source"),
            ["assets/app&main.js"]
        );
        assert_eq!(
            module_sources("<script type=module src=first.js src=second.js></script>")
                .expect("extract first duplicate source"),
            ["first.js"]
        );
    }

    #[test]
    fn extracts_asset_urls_from_quoted_and_unquoted_attributes() {
        let html = "<!-- <img src=hidden> --><img src=assets/app.js>\
                    <a href='assets/a&#47;b.css?x=1&amp;y=2'></a>";
        assert_eq!(
            asset_references(html).expect("extract asset URLs"),
            ["assets/app.js", "assets/a/b.css?x=1&y=2"]
        );
    }

    #[test]
    fn decodes_common_named_references_and_rejects_malformed_numeric_references() {
        assert_eq!(
            asset_references("<img src='assets/copy&copy;.js'>").expect("decode copy"),
            ["assets/copy©.js"]
        );
        assert_eq!(
            asset_references("<a href='assets&sol;app&period;js?x&equals;1'>")
                .expect("decode URL delimiters"),
            ["assets/app.js?x=1"]
        );
        assert!(asset_references("<img src='assets/a&#xZZ;.js'>").is_err());
        assert_eq!(
            asset_references("<img src='assets/a&colon;.js'>").expect("decode colon"),
            ["assets/a:.js"]
        );
        assert!(asset_references("<img src='assets/a&vendor;.js'>").is_err());
        assert_eq!(
            asset_references("<a href='assets/app.js?x=1&y=2'>")
                .expect("preserve query ampersands"),
            ["assets/app.js?x=1&y=2"]
        );
        assert!(module_sources("<script type=module src></script>").is_err());
    }

    #[test]
    fn decodes_named_hyphens_in_asset_and_module_paths() {
        let html = "<script type=module src='assets/app&hyphen;main.js'></script>\
                    <link href=assets/app&hyphen;main.css>\
                    <img src='assets/app-main.png'>";
        assert_eq!(
            module_sources(html).expect("decode module path hyphen"),
            ["assets/app\u{2010}main.js"]
        );
        assert_eq!(
            asset_references(html).expect("decode asset path hyphens"),
            [
                "assets/app\u{2010}main.js",
                "assets/app\u{2010}main.css",
                "assets/app-main.png"
            ]
        );
    }

    #[test]
    fn decodes_semicolonless_numeric_references_to_the_digit_boundary() {
        assert_eq!(
            module_sources("<script type=module src='assets&#47app.js'></script>")
                .expect("decode decimal path separator"),
            ["assets/app.js"]
        );
        assert_eq!(
            asset_references("<img src='assets&#x2fgapp.js'>")
                .expect("decode hexadecimal path separator"),
            ["assets/gapp.js"]
        );
        assert!(asset_references("<img src='assets&#xZZ;.js'>").is_err());
    }
}

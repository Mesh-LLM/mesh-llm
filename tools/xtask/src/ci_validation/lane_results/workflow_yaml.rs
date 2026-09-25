//! A block-structured reader for the GitHub Actions workflow subset the CI
//! lanes use: indented mappings and sequences, plain/quoted scalars, flow
//! sequences, `{}`, block scalars, anchors and aliases (kept as text).
//! Workflow graph contracts read this tree instead of matching text.

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Node {
    Scalar(String),
    Seq(Vec<Node>),
    Map(Vec<(String, Node)>),
}

impl Node {
    pub(crate) fn get(&self, key: &str) -> Option<&Node> {
        match self {
            Node::Map(entries) => entries.iter().find(|(name, _)| name == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub(crate) fn entries(&self) -> &[(String, Node)] {
        match self {
            Node::Map(entries) => entries,
            _ => &[],
        }
    }

    pub(crate) fn text(&self) -> Option<&str> {
        match self {
            Node::Scalar(text) => Some(text),
            _ => None,
        }
    }

    /// A scalar or a sequence of scalars, as `needs` and `branches` allow.
    pub(crate) fn list(&self) -> Vec<&str> {
        match self {
            Node::Scalar(text) if text.is_empty() => Vec::new(),
            Node::Scalar(text) => vec![text.as_str()],
            Node::Seq(items) => items.iter().filter_map(Node::text).collect(),
            Node::Map(_) => Vec::new(),
        }
    }
}

struct Line<'a> {
    indent: usize,
    body: &'a str,
}

struct Reader<'a> {
    raw: Vec<&'a str>,
    at: usize,
}

pub(crate) fn parse(source: &str) -> Result<Node, String> {
    let mut reader = Reader {
        raw: source.lines().collect(),
        at: 0,
    };
    let Some(first) = reader.peek() else {
        return Ok(Node::Map(Vec::new()));
    };
    let node = reader.block(first.indent)?;
    match reader.peek() {
        None => Ok(node),
        Some(_) => Err(format!("unexpected indentation at line {}", reader.at + 1)),
    }
}

impl<'a> Reader<'a> {
    /// The next structural line, skipping blanks and full-line comments.
    fn peek(&mut self) -> Option<Line<'a>> {
        while let Some(raw) = self.raw.get(self.at) {
            let body = raw.trim_start();
            if body.is_empty() || body.starts_with('#') {
                self.at += 1;
                continue;
            }
            return Some(Line {
                indent: raw.len() - body.len(),
                body: body.trim_end(),
            });
        }
        None
    }

    fn block(&mut self, indent: usize) -> Result<Node, String> {
        match self.peek() {
            Some(line) if is_item(line.body) => self.seq(indent),
            _ => self.map(indent, None),
        }
    }

    fn seq(&mut self, indent: usize) -> Result<Node, String> {
        let mut items = Vec::new();
        while let Some(line) = self
            .peek()
            .filter(|l| l.indent == indent && is_item(l.body))
        {
            let rest = line.body[1..].trim_start();
            let inner = indent + (line.body.len() - rest.len());
            if rest.is_empty() {
                self.at += 1;
                items.push(self.nested(indent)?);
            } else if split_key(rest).is_some() {
                items.push(self.map(inner, Some(rest))?);
            } else {
                self.at += 1;
                items.push(inline(rest));
            }
        }
        Ok(Node::Seq(items))
    }

    /// A mapping at `indent`; `first` is an entry sharing a `- ` line.
    fn map(&mut self, indent: usize, first: Option<&'a str>) -> Result<Node, String> {
        let mut entries = Vec::new();
        let mut pending = first;
        loop {
            let body = match pending.take() {
                Some(body) => body,
                None => match self
                    .peek()
                    .filter(|l| l.indent == indent && !is_item(l.body))
                {
                    Some(line) => line.body,
                    None => break,
                },
            };
            let (key, value) =
                split_key(body).ok_or_else(|| format!("expected a key at line {}", self.at + 1))?;
            self.at += 1;
            entries.push((key.to_owned(), self.value(indent, value)?));
        }
        Ok(Node::Map(entries))
    }

    fn value(&mut self, indent: usize, value: &str) -> Result<Node, String> {
        let value = strip_anchor(strip_comment(value));
        if matches!(value.chars().next(), Some('|' | '>')) {
            return Ok(Node::Scalar(self.block_scalar(indent)));
        }
        if !value.is_empty() {
            return Ok(inline(value));
        }
        match self.peek() {
            Some(line) if line.indent == indent && is_item(line.body) => self.seq(indent),
            _ => self.nested(indent),
        }
    }

    fn nested(&mut self, indent: usize) -> Result<Node, String> {
        match self.peek() {
            Some(line) if line.indent > indent => self.block(line.indent),
            _ => Ok(Node::Scalar(String::new())),
        }
    }

    fn block_scalar(&mut self, indent: usize) -> String {
        let mut lines = Vec::new();
        while let Some(raw) = self.raw.get(self.at) {
            let body = raw.trim_start();
            if !body.is_empty() && raw.len() - body.len() <= indent {
                break;
            }
            lines.push(*raw);
            self.at += 1;
        }
        let margin = lines
            .iter()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.len() - line.trim_start().len())
            .min()
            .unwrap_or(0);
        lines
            .iter()
            .map(|line| line.get(margin..).unwrap_or("").trim_end())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn is_item(body: &str) -> bool {
    body == "-" || body.starts_with("- ")
}

fn split_key(body: &str) -> Option<(&str, &str)> {
    if body.starts_with(['"', '\'', '[', '{', '$']) {
        return None;
    }
    let colon = body
        .char_indices()
        .find(|(at, ch)| *ch == ':' && matches!(body[at + 1..].chars().next(), None | Some(' ')))?
        .0;
    Some((body[..colon].trim(), body[colon + 1..].trim()))
}

fn strip_comment(value: &str) -> &str {
    let mut quote = None;
    for (at, ch) in value.char_indices() {
        match (quote, ch) {
            (None, '\'' | '"') => quote = Some(ch),
            (Some(open), _) if ch == open => quote = None,
            (None, '#') if at == 0 || value[..at].ends_with(' ') => return value[..at].trim_end(),
            _ => {}
        }
    }
    value
}

fn strip_anchor(value: &str) -> &str {
    match value.strip_prefix('&') {
        Some(rest) => rest
            .split_once(' ')
            .map_or("", |(_, tail)| tail.trim_start()),
        None => value,
    }
}

fn inline(value: &str) -> Node {
    let value = strip_comment(value);
    if value == "{}" {
        return Node::Map(Vec::new());
    }
    if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
        let items = inner
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty());
        return Node::Seq(items.map(|item| Node::Scalar(unquote(item))).collect());
    }
    Node::Scalar(unquote(value))
}

fn unquote(value: &str) -> String {
    if let Some(inner) = value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')) {
        return inner.replace("''", "'");
    }
    if let Some(inner) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        return inner.replace("\\\"", "\"").replace("\\\\", "\\");
    }
    value.to_owned()
}

#[cfg(test)]
#[path = "workflow_yaml_tests.rs"]
mod tests;

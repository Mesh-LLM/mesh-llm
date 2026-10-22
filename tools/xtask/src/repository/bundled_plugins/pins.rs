//! Reads `DEFAULT_PLUGINS` out of `mesh-llm-plugin-manager`'s source, so the
//! release job bundles exactly what the host pins. Only the fields the
//! release needs are read: `name`, `version`, `repository` and `sha256`.

use syn::{Attribute, Expr, Item, Lit, Member};

/// One default plugin's pins, as `DEFAULT_PLUGINS` states them.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct PinnedDefault {
    pub(super) name: String,
    pub(super) version: String,
    pub(super) repository: String,
    /// `(target triple, lowercase hex SHA-256)`.
    pub(super) sha256: Vec<(String, String)>,
}

impl PinnedDefault {
    /// The archive this default's release publishes for `target`, the name
    /// it has in a bundle's `plugins/` too.
    pub(super) fn archive_name(&self, target: &str) -> String {
        format!("{}-{}-{target}.tar.gz", self.name, self.version)
    }

    /// `<name> <version> <sha256> <url>` for `target`, or `None` when this
    /// default pins nothing for it.
    pub(super) fn bundle_line(&self, target: &str) -> Option<String> {
        let (_, digest) = self.sha256.iter().find(|(triple, _)| triple == target)?;
        Some(format!(
            "{} {} {digest} https://github.com/{}/releases/download/v{}/{}\n",
            self.name,
            self.version,
            self.repository,
            self.version,
            self.archive_name(target)
        ))
    }
}

/// Every entry of `DEFAULT_PLUGINS` in `source`, checked to be well formed.
///
/// The list read here must be the one the host compiles, so anything that
/// could make the two differ is refused rather than guessed at: more than one
/// `DEFAULT_PLUGINS`, an attribute such as `#[cfg]` on it or on an entry, and
/// an entry that takes fields from another (`..BASE`).
pub(super) fn parse(source: &str) -> Result<Vec<PinnedDefault>, String> {
    let file = syn::parse_file(source).map_err(|error| format!("not Rust source: {error}"))?;
    let mut lists = file.items.iter().filter_map(|item| match item {
        Item::Const(item) if item.ident == "DEFAULT_PLUGINS" => Some((&item.attrs, &*item.expr)),
        Item::Static(item) if item.ident == "DEFAULT_PLUGINS" => Some((&item.attrs, &*item.expr)),
        _ => None,
    });
    let (attrs, list) = lists.next().ok_or("no `const DEFAULT_PLUGINS`")?;
    if lists.next().is_some() {
        return Err("more than one `DEFAULT_PLUGINS`".into());
    }
    no_attributes(attrs, "`DEFAULT_PLUGINS`")?;
    let mut defaults = Vec::new();
    for entry in array(list, "DEFAULT_PLUGINS")? {
        let default = entry_pins(entry)?;
        if defaults
            .iter()
            .any(|other: &PinnedDefault| other.name == default.name)
        {
            return Err(format!("{} is listed twice", default.name));
        }
        defaults.push(default);
    }
    Ok(defaults)
}

fn entry_pins(entry: &Expr) -> Result<PinnedDefault, String> {
    let Expr::Struct(entry) = entry else {
        return Err("an entry is not a `DefaultPlugin { .. }` literal".into());
    };
    no_attributes(&entry.attrs, "an entry")?;
    for field in &entry.fields {
        no_attributes(&field.attrs, "an entry's field")?;
    }
    if entry.rest.is_some() || entry.dot2_token.is_some() {
        return Err("an entry takes fields from another (`..`); spell every pin out".into());
    }
    let field = |name: &str| {
        entry
            .fields
            .iter()
            .find(|field| matches!(&field.member, Member::Named(ident) if ident == name))
            .map(|field| &field.expr)
            .ok_or_else(|| format!("an entry has no `{name}`"))
    };
    let name = string(field("name")?, "name")?;
    let in_entry = |error: String| format!("{name}: {error}");
    let version = string(field("version")?, "version").map_err(in_entry)?;
    let repository = string(field("repository")?, "repository").map_err(in_entry)?;
    let mut sha256 = Vec::new();
    for pin in array(field("sha256").map_err(in_entry)?, "sha256").map_err(in_entry)? {
        let pair: Vec<&Expr> = match pin {
            Expr::Tuple(pin) => pin.elems.iter().collect(),
            _ => Vec::new(),
        };
        let [triple, digest] = pair.as_slice() else {
            return Err(in_entry(
                "a sha256 pin is not a (target, digest) pair".into(),
            ));
        };
        sha256.push((
            string(triple, "target").map_err(in_entry)?,
            string(digest, "sha256").map_err(in_entry)?,
        ));
    }
    let default = PinnedDefault {
        name,
        version,
        repository,
        sha256,
    };
    check(&default).map_err(|error| format!("{}: {error}", default.name))?;
    Ok(default)
}

/// Refuses any attribute but a doc comment: a `#[cfg]` or `#[cfg_attr]` would
/// make the list read here differ from the one compiled for some target.
fn no_attributes(attrs: &[Attribute], what: &str) -> Result<(), String> {
    match attrs.iter().find(|attr| !attr.path().is_ident("doc")) {
        Some(attr) => Err(format!(
            "{what} carries an attribute (#[{}]); the release cannot tell which targets it applies to",
            attr.path()
                .get_ident()
                .map_or_else(|| "..".to_owned(), ToString::to_string)
        )),
        None => Ok(()),
    }
}

/// The elements of `[..]` or `&[..]`.
fn array<'a>(expr: &'a Expr, what: &str) -> Result<Vec<&'a Expr>, String> {
    let expr = match expr {
        Expr::Reference(reference) => &*reference.expr,
        other => other,
    };
    match expr {
        Expr::Array(array) => Ok(array.elems.iter().collect()),
        _ => Err(format!("`{what}` is not a literal list")),
    }
}

fn string(expr: &Expr, what: &str) -> Result<String, String> {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Str(value) => Ok(value.value()),
            _ => Err(format!("`{what}` is not a string literal")),
        },
        _ => Err(format!("`{what}` is not a string literal")),
    }
}

/// The values end up in a URL and in file names, so each is held to the
/// characters it may contain.
fn check(default: &PinnedDefault) -> Result<(), String> {
    let word = |value: &str, extra: &str| {
        !value.is_empty()
            && !value.starts_with(['.', '-'])
            && value
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || extra.contains(ch))
    };
    if !word(&default.name, "-_") {
        return Err(format!("name {:?} is not a plugin name", default.name));
    }
    if !word(&default.version, ".-+") {
        return Err(format!("version {:?} is not a version", default.version));
    }
    let repository_ok = default
        .repository
        .split_once('/')
        .is_some_and(|(owner, name)| word(owner, "-") && word(name, "-_."));
    if !repository_ok {
        return Err(format!(
            "repository {:?} is not a GitHub owner/name",
            default.repository
        ));
    }
    for (index, (triple, digest)) in default.sha256.iter().enumerate() {
        if !word(triple, "-_") {
            return Err(format!("{triple:?} is not a target triple"));
        }
        if default.sha256[..index]
            .iter()
            .any(|(other, _)| other == triple)
        {
            return Err(format!("{triple} is pinned twice"));
        }
        let is_hex = digest.len() == 64
            && digest
                .chars()
                .all(|ch| ch.is_ascii_digit() || ('a'..='f').contains(&ch));
        if !is_hex {
            return Err(format!(
                "the {triple} pin {digest:?} is not a lowercase hex SHA-256"
            ));
        }
    }
    Ok(())
}

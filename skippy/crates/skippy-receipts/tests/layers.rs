//! The layers of `skippy-receipts`, checked against the source.
//!
//! `src/lib.rs` puts every top-level module under a `// Layer:` comment:
//! shared, node, client or test support. This test reads those comments
//! and every file under `src/`, and fails when a module names, through a
//! `crate::` path, a module of a layer that it may not use:
//!
//! - a shared module uses shared modules only;
//! - a node module uses shared and node modules;
//! - a client module uses shared and client modules;
//! - test support and test code use every layer.
//!
//! Test code is a module declared under a `cfg` that names `test` (a
//! `tests.rs` file, an inline `mod tests`, the `test-support` feature), and
//! everything inside it. Comments are not code, so a doc comment may point
//! at any module.
//!
//! The check reads `crate::` paths. A `super::` path that leaves its
//! top-level module would get around it, so that fails too.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layer {
    Shared,
    Node,
    Client,
    TestSupport,
}

impl Layer {
    /// The layer a `// Layer:` comment of `src/lib.rs` names.
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "shared" => Self::Shared,
            "node" => Self::Node,
            "client" => Self::Client,
            "test support" => Self::TestSupport,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Node => "node",
            Self::Client => "client",
            Self::TestSupport => "test support",
        }
    }

    /// Whether a module of this layer may name a module of `other`.
    fn may_use(self, other: Self) -> bool {
        match self {
            Self::Shared => other == Self::Shared,
            Self::Node => matches!(other, Self::Shared | Self::Node),
            Self::Client => matches!(other, Self::Shared | Self::Client),
            Self::TestSupport => true,
        }
    }

    fn rule(self) -> &'static str {
        match self {
            Self::Shared => "A shared module may use shared modules only.",
            Self::Node => "A node module may use shared and node modules.",
            Self::Client => "A client module may use shared and client modules.",
            Self::TestSupport => "Test support may use every layer.",
        }
    }
}

/// The layer of each top-level module, from the `// Layer:` comments of
/// `src/lib.rs`.
fn layers_of(lib: &str) -> Result<BTreeMap<String, Layer>, String> {
    let mut layers = BTreeMap::new();
    let mut current = None;
    for line in lib.lines() {
        if let Some(name) = line.trim().strip_prefix("// Layer:") {
            let name = name.trim();
            let layer = Layer::named(name)
                .ok_or_else(|| format!("src/lib.rs: `// Layer: {name}` names no layer"))?;
            current = Some(layer);
        } else if let Some((module, _)) = mod_declaration(line) {
            let layer = current.ok_or_else(|| {
                format!("src/lib.rs: `mod {module}` is above the first `// Layer:` comment")
            })?;
            layers.insert(module.to_owned(), layer);
        }
    }
    Ok(layers)
}

/// The name of the module a line declares, and whether its body follows
/// inline.
fn mod_declaration(line: &str) -> Option<(&str, bool)> {
    let line = line.trim();
    let line = match line.strip_prefix("pub") {
        Some(rest) if rest.starts_with('(') => rest.split_once(')')?.1.trim_start(),
        Some(rest) if rest.starts_with(' ') => rest.trim_start(),
        _ => line,
    };
    let rest = line.strip_prefix("mod ")?;
    let end = rest
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    let (name, tail) = rest.split_at(end);
    match tail.trim() {
        ";" => Some((name, false)),
        tail if tail.starts_with('{') && !name.is_empty() => Some((name, true)),
        _ => None,
    }
}

/// A `mod name;` line of a file.
#[derive(Debug, PartialEq, Eq)]
struct ChildModule {
    name: String,
    /// The file a `#[path]` attribute gives it.
    path: Option<String>,
    /// Declared under a `cfg` that names `test`, or inside a test module.
    test: bool,
}

/// What the check reads of one source file.
#[derive(Debug, PartialEq, Eq)]
struct Source {
    /// The file up to its inline test module, without comments.
    code: String,
    children: Vec<ChildModule>,
}

fn read_source(text: &str) -> Source {
    let mut code = String::new();
    let mut children = Vec::new();
    let mut in_tests = false;
    let (mut cfg_test, mut path) = (false, None);
    for line in text.lines() {
        let line = without_comment(line);
        let trimmed = line.trim();
        if trimmed.starts_with("#[") {
            cfg_test |= trimmed.starts_with("#[cfg(") && trimmed.contains("test");
            if let Some(file) = trimmed.strip_prefix("#[path = \"") {
                path = file.split('"').next().map(str::to_owned);
            }
        } else if !trimmed.is_empty() {
            match mod_declaration(trimmed) {
                Some((name, false)) => children.push(ChildModule {
                    name: name.to_owned(),
                    path: path.take(),
                    test: in_tests || cfg_test,
                }),
                Some((_, true)) => in_tests |= cfg_test,
                None => {}
            }
            (cfg_test, path) = (false, None);
        }
        // every line is kept, so that a position in `code` is a line of the file
        if !in_tests {
            code.push_str(line);
        }
        code.push('\n');
    }
    Source { code, children }
}

/// The line up to its `//` comment. A `//` inside a string literal is kept.
fn without_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut previous = '\0';
    for (at, c) in line.char_indices() {
        match c {
            '"' if previous != '\\' => in_string = !in_string,
            '/' if previous == '/' && !in_string => return &line[..at - 1],
            _ => {}
        }
        previous = c;
    }
    line
}

/// A path that leaves its module.
#[derive(Debug, PartialEq, Eq)]
enum Reference {
    /// `crate::<name>`.
    Crate(String),
    /// A `super::` chain of this length.
    Super(usize),
}

/// Every `crate::` path and `super::` chain of `code`, with its line.
fn references(code: &str) -> Vec<(usize, Reference)> {
    let line_of = |at: usize| code[..at].matches('\n').count() + 1;
    let starts_a_path = |at: usize| !code[..at].ends_with(is_name_char);
    let mut found = Vec::new();
    for (at, keyword) in code.match_indices("crate::") {
        if starts_a_path(at) {
            let path = at + keyword.len();
            for (offset, name) in first_segments(&code[path..]) {
                found.push((line_of(path + offset), Reference::Crate(name)));
            }
        }
    }
    for (at, keyword) in code.match_indices("super::") {
        if starts_a_path(at) && !code[..at].ends_with(keyword) {
            let chain = code[at..]
                .split(keyword)
                .take_while(|s| s.is_empty())
                .count();
            found.push((line_of(at), Reference::Super(chain)));
        }
    }
    found.sort_by_key(|(line, _)| *line);
    found
}

fn is_name_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The first segment of each path that `path` starts, with where it is in
/// `path`: one for `a::b`, one per branch for a group `{a::b, c}`.
fn first_segments(path: &str) -> Vec<(usize, String)> {
    let Some(group) = path.strip_prefix('{') else {
        return vec![(0, first_name(path))];
    };
    let mut segments = Vec::new();
    let (mut depth, mut start) = (0usize, 0);
    let mut branch = |from: usize, to: usize| {
        let text = &group[from..to];
        let skipped = text.len() - text.trim_start().len();
        if !text.trim().is_empty() {
            segments.push((1 + from + skipped, first_name(text.trim_start())));
        }
    };
    for (at, c) in group.char_indices() {
        match c {
            '{' => depth += 1,
            '}' if depth == 0 => {
                branch(start, at);
                return segments;
            }
            '}' => depth -= 1,
            ',' if depth == 0 => {
                branch(start, at);
                start = at + 1;
            }
            _ => {}
        }
    }
    branch(start, group.len());
    segments
}

fn first_name(path: &str) -> String {
    path.chars().take_while(|c| is_name_char(*c)).collect()
}

/// One module file as the check walks it.
struct Module<'a> {
    /// Its file, relative to `src/`.
    file: &'a Path,
    /// Its top-level module, and that module's layer.
    top: &'a str,
    layer: Layer,
    /// How many modules deep it is: 1 for a top-level module.
    depth: usize,
}

/// The paths of `code` that the layer of `module` may not hold.
fn violations(module: &Module<'_>, code: &str, layers: &BTreeMap<String, Layer>) -> Vec<String> {
    let Module {
        file,
        top,
        layer,
        depth,
    } = module;
    let mut found = Vec::new();
    for (line, reference) in references(code) {
        let at = format!("src/{}:{line}", file.display());
        match reference {
            Reference::Crate(name) => match layers.get(&name) {
                Some(other) if layer.may_use(*other) => {}
                Some(other) => found.push(format!(
                    "{at}: `crate::{name}`. `{top}` is in the {} layer and `{name}` is in the {} \
                     layer. {}",
                    layer.name(),
                    other.name(),
                    layer.rule()
                )),
                None => found.push(format!(
                    "{at}: `crate::{name}` is not a module that src/lib.rs puts in a layer. Name \
                     the module that owns the item."
                )),
            },
            Reference::Super(chain) if chain >= *depth => found.push(format!(
                "{at}: a `super::` path leaves `{top}`. Write it from `crate::`, so that this \
                 check can read it."
            )),
            Reference::Super(_) => {}
        }
    }
    found
}

/// The file of a child module of `parent`.
fn child_file(src: &Path, parent: &Path, own_dir: bool, child: &ChildModule) -> Option<PathBuf> {
    let parent_dir = parent.parent().unwrap_or(Path::new(""));
    if let Some(path) = &child.path {
        return Some(parent_dir.join(path));
    }
    let dir = if own_dir {
        parent_dir.to_path_buf()
    } else {
        parent_dir.join(parent.file_stem()?)
    };
    [
        dir.join(format!("{}.rs", child.name)),
        dir.join(&child.name).join("mod.rs"),
    ]
    .into_iter()
    .find(|file| src.join(file).is_file())
}

/// Walks the module tree of the crate at `src` and collects what breaks
/// the layering.
struct Walk<'a> {
    src: &'a Path,
    layers: &'a BTreeMap<String, Layer>,
    visited: BTreeSet<PathBuf>,
    problems: Vec<String>,
}

impl Walk<'_> {
    /// Reads `module` and the modules it declares. `own_dir` says that its
    /// children are in its own directory, as for `lib.rs` and a file that a
    /// `#[path]` names. `test` says that it is test code.
    fn module(&mut self, module: &Module<'_>, own_dir: bool, test: bool) {
        let Ok(text) = fs::read_to_string(self.src.join(module.file)) else {
            self.problems
                .push(format!("src/{} cannot be read", module.file.display()));
            return;
        };
        self.visited.insert(module.file.to_path_buf());
        let source = read_source(&text);
        if !test && module.depth > 0 {
            self.problems
                .extend(violations(module, &source.code, self.layers));
        }
        for child in &source.children {
            let Some(file) = child_file(self.src, module.file, own_dir, child) else {
                self.problems.push(format!(
                    "src/{}: the file of `mod {}` was not found",
                    module.file.display(),
                    child.name
                ));
                continue;
            };
            let (top, layer) = if module.depth == 0 {
                match self.layers.get(&child.name) {
                    Some(layer) => (child.name.as_str(), *layer),
                    None => {
                        self.problems
                            .push(format!("src/lib.rs: `mod {}` has no layer", child.name));
                        continue;
                    }
                }
            } else {
                (module.top, module.layer)
            };
            let child_module = Module {
                file: &file,
                top,
                layer,
                depth: module.depth + 1,
            };
            self.module(&child_module, child.path.is_some(), test || child.test);
        }
    }
}

/// Every `.rs` file under `dir`, relative to `src`.
fn rust_files(src: &Path, dir: &Path, files: &mut BTreeSet<PathBuf>) {
    let entries = fs::read_dir(src.join(dir)).expect("read a source directory");
    for entry in entries {
        let name = entry.expect("read a source directory entry").file_name();
        let path = dir.join(name);
        if src.join(&path).is_dir() {
            rust_files(src, &path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.insert(path);
        }
    }
}

/// What breaks the layering of the crate at `src`.
fn problems_of(src: &Path) -> Vec<String> {
    let lib = fs::read_to_string(src.join("lib.rs")).expect("read src/lib.rs");
    let layers = match layers_of(&lib) {
        Ok(layers) => layers,
        Err(problem) => return vec![problem],
    };
    let mut walk = Walk {
        src,
        layers: &layers,
        visited: BTreeSet::new(),
        problems: Vec::new(),
    };
    let root = Module {
        file: Path::new("lib.rs"),
        top: "",
        layer: Layer::TestSupport,
        depth: 0,
    };
    walk.module(&root, true, false);
    let mut files = BTreeSet::new();
    rust_files(src, Path::new(""), &mut files);
    for file in files.difference(&walk.visited) {
        walk.problems.push(format!(
            "src/{}: no `mod` line of the crate leads to this file, so its layer is unknown",
            file.display()
        ));
    }
    walk.problems
}

#[test]
fn every_module_stays_in_its_layer() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let problems = problems_of(&src);
    assert!(
        problems.is_empty(),
        "skippy-receipts: {} problem(s) with the layers that src/lib.rs lists.\n\n{}\n\nMove \
         what both sides need into a shared module, or put the module under the `// Layer:` \
         comment of its layer.\n",
        problems.len(),
        problems.join("\n")
    );
}

fn layers() -> BTreeMap<String, Layer> {
    let lib = "\
// Layer: shared
pub mod wire;
// Layer: node
pub mod recorder;
// Layer: client
pub mod audit;
pub mod weights;
// Layer: test support
#[cfg(any(test, feature = \"test-support\"))]
pub mod toy_turn;
";
    layers_of(lib).expect("layers")
}

/// What the check says of `text` as the file of a module of `top`.
fn check(top: &str, file: &str, depth: usize, text: &str) -> Vec<String> {
    let layers = layers();
    let module = Module {
        file: Path::new(file),
        top,
        layer: layers[top],
        depth,
    };
    violations(&module, &read_source(text).code, &layers)
}

#[test]
fn lib_rs_gives_every_module_a_layer() {
    let layers = layers();
    assert_eq!(layers["wire"], Layer::Shared);
    assert_eq!(layers["recorder"], Layer::Node);
    assert_eq!(layers["weights"], Layer::Client);
    assert_eq!(layers["toy_turn"], Layer::TestSupport);
    let unplaced = layers_of("pub mod wire;\n// Layer: shared\n").unwrap_err();
    assert!(
        unplaced.contains("`mod wire` is above the first"),
        "{unplaced}"
    );
    let unknown = layers_of("// Layer: server\npub mod wire;\n").unwrap_err();
    assert!(unknown.contains("names no layer"), "{unknown}");
}

#[test]
fn a_path_into_a_layer_a_module_may_not_use_is_found() {
    let client = check("audit", "audit.rs", 1, "use crate::recorder::Recorder;\n");
    assert_eq!(client.len(), 1, "{client:?}");
    assert!(
        client[0].starts_with("src/audit.rs:1: `crate::recorder`. `audit` is in the client layer")
            && client[0].contains("`recorder` is in the node layer"),
        "{client:?}"
    );
    // in a group, in an expression, and through a macro's `$crate`
    for (line, text) in [
        (
            3,
            "use crate::{\n    wire::Receipt,\n    recorder::{Recorder, TurnId},\n};\n",
        ),
        (
            2,
            "fn f() {\n    let _ = crate::recorder::DEFAULT_WINDOW_BYTES;\n}\n",
        ),
        (
            3,
            "macro_rules! m {\n    () => {\n        $crate::recorder::Recorder\n    };\n}\n",
        ),
    ] {
        let found = check("audit", "audit.rs", 1, text);
        assert_eq!(found.len(), 1, "{text}: {found:?}");
        let expected = format!("src/audit.rs:{line}: `crate::recorder`");
        assert!(found[0].starts_with(&expected), "{found:?}");
    }
    // every direction that the rules close
    assert_eq!(
        check(
            "recorder",
            "recorder.rs",
            1,
            "use crate::weights::WeightFile;\n"
        )
        .len(),
        1
    );
    assert_eq!(
        check("wire", "wire.rs", 1, "use crate::recorder::Recorder;\n").len(),
        1
    );
    assert_eq!(
        check("wire", "wire.rs", 1, "use crate::audit::Profile;\n").len(),
        1
    );
    assert_eq!(
        check("audit", "audit.rs", 1, "use crate::toy_turn::Turn;\n").len(),
        1
    );
}

#[test]
fn a_path_that_the_rules_allow_passes() {
    for (top, text) in [
        ("wire", "use crate::wire::Receipt;\n"),
        (
            "recorder",
            "use crate::wire::Receipt;\nuse crate::recorder::Recorder;\n",
        ),
        (
            "audit",
            "use crate::{wire::Receipt, weights::WeightFile};\n",
        ),
        (
            "toy_turn",
            "use crate::recorder::Recorder;\nuse crate::audit::check_rows;\n",
        ),
    ] {
        let found = check(top, "module.rs", 1, text);
        assert!(found.is_empty(), "{top}: {found:?}");
    }
}

#[test]
fn comments_and_test_modules_are_not_checked() {
    let text = "\
//! Opens what [`crate::recorder`] committed.
/// See [`crate::recorder::Recorder`].
fn f() {} // unlike crate::recorder
const URL: &str = \"https://example.invalid\"; // crate::recorder

#[cfg(test)]
mod tests {
    use crate::recorder::Recorder;
}
";
    let found = check("audit", "audit.rs", 1, text);
    assert!(found.is_empty(), "{found:?}");
    // a string keeps its `//`, and what follows it on the line is code
    let text = "const A: &str = \"//\"; use crate::recorder::Recorder;\n";
    assert_eq!(check("audit", "audit.rs", 1, text).len(), 1);
}

#[test]
fn a_name_that_is_no_module_or_a_path_around_the_check_is_found() {
    let unknown = check("audit", "audit.rs", 1, "use crate::Recorder;\n");
    assert!(
        unknown[0].contains("is not a module that src/lib.rs puts in a layer"),
        "{unknown:?}"
    );
    // from `audit/pairs.rs`, one `super` is `audit` and two are the crate
    assert!(check("audit", "audit/pairs.rs", 2, "use super::Profile;\n").is_empty());
    let around = check(
        "audit",
        "audit/pairs.rs",
        2,
        "use super::super::recorder::Recorder;\n",
    );
    assert!(
        around[0].contains("a `super::` path leaves `audit`"),
        "{around:?}"
    );
    assert_eq!(
        check("audit", "audit.rs", 1, "use super::recorder::Recorder;\n").len(),
        1
    );
}

#[test]
fn a_file_declares_its_child_modules() {
    let source = read_source(
        "\
mod header;
pub mod limits;
/// Small GGUF files in memory, for tests.
#[cfg(test)]
pub(crate) mod test_gguf;
#[cfg(test)]
#[path = \"recorder/tests.rs\"]
mod tests;
",
    );
    let child = |name: &str, path: Option<&str>, test| ChildModule {
        name: name.to_owned(),
        path: path.map(str::to_owned),
        test,
    };
    assert_eq!(
        source.children,
        vec![
            child("header", None, false),
            child("limits", None, false),
            child("test_gguf", None, true),
            child("tests", Some("recorder/tests.rs"), true),
        ]
    );
}

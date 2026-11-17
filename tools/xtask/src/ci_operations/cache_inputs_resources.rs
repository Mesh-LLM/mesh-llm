//! Production module/include admission; deliberately not a Rust compiler or cfg evaluator.
use super::Inventory;
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use syn::parse::Parser;
use syn::visit::Visit;

pub(super) fn check(root: &Path, inventory: Inventory) -> DynResult<()> {
    let root = root.canonicalize()?;
    let inputs = inventory.inputs();
    let patterns = inputs
        .iter()
        .map(|p| glob::Pattern::new(p))
        .collect::<Result<Vec<_>, _>>()?;
    for source in inputs.iter().filter_map(|p| p.strip_suffix("/src/**")) {
        let package = root.join(source);
        let manifest: toml::Value =
            toml::from_str(&std::fs::read_to_string(package.join("Cargo.toml"))?)?;
        let mut roots = BTreeSet::new();
        let name = manifest
            .get("package")
            .and_then(|v| v.get("name"))
            .and_then(toml::Value::as_str)
            .ok_or("package name missing")?;
        let binary = matches!(
            (inventory, name),
            (
                Inventory::HostedXtask | Inventory::LinuxRuntime | Inventory::WindowsAbi,
                "xtask"
            ) | (Inventory::HostedController, "ci-controller")
                | (Inventory::LinuxRuntime, "skippy-package-builder")
        );
        for (table, default) in [("lib", "src/lib.rs"), ("bin", "src/main.rs")] {
            if table == "bin" && !binary {
                continue;
            }
            let declared = manifest.get(table);
            if table == "bin" {
                if let Some(bins) = declared.and_then(toml::Value::as_array) {
                    for bin in bins {
                        if bin.get("name").and_then(toml::Value::as_str) != Some(name) {
                            continue;
                        }
                        // Feature-gated auxiliary executables are not built by these inventories.
                        if bin.get("required-features").is_some() {
                            continue;
                        }
                        if let Some(path) = bin.get("path").and_then(toml::Value::as_str) {
                            roots.insert(package.join(path));
                        }
                    }
                }
            } else if let Some(path) = declared
                .and_then(|v| v.get("path"))
                .and_then(toml::Value::as_str)
            {
                roots.insert(package.join(path));
            }
            if package.join(default).is_file() {
                roots.insert(package.join(default));
            }
        }
        let mut visited = BTreeSet::new();
        let build = manifest.get("package").and_then(|value| value.get("build"));
        if build.and_then(toml::Value::as_bool) != Some(false) {
            let path = package.join(build.and_then(toml::Value::as_str).unwrap_or("build.rs"));
            if path.is_file() {
                roots.insert(path);
            }
        }
        for file in roots {
            audit_file(&root, &package, &file, &patterns, &mut visited)?;
        }
    }
    Ok(())
}

fn covered(root: &Path, path: &Path, patterns: &[glob::Pattern]) -> DynResult<()> {
    let path = path.canonicalize()?;
    let relative = path
        .strip_prefix(root)
        .map_err(|_| format!("{}: include escapes repository", path.display()))?;
    if patterns.iter().any(|p| p.matches_path(relative)) {
        return Ok(());
    }
    Err(format!(
        "{}: compiled resource is absent from cache inventory; declare its owning input",
        relative.display()
    )
    .into())
}

fn audit_file(
    root: &Path,
    package: &Path,
    file: &Path,
    patterns: &[glob::Pattern],
    visited: &mut BTreeSet<(PathBuf, bool)>,
) -> DynResult<()> {
    audit_rust_file(root, package, file, patterns, visited, false)
}

fn audit_rust_file(
    root: &Path,
    package: &Path,
    file: &Path,
    patterns: &[glob::Pattern],
    visited: &mut BTreeSet<(PathBuf, bool)>,
    included: bool,
) -> DynResult<()> {
    let file = file.canonicalize()?;
    if !visited.insert((file.clone(), included)) {
        return Ok(());
    }
    covered(root, &file, patterns)?;
    let source = std::fs::read_to_string(&file)?;
    let parent = file.parent().ok_or("module parent missing")?;
    let directory = if included
        || matches!(
            file.file_name().and_then(|s| s.to_str()),
            Some("lib.rs" | "main.rs" | "mod.rs")
        ) {
        parent.to_path_buf()
    } else {
        parent.join(file.file_stem().ok_or("module stem missing")?)
    };
    let mut audit = Audit {
        root,
        package,
        file: &file,
        directory,
        inline_module: false,
        patterns,
        visited,
        error: None,
    };
    match syn::parse_file(&source) {
        Ok(syntax) => audit.visit_file(&syntax),
        Err(_) if included => audit.visit_expr(&syn::parse_str::<syn::Expr>(&source)?),
        Err(error) => return Err(error.into()),
    }
    match audit.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

struct Audit<'a> {
    root: &'a Path,
    package: &'a Path,
    file: &'a Path,
    directory: PathBuf,
    inline_module: bool,
    patterns: &'a [glob::Pattern],
    visited: &'a mut BTreeSet<(PathBuf, bool)>,
    error: Option<crate::command::DynError>,
}

fn test_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr
                .parse_args::<syn::Meta>()
                .is_ok_and(|meta| requires_test(&meta))
    })
}

fn requires_test(meta: &syn::Meta) -> bool {
    match meta {
        syn::Meta::Path(path) => path.is_ident("test"),
        syn::Meta::List(list) if list.path.is_ident("all") => list
            .parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
            .is_ok_and(|items| items.iter().any(requires_test)),
        syn::Meta::List(list) if list.path.is_ident("any") => list
            .parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
            .is_ok_and(|items| !items.is_empty() && items.iter().all(requires_test)),
        _ => false,
    }
}

impl<'ast> Visit<'ast> for Audit<'_> {
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        if !test_only(&item.attrs) {
            syn::visit::visit_impl_item_fn(self, item);
        }
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        if !test_only(&item.attrs) {
            syn::visit::visit_trait_item_fn(self, item);
        }
    }

    fn visit_item(&mut self, item: &'ast syn::Item) {
        let attrs: &[syn::Attribute] = match item {
            syn::Item::Const(v) => &v.attrs,
            syn::Item::Enum(v) => &v.attrs,
            syn::Item::Fn(v) => &v.attrs,
            syn::Item::Impl(v) => &v.attrs,
            syn::Item::Macro(v) => &v.attrs,
            syn::Item::Mod(v) => &v.attrs,
            syn::Item::Static(v) => &v.attrs,
            syn::Item::Struct(v) => &v.attrs,
            syn::Item::Trait(v) => &v.attrs,
            syn::Item::Type(v) => &v.attrs,
            _ => &[],
        };
        if !test_only(attrs) && self.error.is_none() {
            syn::visit::visit_item(self, item);
        }
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        let explicit = match module_path(module) {
            Ok(path) => path,
            Err(error) => {
                self.error = Some(error);
                return;
            }
        };
        if let Some((_, items)) = &module.content {
            let previous = self.directory.clone();
            let previous_inline = self.inline_module;
            if let Some(path) = &explicit {
                let base = if self.inline_module {
                    &self.directory
                } else {
                    self.file.parent().unwrap_or(self.root)
                };
                self.directory = base.join(path);
            } else {
                self.directory.push(module.ident.to_string());
            }
            self.inline_module = true;
            for item in items {
                self.visit_item(item);
            }
            self.directory = previous;
            self.inline_module = previous_inline;
            return;
        }
        let result = (|| -> DynResult<()> {
            let file =
                if let Some(path) = &explicit {
                    let base = if self.inline_module {
                        &self.directory
                    } else {
                        self.file.parent().ok_or("module parent missing")?
                    };
                    base.join(path)
                } else {
                    let name = module.ident.to_string();
                    let flat = self.directory.join(format!("{name}.rs"));
                    let nested = self.directory.join(name).join("mod.rs");
                    match (flat.is_file(), nested.is_file()) {
                        (true, false) => flat,
                        (false, true) => nested,
                        _ => return Err(format!(
                            "{}: module {} must resolve to one source; declare inventory ownership",
                            self.file.display(),
                            module.ident
                        )
                        .into()),
                    }
                };
            audit_rust_file(
                self.root,
                self.package,
                &file,
                self.patterns,
                self.visited,
                explicit.is_some(),
            )
        })();
        if let Err(error) = result {
            self.error = Some(error);
        }
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        let Some(rust_source) = include_macro(&mac.path) else {
            self.visit_nested_tokens(mac.tokens.clone());
            return;
        };
        let result = (|| -> DynResult<()> {
            let expr = syn::parse2::<syn::Expr>(mac.tokens.clone())?;
            match include_path(&expr)? {
                Include::Relative(path) => self.audit_include(
                    &self
                        .file
                        .parent()
                        .ok_or("include parent missing")?
                        .join(path),
                    rust_source,
                ),
                Include::Manifest(path) => self.audit_include(
                    &self.package.join(path.trim_start_matches('/')),
                    rust_source,
                ),
                Include::Generated => {
                    // Reviewed generator boundary: prost reads the complete admitted proto tree.
                    if rust_source && self.package.ends_with("skippy/crates/skippy-protocol") {
                        covered(self.root, &self.package.join("build.rs"), self.patterns)?;
                        covered(
                            self.root,
                            &self.package.join("proto/stage.proto"),
                            self.patterns,
                        )
                    } else {
                        Err("OUT_DIR include requires an explicit reviewed generator/resource boundary".into())
                    }
                }
            }
        })();
        if let Err(error) = result {
            self.error = Some(
                format!(
                    "{}: include resource ownership: {error}",
                    self.file.display()
                )
                .into(),
            );
        }
    }
}

fn module_path(module: &syn::ItemMod) -> DynResult<Option<String>> {
    for attr in &module.attrs {
        reject_conditional_path(&attr.meta)?;
    }
    let Some(attr) = module
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("path"))
    else {
        return Ok(None);
    };
    let syn::Meta::NameValue(value) = &attr.meta else {
        return Err("unsupported module path attribute".into());
    };
    let syn::Expr::Lit(value) = &value.value else {
        return Err("module path must be literal".into());
    };
    let syn::Lit::Str(value) = &value.lit else {
        return Err("module path must be string".into());
    };
    Ok(Some(value.value()))
}

fn reject_conditional_path(meta: &syn::Meta) -> DynResult<()> {
    let syn::Meta::List(list) = meta else {
        return Ok(());
    };
    if !list.path.is_ident("cfg_attr") {
        return Ok(());
    }
    let attributes = list.parse_args_with(
        syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
    )?;
    for attribute in attributes.iter().skip(1) {
        if attribute.path().is_ident("path") {
            return Err("conditional module path requires explicit resource ownership".into());
        }
        reject_conditional_path(attribute)?;
    }
    Ok(())
}

impl Audit<'_> {
    fn audit_include(&mut self, path: &Path, rust_source: bool) -> DynResult<()> {
        if rust_source {
            audit_rust_file(
                self.root,
                self.package,
                path,
                self.patterns,
                self.visited,
                true,
            )
        } else {
            covered(self.root, path, self.patterns)
        }
    }

    fn visit_nested_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        use proc_macro2::TokenTree;
        let tokens: Vec<_> = tokens.into_iter().collect();
        let mut index = 0;
        while index < tokens.len() && self.error.is_none() {
            if let [
                TokenTree::Ident(name),
                TokenTree::Punct(bang),
                TokenTree::Group(_),
                ..,
            ] = &tokens[index..]
                && bang.as_char() == '!'
                && matches!(
                    name.to_string().as_str(),
                    "include" | "include_str" | "include_bytes"
                )
            {
                let nested: proc_macro2::TokenStream =
                    tokens[index..index + 3].iter().cloned().collect();
                match syn::parse2::<syn::Macro>(nested) {
                    Ok(mac) => self.visit_macro(&mac),
                    Err(error) => self.error = Some(error.into()),
                }
                index += 3;
            } else {
                if let TokenTree::Group(group) = &tokens[index] {
                    self.visit_nested_tokens(group.stream());
                }
                index += 1;
            }
        }
    }
}

fn include_macro(path: &syn::Path) -> Option<bool> {
    let mut segments = path.segments.iter();
    let first = segments.next()?;
    let name = match segments.next() {
        None => &first.ident,
        Some(last)
            if (first.ident == "std" || first.ident == "core") && segments.next().is_none() =>
        {
            &last.ident
        }
        Some(_) => return None,
    };
    match name.to_string().as_str() {
        "include" => Some(true),
        "include_str" | "include_bytes" => Some(false),
        _ => None,
    }
}

enum Include {
    Relative(String),
    Manifest(String),
    Generated,
}

fn include_path(expr: &syn::Expr) -> DynResult<Include> {
    if let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(value),
        ..
    }) = expr
    {
        return Ok(Include::Relative(value.value()));
    }
    if let syn::Expr::Macro(value) = expr
        && value.mac.path.is_ident("concat")
    {
        let parts = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
            .parse2(value.mac.tokens.clone())?;
        let mut parts = parts.iter();
        if let Some(syn::Expr::Macro(base)) = parts.next()
            && base.mac.path.is_ident("env")
        {
            let environment = syn::parse2::<syn::LitStr>(base.mac.tokens.clone())?.value();
            let mut path = String::new();
            for part in parts {
                let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(value),
                    ..
                }) = part
                else {
                    return Err(
                        "concat include suffix must be literal; declare resource ownership".into(),
                    );
                };
                path.push_str(&value.value());
            }
            return match environment.as_str() {
                "CARGO_MANIFEST_DIR" => Ok(Include::Manifest(path)),
                "OUT_DIR" => Ok(Include::Generated),
                _ => Err(format!(
                    "unsupported include environment {environment}; declare resource ownership"
                )
                .into()),
            };
        }
    }
    Err("unsupported include expression; declare an explicit resource boundary".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_module_paths_require_explicit_ownership() -> DynResult<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().canonicalize()?;
        std::fs::create_dir(root.join("src"))?;
        std::fs::write(root.join("src/child.rs"), "const SAFE: u8 = 0;")?;
        std::fs::write(
            root.join("src/alternate.rs"),
            "const DATA: &str = include_str!(\"../resource.txt\");",
        )?;
        std::fs::write(root.join("resource.txt"), "data")?;
        for attribute in [
            "#[cfg_attr(unix, path=\"alternate.rs\")]",
            "#[cfg_attr(unix, cfg_attr(target_os=\"macos\", path=\"alternate.rs\"))]",
        ] {
            std::fs::write(root.join("src/main.rs"), format!("{attribute} mod child;"))?;
            let error = audit_file(
                &root,
                &root,
                &root.join("src/main.rs"),
                &[glob::Pattern::new("src/**")?],
                &mut BTreeSet::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("conditional module path"));
        }
        std::fs::write(
            root.join("src/main.rs"),
            "#[cfg_attr(unix, allow(dead_code))] mod child;",
        )?;
        audit_file(
            &root,
            &root,
            &root.join("src/main.rs"),
            &[glob::Pattern::new("src/**")?],
            &mut BTreeSet::new(),
        )
    }

    #[test]
    fn module_path_attributes_audit_the_compiled_child() -> DynResult<()> {
        for (source, selected, decoy) in [
            (
                "#[path=\"loaded.rs\"] mod loaded;",
                "src/child.rs",
                "src/loaded/child.rs",
            ),
            (
                "mod wrapper { #[path=\"child.rs\"] mod child; }",
                "src/wrapper/child.rs",
                "src/child.rs",
            ),
            (
                "#[path=\"alternate\"] mod wrapper { mod child; }",
                "src/alternate/child.rs",
                "src/wrapper/child.rs",
            ),
        ] {
            let temp = tempfile::tempdir()?;
            let root = temp.path().canonicalize()?;
            for child in [selected, decoy] {
                let path = root.join(child);
                std::fs::create_dir_all(path.parent().ok_or("fixture parent")?)?;
                std::fs::write(path, "const SAFE: u8 = 0;")?;
            }
            std::fs::write(root.join("src/main.rs"), source)?;
            std::fs::write(root.join("src/loaded.rs"), "mod child;")?;
            std::fs::write(root.join("resource.txt"), "data")?;
            std::fs::write(
                root.join(selected),
                "const DATA: &str = include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/resource.txt\"));",
            )?;
            let error = audit_file(
                &root,
                &root,
                &root.join("src/main.rs"),
                &[glob::Pattern::new("src/**")?],
                &mut BTreeSet::new(),
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("resource.txt"),
                "{source}: {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn shared_source_is_audited_in_both_resolution_contexts() -> DynResult<()> {
        for (source, unsafe_child) in [
            ("mod shared; include!(\"shared.rs\");", "src/child.rs"),
            (
                "include!(\"shared.rs\"); mod shared;",
                "src/shared/child.rs",
            ),
        ] {
            let temp = tempfile::tempdir()?;
            let root = temp.path().canonicalize()?;
            std::fs::create_dir_all(root.join("src/shared"))?;
            std::fs::write(root.join("src/main.rs"), source)?;
            std::fs::write(root.join("src/shared.rs"), "mod child;")?;
            for child in ["src/child.rs", "src/shared/child.rs"] {
                std::fs::write(root.join(child), "const SAFE: u8 = 0;")?;
            }
            std::fs::write(root.join("resource.txt"), "data")?;
            std::fs::write(
                root.join(unsafe_child),
                "const DATA: &str = include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/resource.txt\"));",
            )?;
            let error = audit_file(
                &root,
                &root,
                &root.join("src/main.rs"),
                &[glob::Pattern::new("src/**")?],
                &mut BTreeSet::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("resource.txt"));
        }
        Ok(())
    }

    #[test]
    fn expression_includes_admit_owned_and_reject_unowned_resources() -> DynResult<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().canonicalize()?;
        std::fs::create_dir(root.join("src"))?;
        std::fs::write(
            root.join("src/main.rs"),
            "const DATA: &str = include!(\"expression.rs\");",
        )?;
        std::fs::write(
            root.join("src/expression.rs"),
            "include_str!(\"../resource.txt\")",
        )?;
        std::fs::write(root.join("resource.txt"), "data")?;
        audit_file(
            &root,
            &root,
            &root.join("src/main.rs"),
            &[
                glob::Pattern::new("src/**")?,
                glob::Pattern::new("resource.txt")?,
            ],
            &mut BTreeSet::new(),
        )?;
        let error = audit_file(
            &root,
            &root,
            &root.join("src/main.rs"),
            &[glob::Pattern::new("src/**")?],
            &mut BTreeSet::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("resource.txt"));
        Ok(())
    }

    #[test]
    fn qualified_standard_includes_require_resource_ownership() -> DynResult<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().canonicalize()?;
        std::fs::create_dir(root.join("src"))?;
        std::fs::write(root.join("resource.txt"), "data")?;
        for prefix in ["std::", "core::", "::std::", "::core::"] {
            std::fs::write(
                root.join("src/main.rs"),
                format!("const DATA: &str = {prefix}include_str!(\"../resource.txt\");"),
            )?;
            let error = audit_file(
                &root,
                &root,
                &root.join("src/main.rs"),
                &[glob::Pattern::new("src/**")?],
                &mut BTreeSet::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("resource.txt"));
        }
        Ok(())
    }

    #[test]
    fn included_modules_use_the_included_files_parent() -> DynResult<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().canonicalize()?;
        std::fs::create_dir_all(root.join("src/included"))?;
        std::fs::write(root.join("src/main.rs"), "include!(\"included.rs\");")?;
        std::fs::write(root.join("src/included.rs"), "mod child;")?;
        std::fs::write(root.join("src/included/child.rs"), "const SAFE: u8 = 0;")?;
        std::fs::write(
            root.join("src/child.rs"),
            "const DATA: &str = include_str!(\"../resource.txt\");",
        )?;
        std::fs::write(root.join("resource.txt"), "data")?;
        let error = audit_file(
            &root,
            &root,
            &root.join("src/main.rs"),
            &[glob::Pattern::new("src/**")?],
            &mut BTreeSet::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("resource.txt"));
        Ok(())
    }

    #[test]
    fn recursive_rust_includes_cannot_hide_external_resources() -> DynResult<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().canonicalize()?;
        std::fs::create_dir(root.join("src"))?;
        std::fs::write(root.join("src/main.rs"), "include!(\"included.rs\");")?;
        std::fs::write(
            root.join("src/included.rs"),
            "const DATA: &str = include_str!(\"../resource.txt\");",
        )?;
        std::fs::write(root.join("resource.txt"), "data")?;
        let error = audit_file(
            &root,
            &root,
            &root.join("src/main.rs"),
            &[glob::Pattern::new("src/**")?],
            &mut BTreeSet::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("resource.txt"));
        Ok(())
    }

    #[test]
    fn macro_arguments_cannot_hide_external_resources() -> DynResult<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().canonicalize()?;
        std::fs::create_dir(root.join("src"))?;
        std::fs::write(
            root.join("src/main.rs"),
            "fn main() { println!(\"{}\", include_str!(\"../resource.txt\")); }",
        )?;
        std::fs::write(root.join("resource.txt"), "data")?;
        let error = audit_file(
            &root,
            &root,
            &root.join("src/main.rs"),
            &[glob::Pattern::new("src/**")?],
            &mut BTreeSet::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("resource.txt"));
        Ok(())
    }

    #[test]
    fn external_include_additions_require_admission_and_admitted_mutations_remain_covered()
    -> DynResult<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().canonicalize()?;
        let package = root.join("tool");
        std::fs::create_dir_all(package.join("src"))?;
        std::fs::write(root.join("resource.txt"), "before")?;
        std::fs::write(
            package.join("src/main.rs"),
            "const DATA: &str = include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/../resource.txt\"));",
        )?;
        let source = package.join("src/main.rs");
        let patterns = vec![glob::Pattern::new("tool/src/**")?];
        assert!(audit_file(&root, &package, &source, &patterns, &mut BTreeSet::new()).is_err());
        let admitted = vec![
            glob::Pattern::new("tool/src/**")?,
            glob::Pattern::new("resource.txt")?,
        ];
        audit_file(&root, &package, &source, &admitted, &mut BTreeSet::new())?;
        std::fs::write(root.join("resource.txt"), "after")?;
        audit_file(&root, &package, &source, &admitted, &mut BTreeSet::new())?;
        std::fs::write(root.join("other.txt"), "new resource")?;
        std::fs::write(
            &source,
            "const DATA: &str = include_str!(\"../../other.txt\");",
        )?;
        assert!(audit_file(&root, &package, &source, &admitted, &mut BTreeSet::new()).is_err());
        Ok(())
    }

    #[test]
    fn test_modules_are_skipped_but_production_path_modules_are_followed() -> DynResult<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().canonicalize()?;
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join("src/main.rs"),
            "#[cfg(all(test, feature=\"parquet\"))] #[path=\"absent.rs\"] mod tests; #[path=\"production.rs\"] mod real;",
        )?;
        std::fs::write(
            root.join("src/production.rs"),
            "const DATA: &str = include_str!(unknown!());",
        )?;
        let patterns = vec![glob::Pattern::new("src/**")?];
        let error = audit_file(
            &root,
            &root,
            &root.join("src/main.rs"),
            &patterns,
            &mut BTreeSet::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unsupported include expression"));
        Ok(())
    }
}

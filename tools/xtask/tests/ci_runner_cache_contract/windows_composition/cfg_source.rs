//! Runtime platform branches remain separate from test-only platform evidence.
use super::{divergent_tokens, predicate};
use proc_macro2::TokenTree;
#[derive(Default, Clone, Copy)]
pub(super) struct Sites {
    pub(super) runtime: bool,
    pub(super) test: bool,
}
impl Sites {
    pub(super) fn merge(&mut self, other: Self) {
        self.runtime |= other.runtime;
        self.test |= other.test;
    }
    pub(super) fn any(self) -> bool {
        self.runtime || self.test
    }
}
fn test_required(meta: &syn::Meta) -> bool {
    match meta {
        syn::Meta::Path(path) => path.is_ident("test"),
        syn::Meta::List(list) if list.path.is_ident("all") || list.path.is_ident("any") => {
            use syn::parse::Parser;
            let parts = syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated
                .parse2(list.tokens.clone())
                .expect("cfg predicates");
            if list.path.is_ident("all") {
                parts.iter().any(test_required)
            } else {
                !parts.is_empty() && parts.iter().all(test_required)
            }
        }
        _ => false,
    }
}
fn test_only(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("test")
            || (attribute.path().is_ident("cfg")
                && attribute
                    .parse_args::<syn::Meta>()
                    .is_ok_and(|meta| test_required(&meta)))
    })
}
pub(super) fn classify_sites(source: &str) -> Sites {
    struct Census {
        inside_test: bool,
        sites: Sites,
    }
    impl<'ast> syn::visit::Visit<'ast> for Census {
        fn visit_macro(&mut self, node: &'ast syn::Macro) {
            if (node.path.is_ident("cfg") && predicate(node.tokens.clone()))
                || divergent_tokens(node.tokens.clone())
            {
                if self.inside_test {
                    self.sites.test = true;
                } else {
                    self.sites.runtime = true;
                }
            }
        }
        fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
            if let syn::Meta::List(list) = &attribute.meta {
                let tokens = if list.path.is_ident("cfg_attr") {
                    list.tokens
                        .clone()
                        .into_iter()
                        .take_while(
                            |token| !matches!(token, TokenTree::Punct(p) if p.as_char() == ','),
                        )
                        .collect()
                } else {
                    list.tokens.clone()
                };
                if (list.path.is_ident("cfg") || list.path.is_ident("cfg_attr"))
                    && predicate(tokens)
                {
                    if self.inside_test {
                        self.sites.test = true;
                    } else {
                        self.sites.runtime = true;
                    }
                }
            }
        }
        fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
            let previous = self.inside_test;
            self.inside_test |= test_only(&item.attrs);
            syn::visit::visit_item_fn(self, item);
            self.inside_test = previous;
        }
        fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
            let previous = self.inside_test;
            self.inside_test |= test_only(&item.attrs);
            syn::visit::visit_item_mod(self, item);
            self.inside_test = previous;
        }
        fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
            let previous = self.inside_test;
            self.inside_test |= test_only(&item.attrs);
            syn::visit::visit_impl_item_fn(self, item);
            self.inside_test = previous;
        }
    }
    let mut census = Census {
        inside_test: false,
        sites: Sites::default(),
    };
    let file = syn::parse_file(source).expect("owning Rust syntax");
    census.inside_test = test_only(&file.attrs);
    syn::visit::Visit::visit_file(&mut census, &file);
    census.sites
}

#[test]
fn platform_test_sites_do_not_claim_runtime_qualification_or_hide_new_runtime_branches() {
    for source in [
        "#[cfg(unix)] #[test] fn filesystem_refusal() {}",
        "#[cfg(test)] mod tests { #[cfg(unix)] fn filesystem_refusal() {} }",
        "#[cfg(all(test, unix))] fn filesystem_refusal() {}",
        "#[cfg(any(test, all(test, unix)))] fn filesystem_refusal() {}",
    ] {
        let test = classify_sites(source);
        assert!(test.test && !test.runtime);
        let mutated = format!(
            "{source}\n#[cfg(unix)] fn runtime_path() {{}}\n#[cfg(windows)] fn runtime_path() {{}}"
        );
        let runtime = classify_sites(&mutated);
        assert!(runtime.runtime && runtime.test);
    }
    for source in [
        "#[cfg(any(test, unix))] fn runtime_path() {}",
        "#[cfg(not(test))] fn runtime_path() { let _ = cfg!(windows); }",
        "#[cfg_attr(test, allow(dead_code))] #[cfg(unix)] fn runtime_path() {}",
        "fn runtime_path() { let _ = cfg!(windows); }",
    ] {
        assert!(classify_sites(source).runtime, "{source}");
    }
}

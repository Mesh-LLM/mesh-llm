use std::env;
use std::path::Path;

use anyhow::Result;
use clap::Parser;

use crate::direct_convert::{DirectConvertArgs, run_direct_convert};
use crate::direct_quantize::{DirectQuantizeArgs, run_direct_quantize};

pub(crate) fn dispatch_argv0_alias() -> Option<Result<()>> {
    let argv0 = env::args_os().next()?;
    let file_name = Path::new(&argv0).file_name()?.to_string_lossy();
    match argv0_alias_kind(&file_name) {
        Some(Argv0AliasKind::Quantize) => Some(run_direct_quantize(DirectQuantizeArgs::parse())),
        Some(Argv0AliasKind::Convert) => Some(run_direct_convert(DirectConvertArgs::parse())),
        None => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Argv0AliasKind {
    Convert,
    Quantize,
}

fn argv0_alias_kind(file_name: &str) -> Option<Argv0AliasKind> {
    let normalized = file_name.replace('_', "-");
    match normalized.as_str() {
        "llama-quantize" | "skippy-quantize-llama-quantize" => Some(Argv0AliasKind::Quantize),
        "convert-hf-to-gguf"
        | "convert-hf-to-gguf.py"
        | "hf-to-gguf.py"
        | "hf-to-gguff.py"
        | "skippy-quantize-convert-hf-to-gguf" => Some(Argv0AliasKind::Convert),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalizes_underscore_names_like_python_converter() {
        assert_eq!(
            "convert_hf_to_gguf.py".replace('_', "-"),
            "convert-hf-to-gguf.py"
        );
    }

    #[test]
    fn recognizes_converter_aliases() {
        for alias in [
            "convert_hf_to_gguf.py",
            "convert-hf-to-gguf",
            "hf_to_gguf.py",
            "hf_to_gguff.py",
            "skippy-quantize-convert-hf-to-gguf",
        ] {
            assert_eq!(
                super::argv0_alias_kind(alias),
                Some(super::Argv0AliasKind::Convert)
            );
        }
    }

    #[test]
    fn recognizes_quantize_aliases() {
        assert_eq!(
            super::argv0_alias_kind("llama-quantize"),
            Some(super::Argv0AliasKind::Quantize)
        );
    }
}

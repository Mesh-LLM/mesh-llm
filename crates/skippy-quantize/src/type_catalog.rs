use anyhow::Result;
use clap::Parser;
use serde::Serialize;

use crate::output::{print_info, print_json_pretty, print_success};
use crate::types::{QuantType, TensorType};

#[derive(Debug, Parser)]
pub(crate) struct TypeCatalogArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Serialize)]
struct QuantCatalog {
    whole_model_quant_modes: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct TensorTypeCatalog {
    raw_tensor_types: Vec<&'static str>,
}

pub(crate) fn list_quants(args: TypeCatalogArgs) -> Result<()> {
    let names = QuantType::ALL
        .iter()
        .map(|quant| quant.as_llama_name())
        .collect::<Vec<_>>();
    if args.json {
        print_json_pretty(&QuantCatalog {
            whole_model_quant_modes: names,
        })?;
    } else {
        print_success("Whole-model quant modes");
        for name in names {
            println!("   • {name}");
        }
        print_info("Use --tensor-type-file for custom tensor recipes");
    }
    Ok(())
}

pub(crate) fn list_tensor_types(args: TypeCatalogArgs) -> Result<()> {
    let names = TensorType::ALL
        .iter()
        .map(|tensor_type| tensor_type.as_ggml_name())
        .collect::<Vec<_>>();
    if args.json {
        print_json_pretty(&TensorTypeCatalog {
            raw_tensor_types: names,
        })?;
    } else {
        print_success("Raw tensor override types");
        for name in names {
            println!("   • {name}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogs_are_not_empty() {
        assert!(!QuantType::ALL.is_empty());
        assert!(!TensorType::ALL.is_empty());
    }
}

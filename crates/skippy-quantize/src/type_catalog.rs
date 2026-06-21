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
    known_recipe_labels: Vec<KnownRecipeLabel>,
}

#[derive(Debug, Serialize)]
struct KnownRecipeLabel {
    label: &'static str,
    base_quant: &'static str,
    note: &'static str,
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
    let labels = known_recipe_labels();
    if args.json {
        print_json_pretty(&QuantCatalog {
            whole_model_quant_modes: names,
            known_recipe_labels: labels,
        })?;
    } else {
        print_success("Whole-model quant modes");
        for name in names {
            println!("   • {name}");
        }
        print_info("Known recipe labels");
        for label in labels {
            println!(
                "   • {} -> {} ({})",
                label.label, label.base_quant, label.note
            );
        }
    }
    Ok(())
}

fn known_recipe_labels() -> Vec<KnownRecipeLabel> {
    vec![
        KnownRecipeLabel {
            label: "UD-Q3_K_S",
            base_quant: "Q3_K_S",
            note: "custom dynamic tensor-type recipe; pass the recipe with --tensor-type-file",
        },
        KnownRecipeLabel {
            label: "Q4_K_XL",
            base_quant: "Q4_K_M",
            note: "custom high-quality tensor-type recipe; pass the recipe with --tensor-type-file",
        },
    ]
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
        assert!(!known_recipe_labels().is_empty());
    }
}

use anyhow::{Result, anyhow, ensure};

use crate::types::TensorType;

pub fn ensure_tensor_type_entry(token: &str) -> Result<()> {
    normalize_tensor_type_entry(token).map(|_| ())
}

pub fn normalize_tensor_type_entry(token: &str) -> Result<String> {
    let (name, raw_type) = token
        .split_once('=')
        .ok_or_else(|| anyhow!("malformed tensor type entry {token:?}"))?;
    ensure!(!name.is_empty(), "tensor type entry has empty tensor name");
    ensure_raw_tensor_type(raw_type).map_err(|error| {
        anyhow!("unsupported raw ggml tensor type {raw_type:?} in entry {token:?}: {error}")
    })?;
    Ok(format!("{}={raw_type}", name.to_ascii_lowercase()))
}

fn ensure_raw_tensor_type(raw_type: &str) -> Result<()> {
    ensure!(
        TensorType::parse(raw_type).is_some(),
        "unsupported raw ggml tensor type {raw_type:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_tensor_type_entry() {
        assert_eq!(
            normalize_tensor_type_entry("MTP_Head.Weight=NVFP4").unwrap(),
            "mtp_head.weight=NVFP4"
        );
    }

    #[test]
    fn rejects_unknown_tensor_type() {
        assert!(normalize_tensor_type_entry("foo=NOT_A_TYPE").is_err());
    }
}

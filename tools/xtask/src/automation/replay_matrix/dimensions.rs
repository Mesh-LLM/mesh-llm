use super::Reader;
use crate::command::DynError;
use std::{fs::File, path::Path};

#[derive(Debug, thiserror::Error)]
pub(in crate::automation) enum DimensionsError {
    #[error("GGUF metadata read failed: {0}")]
    Metadata(#[source] DynError),
    #[error("invalid GGUF dimension field {field}: {reason}")]
    Field {
        field: &'static str,
        reason: &'static str,
    },
    #[error("GGUF activation width exceeds i32")]
    WidthOverflow,
}

#[derive(Debug, Eq, PartialEq)]
pub(in crate::automation) struct Dimensions {
    pub(in crate::automation) architecture: String,
    pub(in crate::automation) block_count: u64,
    pub(in crate::automation) activation_width: u64,
    pub(in crate::automation) mtp_layers: u64,
}

#[derive(Default)]
struct Projection {
    architecture: Option<String>,
    blocks: Option<(String, u64)>,
    embedding: Option<(String, u64)>,
    mtp: Option<(String, u64)>,
    hyper: Option<(String, u64)>,
    output: Option<(String, u64)>,
}

pub(in crate::automation) fn inspect(path: &Path) -> Result<Option<Dimensions>, DimensionsError> {
    let mut reader = Reader {
        file: File::open(path).map_err(|error| DimensionsError::Metadata(error.into()))?,
    };
    project(&mut reader)
}

fn project(reader: &mut Reader) -> Result<Option<Dimensions>, DimensionsError> {
    if reader.bytes::<4>().map_err(DimensionsError::Metadata)? != *b"GGUF"
        || !matches!(reader.u32().map_err(DimensionsError::Metadata)?, 2 | 3)
    {
        return Err(field("header", "requires GGUF version 2 or 3"));
    }
    reader.u64().map_err(DimensionsError::Metadata)?;
    let count = reader.u64().map_err(DimensionsError::Metadata)?;
    if count > 1_000_000 {
        return Err(field("header", "metadata count exceeds bound"));
    }
    let mut projection = Projection::default();
    for _ in 0..count {
        let key = reader.string().map_err(DimensionsError::Metadata)?;
        let kind = reader.u32().map_err(DimensionsError::Metadata)?;
        if key == "general.architecture" {
            if kind != 8 || projection.architecture.is_some() {
                return Err(field("architecture", "requires one string"));
            }
            projection.architecture = Some(reader.string().map_err(DimensionsError::Metadata)?);
        } else if let Some((name, slot)) = projection.slot(&key) {
            if slot.is_some() {
                return Err(field(name, "duplicate dimension metadata"));
            }
            let value = reader
                .integer(kind)
                .map_err(|_| field(name, "requires nonnegative integer"))?;
            *slot = Some((key, value));
        } else {
            reader.skip(kind, 0).map_err(DimensionsError::Metadata)?;
        }
    }
    projection.finish()
}

impl Projection {
    fn slot(&mut self, key: &str) -> Option<(&'static str, &mut Option<(String, u64)>)> {
        for (suffix, slot) in [
            ("block_count", &mut self.blocks),
            ("embedding_length", &mut self.embedding),
            ("nextn_predict_layers", &mut self.mtp),
            ("hyper_connection.count", &mut self.hyper),
            ("embedding_length_out", &mut self.output),
        ] {
            if key.ends_with(&format!(".{suffix}")) {
                return Some((suffix, slot));
            }
        }
        None
    }

    fn finish(self) -> Result<Option<Dimensions>, DimensionsError> {
        if self.blocks.is_none() && self.embedding.is_none() {
            if self.mtp.is_some() || self.hyper.is_some() || self.output.is_some() {
                return Err(field("block_count", "partial dimension metadata"));
            }
            return Ok(None);
        }
        let architecture = self
            .architecture
            .ok_or_else(|| field("architecture", "missing"))?;
        if architecture.is_empty()
            || !architecture.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
            })
        {
            return Err(field("architecture", "invalid architecture label"));
        }
        let blocks = qualified(self.blocks, &architecture, "block_count")?
            .ok_or_else(|| field("block_count", "missing"))?;
        let embedding = qualified(self.embedding, &architecture, "embedding_length")?
            .ok_or_else(|| field("embedding_length", "missing"))?;
        if blocks == 0 || embedding == 0 {
            return Err(field("dimensions", "must be positive"));
        }
        let mtp = qualified(self.mtp, &architecture, "nextn_predict_layers")?.unwrap_or(0);
        if mtp >= blocks {
            return Err(field(
                "nextn_predict_layers",
                "must be less than block_count",
            ));
        }
        let hyper = qualified(self.hyper, &architecture, "hyper_connection.count")?;
        let output = qualified(self.output, &architecture, "embedding_length_out")?;
        let width = match (architecture.as_str(), hyper) {
            ("qwen4exp", count) | ("dflash", count @ Some(_)) => {
                let count = count.filter(|count| *count > 0).ok_or_else(|| {
                    field("hyper_connection.count", "requires one positive count")
                })?;
                let width = embedding
                    .checked_mul(count)
                    .filter(|width| *width <= 0x7fff_ffff)
                    .ok_or(DimensionsError::WidthOverflow)?;
                if output.is_some_and(|output| output != width) {
                    return Err(field(
                        "embedding_length_out",
                        "does not match activation width",
                    ));
                }
                width
            }
            _ => embedding,
        };
        Ok(Some(Dimensions {
            architecture,
            block_count: blocks,
            activation_width: width,
            mtp_layers: mtp,
        }))
    }
}

fn qualified(
    value: Option<(String, u64)>,
    architecture: &str,
    suffix: &'static str,
) -> Result<Option<u64>, DimensionsError> {
    match value {
        Some((key, value)) if key == format!("{architecture}.{suffix}") => Ok(Some(value)),
        Some(_) => Err(field(suffix, "architecture qualifier mismatch")),
        None => Ok(None),
    }
}

fn field(field: &'static str, reason: &'static str) -> DimensionsError {
    DimensionsError::Field { field, reason }
}

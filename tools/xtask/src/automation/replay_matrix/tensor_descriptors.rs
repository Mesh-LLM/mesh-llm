//! Bounded GGUF descriptor admission without reading tensor payloads.
use super::{Reader, tensor_layouts::Layouts};
use crate::command::DynResult;
use std::{fs::File, io::Seek, path::Path};

pub(in crate::automation) fn inspect(path: &Path, layouts: &Layouts) -> DynResult<u64> {
    let mut reader = Reader {
        file: File::open(path)?,
    };
    scan(&mut reader, layouts).map_err(|error| {
        format!(
            "GGUF shard {} tensor descriptors are invalid: {error}",
            path.display()
        )
        .into()
    })
}

fn scan(reader: &mut Reader, layouts: &Layouts) -> DynResult<u64> {
    if reader.bytes::<4>()? != *b"GGUF" || !matches!(reader.u32()?, 2 | 3) {
        return Err("requires GGUF version 2 or 3".into());
    }
    let tensors = reader.u64()?;
    let metadata = reader.u64()?;
    if tensors > 1_000_000 || metadata > 1_000_000 {
        return Err("GGUF descriptor or metadata count exceeds bound".into());
    }
    for _ in 0..metadata {
        reader.string()?;
        let kind = reader.u32()?;
        reader.skip(kind, 0)?;
        budget(reader)?;
    }
    let mut total = 0_u64;
    for _ in 0..tensors {
        let name = reader.string()?;
        let size = tensor(reader, layouts).map_err(|error| format!("tensor {name:?}: {error}"))?;
        total = total.checked_add(size).ok_or("tensor byte sum overflow")?;
        budget(reader)?;
    }
    Ok(total)
}

fn budget(reader: &mut Reader) -> DynResult<()> {
    if reader.file.stream_position()? > 64 * 1024 * 1024 {
        return Err("GGUF descriptor metadata exceeds 64 MiB".into());
    }
    Ok(())
}

fn tensor(reader: &mut Reader, layouts: &Layouts) -> DynResult<u64> {
    let rank = reader.u32()?;
    if !(1..=8).contains(&rank) {
        return Err("tensor rank must be between 1 and 8".into());
    }
    let mut dimensions = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        let size = reader.u64()?;
        if size == 0 {
            return Err("tensor dimensions must be positive".into());
        }
        dimensions.push(size);
    }
    let kind = reader.u32()?;
    reader.u64()?; // Payload offset is not followed; no tensor data is read.
    let &(block, size) = layouts
        .get(&kind)
        .ok_or_else(|| format!("unknown GGML tensor type {kind}"))?;
    if block == 0 || size == 0 {
        return Err("invalid admitted GGML layout".into());
    }
    if dimensions[0] % block != 0 {
        return Err(format!(
            "first dimension {} is unaligned for GGML type {kind} block {block}",
            dimensions[0]
        )
        .into());
    }
    let initial = (dimensions[0] / block)
        .checked_mul(size)
        .ok_or("tensor byte product overflow")?;
    dimensions[1..]
        .iter()
        .try_fold(initial, |bytes, dimension| {
            bytes
                .checked_mul(*dimension)
                .ok_or_else(|| "tensor byte product overflow".into())
        })
}

#[cfg(test)]
#[path = "tensor_descriptors_tests.rs"]
mod tests;

//! Finite data admission for the prepared pin's GGML type/layout declarations.
use crate::command::DynResult;
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

pub(in crate::automation) type Layouts = BTreeMap<u32, (u64, u64)>;
const MAX_SOURCE: usize = 1024 * 1024;

pub(in crate::automation) fn read(path: &Path) -> DynResult<(Vec<u8>, Layouts)> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take((MAX_SOURCE + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_SOURCE {
        return Err("prepared GGML layout source exceeds 1 MiB".into());
    }
    let layouts = parse(std::str::from_utf8(&bytes)?)?;
    Ok((bytes, layouts))
}

fn declaration<'a>(source: &'a str, marker: &str) -> DynResult<Vec<&'a str>> {
    let positions: Vec<_> = source
        .lines()
        .enumerate()
        .filter(|(_, line)| line.trim() == marker)
        .map(|(index, _)| index)
        .collect();
    let [position] = positions.as_slice() else {
        return Err("requires one prepared GGML declaration".into());
    };
    let table = marker.starts_with("GGML_QUANT_SIZES");
    let mut body = Vec::new();
    for line in source.lines().skip(position + 1) {
        if !line.starts_with(' ') && !line.trim().is_empty() {
            if table && line.trim() != "}" {
                return Err("unterminated prepared GGML layout table".into());
            }
            return Ok(body);
        }
        let line = line.split('#').next().unwrap().trim();
        if !line.is_empty() {
            body.push(line);
        }
    }
    if table {
        Err("unterminated prepared GGML layout table".into())
    } else {
        Ok(body)
    }
}

pub(in crate::automation) fn parse(source: &str) -> DynResult<Layouts> {
    let mut types = BTreeMap::new();
    for line in declaration(source, "class GGMLQuantizationType(IntEnum):")? {
        let (name, value) = line
            .split_once('=')
            .ok_or("invalid GGML type declaration")?;
        let name = name.trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
        {
            return Err("invalid GGML type name".into());
        }
        let value = decimal(value.trim())?;
        let value = u32::try_from(value)?;
        if types.insert(name, value).is_some() {
            return Err("duplicate GGML type name".into());
        }
    }
    let qk: Vec<_> = source
        .lines()
        .filter_map(|line| line.strip_prefix("QK_K = "))
        .collect();
    let [qk] = qk.as_slice() else {
        return Err("requires one prepared QK_K declaration".into());
    };
    let qk = decimal(qk.trim())?;
    if qk == 0 {
        return Err("QK_K must be positive".into());
    }
    let mut layouts = Layouts::new();
    for line in declaration(
        source,
        "GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {",
    )? {
        let (name, tuple) = line
            .split_once(':')
            .ok_or("invalid GGML layout declaration")?;
        let name = name
            .trim()
            .strip_prefix("GGMLQuantizationType.")
            .ok_or("invalid GGML layout type")?;
        let kind = *types
            .get(name)
            .ok_or("layout refers to unknown GGML type")?;
        let tuple = tuple.trim().strip_suffix(',').unwrap_or(tuple.trim());
        let tuple = tuple
            .strip_prefix('(')
            .and_then(|v| v.strip_suffix(')'))
            .ok_or("invalid GGML layout tuple")?;
        let (block, size) = tuple
            .split_once(',')
            .ok_or("GGML layout requires block and size")?;
        let block = expression(block, qk)?;
        let size = expression(size, qk)?;
        if block == 0 || size == 0 || layouts.insert(kind, (block, size)).is_some() {
            return Err("invalid or duplicate GGML layout".into());
        }
    }
    if layouts.is_empty() || layouts.len() > 256 {
        return Err("GGML layout count outside finite bound".into());
    }
    Ok(layouts)
}

fn decimal(value: &str) -> DynResult<u64> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err("GGML layout requires unsigned decimal data".into());
    }
    Ok(value.parse()?)
}

// This accepts only the pinned table's sums/products/integer divisions, never
// calls, imports, attributes, indexing, or other executable Python expressions.
fn expression(value: &str, qk: u64) -> DynResult<u64> {
    value.split('+').try_fold(0_u64, |sum, term| {
        // The prepared table uses either products or divisions per term. Do
        // not reinterpret mixed operators with different Python precedence.
        if term.contains('*') && term.contains("//") {
            return Err("mixed GGML product/division expressions are unsupported".into());
        }
        let mut factors = term.split('*');
        let first = factor(factors.next().ok_or("missing GGML factor")?, qk)?;
        let product = factors.try_fold(first, |product, term| -> DynResult<u64> {
            product
                .checked_mul(factor(term, qk)?)
                .ok_or_else(|| "GGML layout product overflow".into())
        })?;
        sum.checked_add(product)
            .ok_or_else(|| "GGML layout sum overflow".into())
    })
}
fn factor(value: &str, qk: u64) -> DynResult<u64> {
    let mut parts = value.trim().split("//");
    let numerator = parts.next().unwrap().trim();
    let numerator = if numerator == "QK_K" {
        qk
    } else {
        decimal(numerator)?
    };
    match (parts.next(), parts.next()) {
        (None, None) => Ok(numerator),
        (Some(divisor), None) => {
            let divisor = decimal(divisor.trim())?;
            numerator
                .checked_div(divisor)
                .ok_or_else(|| "GGML layout division by zero".into())
        }
        _ => Err("unsupported GGML layout expression".into()),
    }
}

#[cfg(test)]
#[path = "tensor_layouts_tests.rs"]
mod tests;

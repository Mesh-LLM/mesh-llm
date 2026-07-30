use std::io::Write;

use anyhow::{Result, ensure};

pub(crate) const GGUF_TYPE_UINT16: u32 = 2;
pub(crate) const GGUF_TYPE_UINT32: u32 = 4;
pub(crate) const GGUF_TYPE_INT32: u32 = 5;
pub(crate) const GGUF_TYPE_FLOAT32: u32 = 6;
pub(crate) const GGUF_TYPE_BOOL: u32 = 7;
pub(crate) const GGUF_TYPE_STRING: u32 = 8;
pub(crate) const GGUF_TYPE_ARRAY: u32 = 9;
pub(crate) const GGUF_TYPE_UINT64: u32 = 10;

#[derive(Debug, Clone)]
pub(crate) enum GgufKv {
    ArrayBool { key: String, value: Vec<bool> },
    ArrayF32 { key: String, value: Vec<f32> },
    ArrayI32 { key: String, value: Vec<i32> },
    ArrayString { key: String, value: Vec<String> },
    ArrayU32 { key: String, value: Vec<u32> },
    Bool { key: String, value: bool },
    F32 { key: String, value: f32 },
    I32 { key: String, value: i32 },
    String { key: String, value: String },
    U16 { key: String, value: u16 },
    U32 { key: String, value: u32 },
    U64 { key: String, value: u64 },
}

impl GgufKv {
    pub(crate) fn array_bool(key: &str, value: Vec<bool>) -> Self {
        Self::ArrayBool {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn array_f32(key: &str, value: Vec<f32>) -> Self {
        Self::ArrayF32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn array_i32(key: &str, value: Vec<i32>) -> Self {
        Self::ArrayI32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn array_string(key: &str, value: Vec<String>) -> Self {
        Self::ArrayString {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn array_u32(key: &str, value: Vec<u32>) -> Self {
        Self::ArrayU32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn bool(key: &str, value: bool) -> Self {
        Self::Bool {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn f32(key: &str, value: f32) -> Self {
        Self::F32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn i32(key: &str, value: i32) -> Self {
        Self::I32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn string(key: &str, value: &str) -> Self {
        Self::String {
            key: key.to_string(),
            value: value.to_string(),
        }
    }

    pub(crate) fn u16(key: &str, value: u16) -> Self {
        Self::U16 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn u32(key: &str, value: u32) -> Self {
        Self::U32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn u64(key: &str, value: u64) -> Self {
        Self::U64 {
            key: key.to_string(),
            value,
        }
    }
}

pub(crate) fn write_kv<W: Write>(writer: &mut W, kv: &GgufKv) -> Result<()> {
    match kv {
        GgufKv::ArrayBool { key, value } => {
            write_array_header(writer, key, GGUF_TYPE_BOOL, value.len())?;
            for item in value {
                writer.write_all(&[*item as u8])?;
            }
        }
        GgufKv::ArrayF32 { key, value } => {
            write_array_header(writer, key, GGUF_TYPE_FLOAT32, value.len())?;
            for item in value {
                writer.write_all(&item.to_le_bytes())?;
            }
        }
        GgufKv::ArrayI32 { key, value } => {
            write_array_header(writer, key, GGUF_TYPE_INT32, value.len())?;
            for item in value {
                writer.write_all(&item.to_le_bytes())?;
            }
        }
        GgufKv::ArrayString { key, value } => {
            write_array_header(writer, key, GGUF_TYPE_STRING, value.len())?;
            for item in value {
                write_string(writer, item)?;
            }
        }
        GgufKv::ArrayU32 { key, value } => {
            write_array_header(writer, key, GGUF_TYPE_UINT32, value.len())?;
            for item in value {
                writer.write_all(&item.to_le_bytes())?;
            }
        }
        GgufKv::Bool { key, value } => {
            write_scalar_header(writer, key, GGUF_TYPE_BOOL)?;
            writer.write_all(&[*value as u8])?;
        }
        GgufKv::F32 { key, value } => {
            write_scalar_header(writer, key, GGUF_TYPE_FLOAT32)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        GgufKv::I32 { key, value } => {
            write_scalar_header(writer, key, GGUF_TYPE_INT32)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        GgufKv::String { key, value } => {
            write_scalar_header(writer, key, GGUF_TYPE_STRING)?;
            write_string(writer, value)?;
        }
        GgufKv::U16 { key, value } => {
            write_scalar_header(writer, key, GGUF_TYPE_UINT16)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        GgufKv::U32 { key, value } => {
            write_scalar_header(writer, key, GGUF_TYPE_UINT32)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        GgufKv::U64 { key, value } => {
            write_scalar_header(writer, key, GGUF_TYPE_UINT64)?;
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_array_header<W: Write>(
    writer: &mut W,
    key: &str,
    element_type: u32,
    len: usize,
) -> Result<()> {
    ensure!(
        len > 0,
        "GGUF array metadata {key:?} cannot be empty because llama.cpp rejects empty arrays"
    );
    write_scalar_header(writer, key, GGUF_TYPE_ARRAY)?;
    writer.write_all(&element_type.to_le_bytes())?;
    writer.write_all(&(len as u64).to_le_bytes())?;
    Ok(())
}

fn write_scalar_header<W: Write>(writer: &mut W, key: &str, value_type: u32) -> Result<()> {
    ensure!(!key.is_empty(), "GGUF metadata key cannot be empty");
    write_string(writer, key)?;
    writer.write_all(&value_type.to_le_bytes())?;
    Ok(())
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    writer.write_all(&(value.len() as u64).to_le_bytes())?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

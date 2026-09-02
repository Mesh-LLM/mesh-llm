use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;

use crate::gguf_template::metadata_from_hf_config;
use crate::tensor_map::TensorNameMap;

use super::*;

include!("gguf_writer_tests/conversion.rs");
include!("gguf_writer_tests/validation.rs");
include!("gguf_writer_tests/layout.rs");
include!("gguf_writer_tests/parser.rs");
include!("gguf_writer_tests/fixtures.rs");

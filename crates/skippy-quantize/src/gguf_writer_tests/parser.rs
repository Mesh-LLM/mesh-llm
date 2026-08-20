struct ParsedGguf {
    tensor_count: u64,
    metadata_count: u64,
    metadata_string_arrays: BTreeMap<String, Vec<String>>,
    data_start: usize,
    tensors: Vec<ParsedTensor>,
}

impl ParsedGguf {
    fn tensor(&self, name: &str) -> &ParsedTensor {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
    }
}

struct ParsedTensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    absolute_offset: usize,
}

fn parse_test_gguf(bytes: &[u8]) -> ParsedGguf {
    let mut cursor = std::io::Cursor::new(bytes);
    let mut magic = [0_u8; 4];
    cursor.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, GGUF_MAGIC);
    assert_eq!(read_u32(&mut cursor), GGUF_VERSION);
    let tensor_count = read_u64(&mut cursor);
    let metadata_count = read_u64(&mut cursor);
    let mut metadata_string_arrays = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut cursor);
        let value_type = read_u32(&mut cursor);
        match value_type {
            GGUF_TYPE_BOOL => {
                let mut value = [0_u8; 1];
                cursor.read_exact(&mut value).unwrap();
            }
            GGUF_TYPE_UINT16 => {
                let mut value = [0_u8; 2];
                cursor.read_exact(&mut value).unwrap();
            }
            GGUF_TYPE_INT32 => {
                let _ = read_u32(&mut cursor);
            }
            GGUF_TYPE_STRING => {
                let _ = read_string(&mut cursor);
            }
            GGUF_TYPE_UINT32 => {
                let _ = read_u32(&mut cursor);
            }
            GGUF_TYPE_FLOAT32 => {
                let _ = read_u32(&mut cursor);
            }
            GGUF_TYPE_UINT64 => {
                let _ = read_u64(&mut cursor);
            }
            GGUF_TYPE_ARRAY => {
                if let Some(value) = read_string_array_or_skip(&mut cursor) {
                    metadata_string_arrays.insert(key, value);
                }
            }
            other => panic!("unexpected metadata type {other}"),
        }
    }
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut cursor);
        let dim_count = read_u32(&mut cursor);
        let dims = (0..dim_count)
            .map(|_| read_u64(&mut cursor))
            .collect::<Vec<_>>();
        let ggml_type = read_u32(&mut cursor);
        let relative_offset = read_u64(&mut cursor);
        tensors.push((name, dims, ggml_type, relative_offset));
    }
    let data_start = align_to(cursor.position(), GGUF_ALIGNMENT) as usize;
    ParsedGguf {
        tensor_count,
        metadata_count,
        metadata_string_arrays,
        data_start,
        tensors: tensors
            .into_iter()
            .map(|(name, dims, ggml_type, relative_offset)| ParsedTensor {
                name,
                dims,
                ggml_type,
                absolute_offset: data_start + relative_offset as usize,
            })
            .collect(),
    }
}

fn read_string(cursor: &mut std::io::Cursor<&[u8]>) -> String {
    let len = read_u64(cursor);
    let mut bytes = vec![0_u8; len as usize];
    cursor.read_exact(&mut bytes).unwrap();
    String::from_utf8(bytes).unwrap()
}

fn read_u32(cursor: &mut std::io::Cursor<&[u8]>) -> u32 {
    let mut bytes = [0_u8; 4];
    cursor.read_exact(&mut bytes).unwrap();
    u32::from_le_bytes(bytes)
}

fn read_u64(cursor: &mut std::io::Cursor<&[u8]>) -> u64 {
    let mut bytes = [0_u8; 8];
    cursor.read_exact(&mut bytes).unwrap();
    u64::from_le_bytes(bytes)
}

fn read_string_array_or_skip(cursor: &mut std::io::Cursor<&[u8]>) -> Option<Vec<String>> {
    let element_type = read_u32(cursor);
    let len = read_u64(cursor);
    if element_type == GGUF_TYPE_STRING {
        return Some((0..len).map(|_| read_string(cursor)).collect());
    }
    skip_array_items(cursor, element_type, len);
    None
}

fn skip_array_items(cursor: &mut std::io::Cursor<&[u8]>, element_type: u32, len: u64) {
    for _ in 0..len {
        match element_type {
            GGUF_TYPE_BOOL => {
                let mut value = [0_u8; 1];
                cursor.read_exact(&mut value).unwrap();
            }
            GGUF_TYPE_STRING => {
                let _ = read_string(cursor);
            }
            GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 | GGUF_TYPE_UINT32 => {
                let _ = read_u32(cursor);
            }
            other => panic!("unexpected test array element type {other}"),
        }
    }
}

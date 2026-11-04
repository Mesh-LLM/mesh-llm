use std::sync::atomic::{AtomicU64, Ordering};

use super::super::limits::CannotAudit;
use super::super::test_gguf::{Memory, string};
use super::*;
use crate::ggml_types::{GGML_F32, GGML_Q4_K, GGML_Q8_0};

const UINT8: u32 = 0;
const UINT64: u32 = 10;

/// A GGUF written field by field, so a test can write what no writer would.
struct Raw(Vec<u8>);

impl Raw {
    fn new(version: u32, tensors: u64, entries: u64) -> Self {
        let mut out = b"GGUF".to_vec();
        out.extend(version.to_le_bytes());
        out.extend(tensors.to_le_bytes());
        out.extend(entries.to_le_bytes());
        Self(out)
    }

    fn u32(mut self, value: u32) -> Self {
        self.0.extend(value.to_le_bytes());
        self
    }

    fn u64(mut self, value: u64) -> Self {
        self.0.extend(value.to_le_bytes());
        self
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        self.0.extend(bytes);
        self
    }

    fn text(mut self, text: &str) -> Self {
        string(&mut self.0, text);
        self
    }

    /// A metadata key and the type of its value.
    fn key(self, key: &str, ty: u32) -> Self {
        self.text(key).u32(ty)
    }

    /// The type and count that start an array.
    fn array(self, inner: u32, count: u64) -> Self {
        self.u32(inner).u64(count)
    }

    fn tensor(mut self, name: &str, dims: &[u64], ggml_type: u32, relative: u64) -> Self {
        string(&mut self.0, name);
        self.0.extend((dims.len() as u32).to_le_bytes());
        for dim in dims {
            self.0.extend(dim.to_le_bytes());
        }
        self.u32(ggml_type).u64(relative)
    }

    /// Pads to the alignment and adds `data` bytes of tensor data.
    fn data(mut self, alignment: usize, data: usize) -> Vec<u8> {
        let start = self.0.len().div_ceil(alignment) * alignment;
        self.0.resize(start + data, 0);
        self.0
    }
}

fn parse(bytes: Vec<u8>) -> Result<Header> {
    parse_header(&Memory(bytes), FIRST_READ, MAX_HEADER_BYTES)
}

/// The reason a header is refused: always a [`CannotAudit`], never a panic
/// and never another error.
fn refused(bytes: Vec<u8>) -> String {
    let error = parse(bytes).err().expect("the header is refused");
    assert!(CannotAudit::caused(&error), "{error:#}");
    error.to_string()
}

/// One f32 matrix `w` of two rows of four values.
fn one_tensor(entries: u64) -> Raw {
    Raw::new(3, 1, entries)
}

fn with_w(raw: Raw) -> Vec<u8> {
    raw.tensor("w", &[4, 2], GGML_F32, 0).data(32, 32)
}

#[test]
fn a_header_reads_its_metadata_and_places_its_tensors() {
    let bytes = Raw::new(3, 2, 6)
        .key("general.architecture", GGUF_STRING)
        .text("toy")
        .key("toy.block_count", GGUF_UINT32)
        .u32(7)
        .key("toy.eps", GGUF_FLOAT32)
        .bytes(&0.5f32.to_le_bytes())
        .key("toy.flag", GGUF_BOOL)
        .bytes(&[1])
        // an array of strings, and an array of arrays
        .key("tokenizer.tokens", GGUF_ARRAY)
        .array(GGUF_STRING, 2)
        .text("a")
        .bytes(&3u64.to_le_bytes())
        .bytes(&[0xff, 0xfe, 0xfd])
        .key("toy.nested", GGUF_ARRAY)
        .array(GGUF_ARRAY, 2)
        .array(UINT8, 3)
        .bytes(&[1, 2, 3])
        .array(GGUF_STRING, 1)
        .text("deep")
        .tensor("blk.0.w", &[4, 2], GGML_F32, 0)
        .tensor("blk.0.q", &[32, 3], GGML_Q8_0, 32)
        .data(32, 32 + 102);
    let size = bytes.len() as u64;
    let header = parse(bytes).unwrap();
    assert_eq!(
        header.metadata.get("general.architecture"),
        Some(&MetaValue::Str("toy".into()))
    );
    assert_eq!(
        header.metadata.get("toy.block_count"),
        Some(&MetaValue::Int(7))
    );
    assert_eq!(header.metadata.get("toy.eps"), Some(&MetaValue::Float(0.5)));
    assert_eq!(header.metadata.len(), 3);
    assert_eq!(header.data_start % 32, 0);
    let w = &header.tensors["blk.0.w"];
    assert_eq!(
        (w.offset, w.total_bytes().unwrap()),
        (header.data_start, 32)
    );
    let q = &header.tensors["blk.0.q"];
    assert_eq!(q.offset, header.data_start + 32);
    assert_eq!(q.offset + q.total_bytes().unwrap(), size);
}

#[test]
fn arrays_nested_too_deep_are_refused_without_recursion() {
    let nested = |levels: usize| {
        let mut raw = one_tensor(1).key("deep", GGUF_ARRAY);
        for _ in 1..levels {
            raw = raw.array(GGUF_ARRAY, 1);
        }
        with_w(raw.array(UINT8, 1).bytes(&[0]))
    };
    assert!(parse(nested(1)).is_ok());
    assert!(parse(nested(2)).is_ok());
    assert!(refused(nested(3)).contains("nest deeper"));
    // as many levels as a megabyte holds: once a frame of the stack each
    assert!(refused(nested(80_000)).contains("nest deeper"));
}

#[test]
fn lengths_and_counts_past_the_limits_are_refused() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("tensors", Raw::new(3, MAX_TENSORS + 1, 0).data(32, 0)),
        ("tensors", Raw::new(3, u64::MAX, 0).data(32, 0)),
        (
            "metadata entries",
            Raw::new(3, 0, MAX_METADATA_ENTRIES + 1).data(32, 0),
        ),
        ("metadata entries", Raw::new(3, 0, u64::MAX).data(32, 0)),
        // a key length that wraps the cursor
        ("longer than", one_tensor(1).u64(u64::MAX - 7).data(32, 0)),
        (
            "longer than",
            one_tensor(1).u64(MAX_NAME_BYTES + 1).data(32, 0),
        ),
        (
            "longer than",
            one_tensor(1)
                .key("k", GGUF_STRING)
                .u64(MAX_STRING_BYTES + 1)
                .data(32, 0),
        ),
        (
            "longer than",
            one_tensor(1)
                .key("k", GGUF_ARRAY)
                .array(GGUF_STRING, 1)
                .u64(u64::MAX)
                .data(32, 0),
        ),
        (
            "does not fit",
            one_tensor(1)
                .key("k", GGUF_ARRAY)
                .array(UINT64, u64::MAX / 4)
                .data(32, 0),
        ),
        (
            "does not fit",
            one_tensor(1)
                .key("k", GGUF_ARRAY)
                .array(GGUF_STRING, u64::MAX)
                .data(32, 0),
        ),
        (
            "does not fit",
            one_tensor(1)
                .key("k", GGUF_ARRAY)
                .array(GGUF_ARRAY, 1 << 40)
                .data(32, 0),
        ),
        (
            "longer than",
            Raw::new(3, 1, 0).u64(MAX_NAME_BYTES + 1).data(32, 0),
        ),
    ];
    for (reason, bytes) in cases {
        let refused = refused(bytes);
        assert!(refused.contains(reason), "{reason}: {refused}");
    }
}

/// A file of `size` bytes whose start is `held` and whose rest is zeros,
/// which counts what is read of it.
struct Counted {
    held: Vec<u8>,
    size: u64,
    requests: AtomicU64,
    read: AtomicU64,
}

impl Counted {
    fn new(held: Vec<u8>, size: u64) -> Self {
        Self {
            held,
            size,
            requests: AtomicU64::new(0),
            read: AtomicU64::new(0),
        }
    }
}

impl ByteSource for Counted {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        assert!(len <= 64 << 20, "a read of {len} bytes");
        assert!(offset + len <= self.size);
        self.requests.fetch_add(1, Ordering::SeqCst);
        self.read.fetch_add(len, Ordering::SeqCst);
        let mut out = vec![0u8; len as usize];
        if let Some(held) = self.held.get(offset as usize..) {
            let n = held.len().min(out.len());
            out[..n].copy_from_slice(&held[..n]);
        }
        Ok(out)
    }

    fn size(&self) -> u64 {
        self.size
    }
}

#[test]
fn a_length_that_asks_for_the_file_reads_none_of_it() {
    // a remote file of a terabyte whose header says an array is all of it
    let size = 1u64 << 40;
    let start = one_tensor(1)
        .key("k", GGUF_ARRAY)
        .array(UINT8, size - 100)
        .data(32, 0);
    let source = Counted::new(start, size);
    let error = parse_header(&source, FIRST_READ, MAX_HEADER_BYTES)
        .err()
        .unwrap();
    assert!(CannotAudit::caused(&error), "{error:#}");
    // an array that fits the count check and not the header
    let start = one_tensor(1)
        .key("k", GGUF_ARRAY)
        .array(UINT8, MAX_HEADER_BYTES - 16)
        .data(32, 0);
    let fits = Counted::new(start, size);
    let error = parse_header(&fits, FIRST_READ, MAX_HEADER_BYTES)
        .err()
        .unwrap();
    assert!(error.to_string().contains("larger than"), "{error:#}");
    for source in [source, fits] {
        assert_eq!(source.requests.load(Ordering::SeqCst), 1);
        assert_eq!(source.read.load(Ordering::SeqCst), FIRST_READ);
    }
}

#[test]
fn a_header_takes_one_request_and_grows_by_doubling() {
    // a tokenizer of three megabytes: one request, where a megabyte at a
    // time and doubling took three
    let tokens = vec![b'x'; 3 << 20];
    let small = with_w(
        one_tensor(1)
            .key("tokens", GGUF_ARRAY)
            .array(UINT8, tokens.len() as u64)
            .bytes(&tokens),
    );
    let size = small.len() as u64;
    let source = Counted::new(small, size);
    parse_header(&source, FIRST_READ, MAX_HEADER_BYTES).unwrap();
    assert_eq!(source.requests.load(Ordering::SeqCst), 1);
    // twenty megabytes: the first 8, what the array then needs, and as
    // much again, which is the rest
    let tokens = vec![b'x'; 20 << 20];
    let large = with_w(
        one_tensor(1)
            .key("tokens", GGUF_ARRAY)
            .array(UINT8, tokens.len() as u64)
            .bytes(&tokens),
    );
    let size = large.len() as u64;
    let source = Counted::new(large, size);
    parse_header(&source, FIRST_READ, MAX_HEADER_BYTES).unwrap();
    assert_eq!(source.requests.load(Ordering::SeqCst), 3);
    assert_eq!(source.read.load(Ordering::SeqCst), size);
    // a later shard asks for less first
    let shard = Counted::new(with_w(one_tensor(0)), 4 << 20);
    parse_header(&shard, SHARD_FIRST_READ, MAX_HEADER_BYTES).unwrap();
    assert_eq!(shard.read.load(Ordering::SeqCst), SHARD_FIRST_READ);
}

#[test]
fn a_shard_reads_only_what_the_shards_before_it_left_of_the_limit() {
    let bytes = with_w(one_tensor(0));
    let header = parse(bytes.clone()).unwrap();
    assert!(header.header_bytes <= header.data_start);
    let left = |max_bytes: u64| {
        let source = Counted::new(bytes.clone(), 4 << 20);
        let parsed = parse_header(&source, SHARD_FIRST_READ, max_bytes);
        (parsed, source.read.load(Ordering::SeqCst))
    };
    let (parsed, read) = left(header.header_bytes);
    assert!(parsed.is_ok());
    assert_eq!(read, header.header_bytes);
    for max_bytes in [header.header_bytes - 1, 8, 0] {
        let (parsed, read) = left(max_bytes);
        let error = parsed.err().unwrap();
        assert!(CannotAudit::caused(&error), "{error:#}");
        assert!(error.to_string().contains("larger than"), "{error}");
        assert!(read <= max_bytes);
    }
}

#[test]
fn versions_and_byte_orders_the_runtime_does_not_load_are_refused() {
    assert!(parse(with_w(Raw::new(2, 1, 0))).is_ok());
    assert!(parse(with_w(Raw::new(3, 1, 0))).is_ok());
    for version in [0, 1, 4, 0x0001_0000, u32::MAX] {
        let refused = refused(with_w(Raw::new(version, 1, 0)));
        assert!(refused.contains("version"), "{version}: {refused}");
    }
    // version 3 and 2 as a big-endian writer writes them
    for version in [3u32, 2] {
        let refused = refused(with_w(Raw::new(version.swap_bytes(), 1, 0)));
        assert!(refused.contains("big-endian"), "{refused}");
    }
    let mut not_gguf = with_w(Raw::new(3, 1, 0));
    not_gguf[..4].copy_from_slice(b"GGML");
    assert!(refused(not_gguf).contains("not a GGUF"));
    // a hole of a sparse copy
    assert!(refused(vec![0u8; 4096]).contains("not a GGUF"));
    assert!(refused(Vec::new()).contains("past the end"));
}

#[test]
fn names_the_loader_would_read_otherwise_are_refused() {
    let bad_utf8 = [0x62, 0x6c, 0x6b, 0xff];
    let named = |name: &[u8]| {
        Raw::new(3, 1, 0)
            .u64(name.len() as u64)
            .bytes(name)
            .u32(2)
            .u64(4)
            .u64(2)
            .u32(GGML_F32)
            .u64(0)
            .data(32, 32)
    };
    assert!(parse(named(b"blk.0.w")).is_ok());
    assert!(refused(named(&bad_utf8)).contains("not UTF-8"));
    assert!(refused(named(b"")).contains("empty"));
    assert!(refused(named(b"blk.0\0.w")).contains("control"));
    assert!(refused(named(b"blk.0\n.w")).contains("control"));
    let keyed = |key: &[u8]| {
        with_w(
            one_tensor(1)
                .u64(key.len() as u64)
                .bytes(key)
                .u32(GGUF_UINT32)
                .u32(1),
        )
    };
    assert!(parse(keyed(b"general.x")).is_ok());
    assert!(refused(keyed(&bad_utf8)).contains("not UTF-8"));
    assert!(refused(keyed(b"")).contains("empty"));
    assert!(refused(keyed(b"a\x1bb")).contains("control"));
    let valued = with_w(
        one_tensor(1)
            .key("general.name", GGUF_STRING)
            .u64(4)
            .bytes(&bad_utf8),
    );
    assert!(refused(valued).contains("not UTF-8"));
}

#[test]
fn names_that_come_twice_are_refused() {
    let keys = with_w(
        one_tensor(2)
            .key("general.x", GGUF_UINT32)
            .u32(1)
            .key("general.x", GGUF_UINT32)
            .u32(2),
    );
    assert!(refused(keys).contains("twice"));
    // an array is not kept, and its key still counts
    let array_key = with_w(
        one_tensor(2)
            .key("general.x", GGUF_ARRAY)
            .array(UINT8, 0)
            .key("general.x", GGUF_UINT32)
            .u32(2),
    );
    assert!(refused(array_key).contains("twice"));
    let tensors = Raw::new(3, 2, 0)
        .tensor("w", &[4, 2], GGML_F32, 0)
        .tensor("w", &[4, 2], GGML_F32, 32)
        .data(32, 64);
    assert!(refused(tensors).contains("twice"));
}

#[test]
fn metadata_of_unknown_types_and_bad_alignments_are_refused() {
    let typed = |ty: u32| with_w(one_tensor(1).key("k", ty).u64(0));
    for ty in [13, 14, 255, u32::MAX] {
        assert!(refused(typed(ty)).contains("unknown metadata type"));
        let array = with_w(one_tensor(1).key("k", GGUF_ARRAY).array(ty, 1).u64(0));
        assert!(refused(array).contains("unknown metadata type"));
    }
    let aligned = |ty: u32, value: &[u8], alignment: usize| {
        one_tensor(1)
            .key(ALIGNMENT_KEY, ty)
            .bytes(value)
            .tensor("w", &[4, 2], GGML_F32, 0)
            .data(alignment, 32)
    };
    let header = parse(aligned(GGUF_UINT32, &64u32.to_le_bytes(), 64)).unwrap();
    assert_eq!(header.data_start % 64, 0);
    for alignment in [0u32, 3, 48, u32::MAX] {
        let bytes = aligned(GGUF_UINT32, &alignment.to_le_bytes(), 32);
        assert!(refused(bytes).contains("power of two"), "{alignment}");
    }
    // the loader takes a u32 only: another type would abort it
    let u64_typed = aligned(UINT64, &32u64.to_le_bytes(), 32);
    assert!(refused(u64_typed).contains("power of two"));
    let string_typed = with_w(one_tensor(1).key(ALIGNMENT_KEY, GGUF_STRING).text("32"));
    assert!(refused(string_typed).contains("power of two"));
}

#[test]
fn tensors_of_bad_shapes_and_types_are_refused() {
    let shaped = |dims: &[u64], ggml_type: u32| {
        Raw::new(3, 1, 0)
            .tensor("w", dims, ggml_type, 0)
            .data(32, 64)
    };
    assert!(parse(shaped(&[4, 2, 1, 1], GGML_F32)).is_ok());
    assert!(parse(shaped(&[32, 1], GGML_Q8_0)).is_ok());
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("dimensions", shaped(&[], GGML_F32)),
        ("dimensions", shaped(&[1, 1, 1, 1, 1], GGML_F32)),
        // a rank that would read the rest of the file as dimensions
        (
            "dimensions",
            Raw::new(3, 1, 0)
                .text("w")
                .u32(u32::MAX)
                .u64(4)
                .data(32, 64),
        ),
        ("zero", shaped(&[0], GGML_F32)),
        ("zero", shaped(&[4, 0], GGML_F32)),
        ("does not fit", shaped(&[1 << 32, 1 << 32], GGML_F32)),
        ("does not fit", shaped(&[u64::MAX, 2], GGML_F32)),
        ("does not fit", shaped(&[1 << 62, 2], GGML_F32)),
        ("does not fit", shaped(&[1 << 62], GGML_F32)),
        // a row shorter than a block, which has no bytes
        ("whole blocks", shaped(&[16, 2], GGML_Q8_0)),
        ("whole blocks", shaped(&[255, 2], GGML_Q4_K)),
        ("whole blocks", shaped(&[33, 2], GGML_Q8_0)),
        ("does not load", shaped(&[4, 2], 4)),
        ("does not load", shaped(&[4, 2], 5)),
        ("does not load", shaped(&[4, 2], 31)),
        ("does not load", shaped(&[4, 2], 44)),
        ("does not load", shaped(&[4, 2], u32::MAX)),
    ];
    for (reason, bytes) in cases {
        let refused = refused(bytes);
        assert!(refused.contains(reason), "{reason}: {refused}");
    }
}

#[test]
fn tensors_out_of_place_are_refused() {
    let two = |second: u64, data: usize| {
        Raw::new(3, 2, 0)
            .tensor("a", &[4, 2], GGML_F32, 0)
            .tensor("b", &[4, 2], GGML_F32, second)
            .data(32, data)
    };
    assert!(parse(two(32, 64)).is_ok());
    // on the first, off the alignment, after a gap, and wrapped
    for second in [0, 16, 31, 33, 64, u64::MAX, u64::MAX - 31] {
        let refused = refused(two(second, 4096));
        assert!(
            refused.contains("where the tensor before it ends"),
            "{second}"
        );
    }
    // the last tensor ends past the file
    for data in [0, 32, 63] {
        assert!(refused(two(32, data)).contains("past the end"), "{data}");
    }
    let first_misplaced = Raw::new(3, 1, 0)
        .tensor("a", &[4, 2], GGML_F32, 32)
        .data(32, 4096);
    assert!(refused(first_misplaced).contains("not at 0"));
    // a tensor as large as ggml counts, in a file that is not
    let huge = Raw::new(3, 1, 0)
        .tensor("a", &[1 << 60], GGML_F32, 0)
        .data(32, 64);
    assert!(refused(huge).contains("past the end"));
}

#[test]
fn a_header_cut_anywhere_is_refused() {
    let whole = Raw::new(3, 2, 3)
        .key("general.architecture", GGUF_STRING)
        .text("toy")
        .key("toy.list", GGUF_ARRAY)
        .array(GGUF_STRING, 2)
        .text("a")
        .text("b")
        .key("toy.n", GGUF_UINT32)
        .u32(7)
        .tensor("a", &[4, 2], GGML_F32, 0)
        .tensor("b", &[32, 1], GGML_Q8_0, 32)
        .data(32, 32 + 34);
    assert!(parse(whole.clone()).is_ok());
    for cut in 0..whole.len() {
        refused(whole[..cut].to_vec());
    }
}

#[test]
fn any_byte_changed_is_parsed_or_refused_and_never_panics() {
    let whole = Raw::new(3, 2, 3)
        .key("general.architecture", GGUF_STRING)
        .text("toy")
        .key("toy.list", GGUF_ARRAY)
        .array(GGUF_ARRAY, 1)
        .array(UINT8, 2)
        .bytes(&[1, 2])
        .key(ALIGNMENT_KEY, GGUF_UINT32)
        .u32(32)
        .tensor("a", &[4, 2], GGML_F32, 0)
        .tensor("b", &[32, 1], GGML_Q8_0, 32)
        .data(32, 32 + 34);
    let header_end = whole.len() - (32 + 34);
    for at in 0..header_end {
        for value in [0x00, 0x01, 0x7f, 0x80, 0xff] {
            let mut changed = whole.clone();
            if changed[at] == value {
                continue;
            }
            changed[at] = value;
            if let Err(error) = parse(changed) {
                assert!(
                    CannotAudit::caused(&error),
                    "byte {at} to {value}: {error:#}"
                );
            }
        }
    }
}

#[test]
fn the_list_of_suppressed_tokens_is_kept() {
    const INT32: u32 = 5;
    let with = |key: &str, count: u64| {
        let mut raw = one_tensor(1).key(key, GGUF_ARRAY).array(INT32, count);
        for token in 0..count {
            raw = raw.bytes(&(token as i32).to_le_bytes());
        }
        parse(with_w(raw)).unwrap()
    };
    let header = with(SUPPRESSED_TOKENS_KEY, 3);
    assert_eq!(
        header.metadata.get(SUPPRESSED_TOKENS_KEY),
        Some(&MetaValue::Tokens(vec![0, 1, 2]))
    );
    // an empty list, and any other array, as before
    assert_eq!(
        with(SUPPRESSED_TOKENS_KEY, 0)
            .metadata
            .get(SUPPRESSED_TOKENS_KEY),
        Some(&MetaValue::Tokens(Vec::new()))
    );
    assert!(with("tokenizer.ggml.token_type", 3).metadata.is_empty());
    // the runtime refuses a list of another type, and so does the check
    let strings = one_tensor(1)
        .key(SUPPRESSED_TOKENS_KEY, GGUF_ARRAY)
        .array(4, 1)
        .bytes(&7u32.to_le_bytes());
    assert!(parse(with_w(strings)).is_err());
}

const INT32: u32 = 5;

/// A header whose only entries are `tokenizer.ggml.tokens` holding
/// `tokens` and `toy.expert_used_count` holding `used`.
fn with_arrays(tokens: &[&[u8]], used: &[i32]) -> Vec<u8> {
    let mut raw = one_tensor(2)
        .key(kept_arrays::TOKENS_KEY, GGUF_ARRAY)
        .array(GGUF_STRING, tokens.len() as u64);
    for token in tokens {
        raw = raw.u64(token.len() as u64).bytes(token);
    }
    raw = raw
        .key("toy.expert_used_count", GGUF_ARRAY)
        .array(INT32, used.len() as u64);
    for value in used {
        raw = raw.bytes(&value.to_le_bytes());
    }
    with_w(raw)
}

#[test]
fn the_arrays_the_loader_asserts_on_are_kept() {
    let header = parse(with_arrays(&[b"a", b"<0x0A>", b"", b"b"], &[2, -1])).unwrap();
    let Some(MetaArray::Tokens(tokens)) = header.arrays.get(kept_arrays::TOKENS_KEY) else {
        panic!("{:?}", header.arrays);
    };
    assert_eq!(tokens.count, 4);
    assert_eq!(tokens.duplicate, None);
    assert!(tokens.has_byte(10) && tokens.has_byte(b'a') && !tokens.has_byte(0));
    assert!(!tokens.has_every_byte());
    assert_eq!(
        header.arrays.get("toy.expert_used_count"),
        Some(&MetaArray::Ints(vec![2, -1]))
    );
}

#[test]
fn two_tokens_of_one_text_are_found() {
    let twice = parse(with_arrays(&[b"a", b"b", b"a"], &[])).unwrap();
    let Some(MetaArray::Tokens(tokens)) = twice.arrays.get(kept_arrays::TOKENS_KEY) else {
        panic!("no tokens");
    };
    assert_eq!(tokens.duplicate, Some(2));
    // the loader names an empty token by its index, which another may have
    let renamed = parse(with_arrays(&[b"[EMPTY_1]", b""], &[])).unwrap();
    let Some(MetaArray::Tokens(tokens)) = renamed.arrays.get(kept_arrays::TOKENS_KEY) else {
        panic!("no tokens");
    };
    assert_eq!(tokens.duplicate, Some(1));
    assert!(refused(with_arrays(&[], &[])).contains("tokens"));
}

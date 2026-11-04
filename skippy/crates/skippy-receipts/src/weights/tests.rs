use super::limits::{CannotAudit, DEFAULT_FETCH_BUDGET};
use super::*;
use crate::ggml_types::{GGML_F32, GGML_Q4_K, GGML_Q8_0};

#[test]
fn rows_are_read_in_order_across_chunks() {
    let weights = test_gguf::open(
        &[],
        &[("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
    );
    let mut seen = Vec::new();
    weights
        .each_row("w", 1..3, &mut |row, values| {
            seen.push((row, values.to_vec()))
        })
        .unwrap();
    assert_eq!(seen, vec![(1, vec![3.0, 4.0]), (2, vec![5.0, 6.0])]);
    assert!(weights.each_row("w", 0..4, &mut |_, _| {}).is_err());
}

fn shard(no: u64, tensors: &[(&str, Vec<u64>, Vec<f32>)]) -> Box<dyn ByteSource> {
    let mut metadata = vec![
        ("split.no", MetaValue::Int(no)),
        ("split.count", MetaValue::Int(2)),
    ];
    if no == 0 {
        metadata.push(("general.architecture", MetaValue::Str("test".into())));
        metadata.push(("split.tensors.count", MetaValue::Int(2)));
    }
    Box::new(test_gguf::Memory(test_gguf::bytes(&metadata, tensors)))
}

#[test]
fn a_split_gguf_reads_as_one_file() {
    let weights = WeightFile::open_shards(vec![
        shard(0, &[("a", vec![2, 1], vec![1.0, 2.0])]),
        shard(1, &[("b", vec![2, 2], vec![3.0, 4.0, 5.0, 6.0])]),
    ])
    .unwrap();
    assert_eq!(
        weights.metadata_str("general.architecture").unwrap(),
        "test"
    );
    assert_eq!(weights.row("a", 0).unwrap(), vec![1.0, 2.0]);
    assert_eq!(weights.row("b", 1).unwrap(), vec![5.0, 6.0]);
    let (offset, len) = weights.row_range("b", 1).unwrap();
    let bytes = weights.source.read_at(offset, len).unwrap();
    let expected: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(bytes, expected);
}

#[test]
fn shards_out_of_order_are_refused() {
    let swapped = WeightFile::open_shards(vec![
        shard(1, &[("b", vec![2, 2], vec![3.0, 4.0, 5.0, 6.0])]),
        shard(0, &[("a", vec![2, 1], vec![1.0, 2.0])]),
    ]);
    assert!(swapped.is_err());
}

fn refused(result: Result<WeightFile>) -> String {
    let error = result.err().expect("the file is refused");
    assert!(CannotAudit::caused(&error), "{error:#}");
    format!("{error:#}")
}

#[test]
fn a_tensor_in_two_shards_is_refused() {
    let twice = WeightFile::open_shards(vec![
        shard(0, &[("a", vec![2, 1], vec![1.0, 2.0])]),
        shard(1, &[("a", vec![2, 1], vec![1.0, 2.0])]),
    ]);
    assert!(refused(twice).contains("in two of its shards"));
}

#[test]
fn too_many_shards_are_refused_before_any_is_read() {
    /// A shard that fails the test when it is read.
    struct Unread;
    impl ByteSource for Unread {
        fn read_at(&self, _offset: u64, _len: u64) -> Result<Vec<u8>> {
            panic!("a shard past the limit was read")
        }
        fn size(&self) -> u64 {
            1 << 20
        }
    }
    let shards = (0..=MAX_SHARDS)
        .map(|_| Box::new(Unread) as Box<dyn ByteSource>)
        .collect();
    assert!(refused(WeightFile::open_shards(shards)).contains("shards"));
    assert!(refused(WeightFile::open_shards(Vec::new())).contains("shards"));
}

/// A model with one matrix `w` of two rows, whose tensor a test then edits
/// to what a header must never give: the reads are errors, not panics.
fn edited(edit: impl FnOnce(&mut TensorInfo)) -> WeightFile {
    let mut weights = test_gguf::open(&[], &[("w", vec![32, 2], vec![0.5; 64])]);
    edit(weights.tensors.get_mut("w").unwrap());
    weights
}

fn every_read_fails(weights: &WeightFile) {
    assert!(weights.row("w", 0).is_err());
    assert!(weights.min_blocks("w", 0).is_err());
    assert!(weights.row_range("w", 0).is_err());
    assert!(weights.each_row("w", 0..1, &mut |_, _| {}).is_err());
    assert!(weights.stream_rows("w", 0..1, &mut |_, _| {}).is_err());
}

#[test]
fn a_row_shorter_than_a_block_is_an_error_not_a_panic() {
    // 16 values of a type with blocks of 32, and of 256: no bytes per row
    for (ggml_type, row_len) in [(GGML_Q8_0, 16), (GGML_Q4_K, 32), (GGML_Q4_K, 255)] {
        let weights = edited(|t| {
            t.ggml_type = ggml_type;
            t.dims[0] = row_len;
        });
        every_read_fails(&weights);
        assert!(weights.tensor_range("w").is_err());
        assert!(weights.tensors["w"].row_bytes().is_err());
        assert!(weights.tensors["w"].total_bytes().is_err());
    }
    let empty = edited(|t| t.dims[0] = 0);
    every_read_fails(&empty);
    assert!(empty.tensors["w"].total_bytes().is_err());
}

#[test]
fn sizes_that_wrap_are_errors() {
    let huge_rows = edited(|t| t.dims[1] = u64::MAX);
    assert!(huge_rows.tensors["w"].total_bytes().is_err());
    assert!(huge_rows.row_range("w", u64::MAX - 1).is_err());
    let wrapped = edited(|t| t.dims = vec![1 << 32, 1 << 32]);
    assert!(wrapped.tensors["w"].row_count().is_ok());
    assert!(wrapped.tensors["w"].total_bytes().is_err());
    let long_row = edited(|t| t.dims[0] = u64::MAX - 31);
    assert!(long_row.tensors["w"].row_bytes().is_err());
    every_read_fails(&long_row);
    let far = edited(|t| t.offset = u64::MAX - 8);
    assert!(far.row_range("w", 1).is_err());
    assert!(far.row("w", 1).is_err());
    // the layer's bytes saturate; they do not wrap
    let names = ["blk.0.a.weight", "blk.0.b.weight"];
    let tensors: Vec<_> = names
        .iter()
        .map(|name| (*name, vec![32, 2], vec![0.5; 64]))
        .collect();
    let mut layer = test_gguf::open(&[], &tensors);
    for name in names {
        // 2^63 bytes each
        layer.tensors.get_mut(name).unwrap().dims[1] = 1 << 56;
    }
    assert_eq!(layer.layer_bytes()[&0], u64::MAX);
}

#[test]
fn a_tensor_of_an_unknown_type_has_no_size() {
    for ggml_type in [4, 5, 31, 36, 44, u32::MAX] {
        let weights = edited(|t| t.ggml_type = ggml_type);
        let error = weights.tensors["w"].total_bytes().unwrap_err();
        assert!(CannotAudit::caused(&error), "{error:#}");
        assert!(weights.tensor_range("w").is_err());
        every_read_fails(&weights);
    }
    // a type ggml has and the verifier does not decode still has a size
    let sized = edited(|t| t.ggml_type = 19);
    assert!(
        sized.tensors["w"].total_bytes().is_err(),
        "32 is not a block of 256"
    );
    let sized = edited(|t| {
        t.ggml_type = 19;
        t.dims[0] = 256;
    });
    assert_eq!(sized.tensors["w"].total_bytes().unwrap(), 100);
}

#[test]
fn a_row_past_the_tensor_is_refused() {
    let weights = test_gguf::open(
        &[],
        &[
            ("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            ("after", vec![2, 1], vec![9.0, 9.0]),
        ],
    );
    assert_eq!(weights.row("w", 1).unwrap(), vec![3.0, 4.0]);
    assert!(weights.row("w", 2).is_err());
    assert!(weights.min_blocks("w", 2).is_err());
    assert!(weights.row_range("w", 2).is_err());
}

#[test]
fn a_row_of_too_many_values_is_refused() {
    let weights = edited(|t| t.dims[0] = MAX_ROW_VALUES + 32);
    let error = weights.row("w", 0).unwrap_err();
    assert!(CannotAudit::caused(&error), "{error:#}");
    let error = weights.each_row("w", 0..1, &mut |_, _| {}).unwrap_err();
    assert!(CannotAudit::caused(&error), "{error:#}");
}

/// A remote model: its header is held, and every tensor byte is fetched.
/// `w` has 1024 rows of 32 values, 128 bytes each.
fn remote() -> WeightFile {
    let (header, data) = test_gguf::header(&[], &[("w", vec![32, 1024], GGML_F32)]);
    let size = header.len() as u64 + data;
    WeightFile::open(Box::new(test_gguf::Remote::new(header, size))).unwrap()
}

#[test]
fn reads_of_a_remote_file_stop_at_the_budget() {
    let weights = remote();
    // the header has its own limit, and takes nothing from the budget
    assert_eq!(weights.fetched_bytes(), 0);
    weights.set_fetch_budget(300);
    weights.row("w", 0).unwrap();
    weights
        .prefetch(&[weights.row_range("w", 1).unwrap()])
        .unwrap();
    assert_eq!(weights.fetched_bytes(), 256);
    let error = weights.row("w", 2).unwrap_err();
    assert!(CannotAudit::caused(&error), "{error:#}");
    assert!(error.to_string().contains("budget of 300 bytes"), "{error}");
    let range = weights.tensor_range("w").unwrap();
    assert!(weights.prefetch(&[range]).is_err());
    assert!(weights.min_blocks("w", 3).is_err());
    assert!(weights.each_row("w", 4..5, &mut |_, _| {}).is_err());
    assert!(weights.stream_rows("w", 4..5, &mut |_, _| {}).is_err());
    // what was refused was not spent
    assert_eq!(weights.fetched_bytes(), 256);
    // the next audit against the open file starts over
    weights.reset_fetch_budget();
    weights.stream_rows("w", 4..6, &mut |_, _| {}).unwrap();
    assert_eq!(weights.fetched_bytes(), 256);
    assert!(weights.row("w", 9).is_err());
}

#[test]
fn the_default_budget_holds_a_whole_output_matrix() {
    let weights = remote();
    weights.stream_rows("w", 0..1024, &mut |_, _| {}).unwrap();
    assert_eq!(weights.fetched_bytes(), 1024 * 128);
    const { assert!(DEFAULT_FETCH_BUDGET >= 8 << 30) };
}

#[test]
fn a_local_file_spends_no_budget() {
    let weights = test_gguf::open(&[], &[("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])]);
    weights.set_fetch_budget(0);
    assert_eq!(weights.row("w", 1).unwrap(), vec![3.0, 4.0]);
    weights
        .prefetch(&[weights.tensor_range("w").unwrap()])
        .unwrap();
    assert_eq!(weights.fetched_bytes(), 0);
}

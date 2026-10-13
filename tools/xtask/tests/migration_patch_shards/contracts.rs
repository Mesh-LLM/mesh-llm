#[path = "input_contract.rs"]
mod input_contract;
#[path = "section_contract.rs"]
mod section_contract;

use super::{FamilyPatches, ShardError, encode_family_shards};
use serde_json::{Value, json};

const AFMOE: &[u8] = include_bytes!("../fixtures/migration/rewriter-patch-shards/afmoe.patch");
const SHARED: &[u8] =
    include_bytes!("../fixtures/migration/rewriter-patch-shards/deepseek2--mistral4.patch");
const RWKV6: &[u8] = include_bytes!("../fixtures/migration/rewriter-patch-shards/rwkv6.patch");

fn diff(patch: &[u8]) -> &[u8] {
    let start = patch
        .windows(11)
        .position(|bytes| bytes == b"diff --git ")
        .unwrap();
    patch[start..].strip_suffix(b"-- \n2.54.0\n\n").unwrap()
}

fn fixture_map() -> Value {
    serde_json::from_slice(include_bytes!(
        "../fixtures/migration/rewriter-patch-shards/family-map.json"
    ))
    .unwrap()
}

fn fixture_manifest() -> Value {
    serde_json::from_slice(include_bytes!(
        "../fixtures/migration/rewriter-patch-shards/certified.json"
    ))
    .unwrap()
}

fn mixed_diff() -> Vec<u8> {
    [
        b"diff --git a/src/models/unowned.cpp b/elsewhere\nGIT binary patch\nliteral 0\nHcmV?d00001\n\n".as_slice(),
        diff(RWKV6),
        diff(SHARED),
        diff(AFMOE),
    ]
    .concat()
}

fn encode(diff: &[u8]) -> FamilyPatches {
    encode_family_shards(diff, &fixture_map(), &fixture_manifest()).unwrap()
}

#[test]
fn migration_patch_shards_mail_matches_checked_in_legacy_bytes() {
    let input = mixed_diff();

    let result = encode(&input);

    for (shard, expected, digest) in [
        (
            &result.shards[0],
            AFMOE,
            "f24da42643df886f3d854c46722ddafe0713d046a712547b976c77ece30f9b95",
        ),
        (
            &result.shards[1],
            SHARED,
            "cfe4e22719a7b3597ac5067d86eb7d21528af6d7be1b3ec9ca0f8c52c69ccd26",
        ),
        (
            &result.shards[2],
            RWKV6,
            "513f9d910dc2e57461f42a2eebe8a903fca411a3ad2b30ca5a11cb897e3640e0",
        ),
    ] {
        assert_eq!(shard.bytes, expected);
        assert_eq!(shard.sha256, digest);
    }
}

#[test]
fn migration_patch_shards_series_orders_owner_tuples_before_unmapped() {
    let input = mixed_diff();

    let result = encode(&input);

    assert_eq!(result.series, b"0001-family-afmoe.patch\n0002-family-deepseek2--mistral4.patch\n0003-family-rwkv6.patch\n0004-family-unmapped.patch\n");
    assert_eq!(result.shards[1].families, ["deepseek2", "mistral4"]);
    assert_eq!(
        result.shards[2].sources,
        ["src/models/rwkv6.cpp", "src/models/rwkv6qwen2.cpp"]
    );
    assert!(result.shards[3].families.is_empty());
}

#[test]
fn migration_patch_shards_sort_sections_without_reordering_combined_diff() {
    let original = diff(RWKV6);
    let second = original
        .windows(11)
        .enumerate()
        .skip(1)
        .find(|(_, bytes)| *bytes == b"diff --git ")
        .unwrap()
        .0;
    let reversed = [&original[second..], &original[..second]].concat();

    let result = encode(&reversed);

    assert_eq!(result.shards[0].bytes, RWKV6);
    assert_eq!(diff(&result.combined.bytes), reversed);
}

#[test]
fn migration_patch_shards_raw_diff_digest_reuses_verified_encoder_fixture() {
    let input = include_bytes!("../fixtures/migration/rewriter-patch/two-model.diff");
    let expected = include_bytes!("../fixtures/migration/rewriter-patch/two-model.patch");

    let result = encode(input);

    assert_eq!(result.combined.bytes, expected);
    assert_eq!(
        result.combined.diff_sha256,
        "c1bf5a727845c2ab193c43dc8884812906fcc5f4b30a9a8e55a4358ff6a5c9ce"
    );
}

#[test]
fn migration_patch_shards_series_escapes_unmapped_unicode_and_controls() {
    let input =
        "diff --git a/src/models/caf\u{e9}\u{7f}\u{1f680}\t\r.cpp b/other\r\n-hunk\r\n+new\r\n";

    let result = encode(input.as_bytes());

    let json = std::str::from_utf8(&result.series_json).unwrap();
    assert!(json.contains(r#""src/models/caf\u00e9\u007f\ud83d\ude80\t\r.cpp""#));
    assert!(json.ends_with("\n}\n"));
    assert_eq!(diff(&result.shards[0].bytes), input.as_bytes());
}

#[test]
fn migration_patch_shards_tuple_prefix_sorts_before_its_extension() {
    let input = b"diff --git a/src/models/a.cpp b/a\nA\ndiff --git a/src/models/b.cpp b/b\nB\n";
    let map = json!({"schema_version":1,"families":{
        "a":["src/models/a.cpp","src/models/b.cpp"], "b":["src/models/b.cpp"]
    }});

    let result = encode_family_shards(input, &map, &json!({"models":[]})).unwrap();

    assert_eq!(
        result.series,
        b"0001-family-a.patch\n0002-family-a--b.patch\n"
    );
}

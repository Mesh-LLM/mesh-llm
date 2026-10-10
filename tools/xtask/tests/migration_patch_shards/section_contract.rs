use super::super::PatchError;
use super::{AFMOE, ShardError, diff, encode, encode_family_shards, fixture_manifest, fixture_map};

#[test]
fn migration_patch_shards_reject_empty_or_malformed_utf8_diffs() {
    for (input, expected) in [
        (b"".as_slice(), PatchError::EmptyDiff),
        (b"\xff", PatchError::InvalidUtf8),
    ] {
        let result = encode_family_shards(input, &fixture_map(), &fixture_manifest());

        assert!(matches!(result, Err(ShardError::Patch(error)) if error == expected));
    }
}

#[test]
fn migration_patch_shards_reject_prefix_or_absent_model_header() {
    for input in [
        b"prefix\ndiff --git a/src/models/a.cpp b/a\n".as_slice(),
        b"diff --git a/include/a.h b/include/a.h\n",
        b"diff --git a/src/models/ b/a\n",
        b"diff --git a/src/models/a.cpp b/\n",
    ] {
        let result = encode_family_shards(input, &fixture_map(), &fixture_manifest());

        assert!(matches!(result, Err(ShardError::DiffFormat)));
    }
}

#[test]
fn migration_patch_shards_reject_repeated_left_source_even_with_different_destination() {
    let input =
        b"diff --git a/src/models/a.cpp b/a\nA\ndiff --git a/src/models/a.cpp b/renamed\nB\n";

    let result = encode_family_shards(input, &fixture_map(), &fixture_manifest());

    assert!(
        matches!(result, Err(ShardError::RepeatedSource(source)) if source == "src/models/a.cpp")
    );
}

#[test]
fn migration_patch_shards_keep_binary_text_and_unmatched_headers_in_section() {
    let suffix = b"diff --git a/include/other.h b/include/other.h\nGIT binary patch\nliteral 0\nHcmV?d00001\n\n";
    let input = [diff(AFMOE), suffix].concat();

    let result = encode(&input);

    assert_eq!(result.shards.len(), 1);
    assert_eq!(diff(&result.shards[0].bytes), input);
}

#[test]
fn migration_patch_shards_header_source_matches_space_only_exclusion() {
    for source in ["strange.ext", "nested/a.cpp", "a\tb", "a\rb", "a\nb"] {
        let source = format!("src/models/{source}");
        let input = format!("diff --git a/{source} b/unrelated path\r\nbody\r\n");

        let result = encode(input.as_bytes());

        assert_eq!(result.shards[0].sources, [source]);
        assert_eq!(diff(&result.shards[0].bytes), input.as_bytes());
    }
}

#[test]
fn migration_patch_shards_ignore_unanchored_and_malformed_later_headers() {
    let input = b"diff --git a/src/models/a.cpp b/a\ncontext diff --git a/src/models/b.cpp b/b\ndiff --git a/src/models/bad name b/b\nbody";

    let result = encode(input);

    assert_eq!(result.shards[0].sources, ["src/models/a.cpp"]);
    assert_eq!(diff(&result.shards[0].bytes), input);
}

#[test]
fn migration_patch_shards_accept_header_at_end_without_final_lf() {
    let input = b"diff --git a/src/models/header-only b/other";

    let result = encode(input);

    assert_eq!(diff(&result.shards[0].bytes), input);
}

#[test]
fn migration_patch_shards_preserve_complete_legacy_two_section_control() {
    let first = b"diff --git a/src/models/a.cpp b/src/models/a.cpp\n--- a/src/models/a.cpp\n+++ b/src/models/a.cpp\n@@ -1 +1 @@\n-a\n+b\n";
    let second = b"diff --git a/src/models/b.cpp b/src/models/b.cpp\n--- a/src/models/b.cpp\n+++ b/src/models/b.cpp\n@@ -1 +1 @@\n-c\n+d\n";
    let input = [second.as_slice(), first].concat();

    let result = encode(&input);

    assert_eq!(
        diff(&result.shards[0].bytes),
        [first.as_slice(), second].concat()
    );
    assert_eq!(diff(&result.combined.bytes), input);
}

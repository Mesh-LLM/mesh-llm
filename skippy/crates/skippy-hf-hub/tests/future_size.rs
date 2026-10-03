//! Regression tests for the size of the futures returned by builder `send()` finishers.
//!
//! The download/upload/sync builders compile to async state machines tens of kilobytes large.
//! Awaiting them inline inside consumer async fns embeds that state machine at every nesting
//! level, growing the future and deepening its type until downstream crates hit rustc's
//! `error: queries overflow the depth limit!` — even crates that never name hf-hub types. The
//! heavyweight builders therefore box their inner future (`crate::util::boxed_future`) so the
//! `send()` future stays small and type-erased; these tests pin that behavior.
//!
//! Sizes before boxing were 26-30 KB per builder; boxed they are a few hundred bytes (the
//! builder parameters plus a pinned pointer). The threshold leaves headroom for new parameters
//! while still catching an accidentally unboxed state machine.

use hf_hub::HFClient;
use hf_hub::repository::{AddSource, CommitOperation, HFRepository, RepoTypeModel};

const BOXED_SEND_FUTURE_LIMIT: usize = 2048;

fn client() -> HFClient {
    HFClient::builder()
        .endpoint("http://localhost:9")
        .token("hf_test")
        .cache_enabled(false)
        .build()
        .expect("client construction is offline")
}

fn assert_small(name: &str, size: usize) {
    println!("{name}: {size} bytes");
    assert!(
        size < BOXED_SEND_FUTURE_LIMIT,
        "{name} send() future is {size} bytes (limit {BOXED_SEND_FUTURE_LIMIT}); \
         its inner future is no longer boxed — see crate::util::boxed_future"
    );
}

#[test]
fn boxed_send_futures_stay_small() {
    let client = client();
    let repo = client.model("user", "repo");
    let bucket = client.bucket("user", "bucket");

    assert_small(
        "download_file",
        size_of_val(&repo.download_file().filename("f").send()),
    );
    assert_small(
        "download_file_stream",
        size_of_val(&repo.download_file_stream().filename("f").send()),
    );
    assert_small(
        "download_file_to_bytes",
        size_of_val(&repo.download_file_to_bytes().filename("f").send()),
    );
    assert_small(
        "snapshot_download",
        size_of_val(&repo.snapshot_download().send()),
    );
    assert_small(
        "create_commit",
        size_of_val(
            &repo
                .create_commit()
                .operations(vec![CommitOperation::delete("f")])
                .commit_message("m")
                .send(),
        ),
    );
    assert_small(
        "upload_file",
        size_of_val(
            &repo
                .upload_file()
                .source(AddSource::Bytes("x".into()))
                .path_in_repo("f")
                .send(),
        ),
    );
    assert_small(
        "upload_folder",
        size_of_val(&repo.upload_folder().folder_path(".").send()),
    );
    assert_small(
        "delete_file",
        size_of_val(&repo.delete_file().path_in_repo("f").send()),
    );
    assert_small(
        "delete_folder",
        size_of_val(&repo.delete_folder().path_in_repo("f").send()),
    );

    assert_small(
        "bucket.download_file_stream",
        size_of_val(&bucket.download_file_stream().remote_path("f").send()),
    );
    assert_small(
        "bucket.upload_source_files",
        size_of_val(&bucket.upload_source_files().files(vec![]).send()),
    );
    assert_small(
        "bucket.upload_files",
        size_of_val(&bucket.upload_files().files(vec![]).send()),
    );
    assert_small(
        "bucket.download_files",
        size_of_val(&bucket.download_files().files(vec![]).send()),
    );
    assert_small(
        "bucket.sync",
        size_of_val(
            &bucket
                .sync()
                .local_path(".")
                .direction(hf_hub::buckets::sync::BucketSyncDirection::Upload)
                .send(),
        ),
    );
}

// Consumer-shaped regression: a chain of nested async fns awaiting `download_file().send()`,
// mirroring how vLLM's rust/ workspace hit `queries overflow the depth limit!` in targets that
// do not even import hf-hub. Each level must add only its own stack frame, not re-embed the
// whole download state machine.

async fn fetch_level_3(repo: &HFRepository<RepoTypeModel>) -> hf_hub::HFResult<std::path::PathBuf> {
    repo.download_file()
        .filename("model.safetensors")
        .send()
        .await
}

async fn fetch_level_2(repo: &HFRepository<RepoTypeModel>) -> hf_hub::HFResult<std::path::PathBuf> {
    fetch_level_3(repo).await
}

async fn fetch_level_1(repo: &HFRepository<RepoTypeModel>) -> hf_hub::HFResult<std::path::PathBuf> {
    fetch_level_2(repo).await
}

#[test]
fn nested_consumer_future_stays_shallow() {
    let client = client();
    let repo = client.model("user", "repo");

    let nested = fetch_level_1(&repo);
    let size = size_of_val(&nested);
    println!("three-level nested consumer future: {size} bytes");
    assert!(
        size < 2 * BOXED_SEND_FUTURE_LIMIT,
        "nested consumer future is {size} bytes"
    );

    fn assert_send<F: Send>(_: &F) {}
    assert_send(&nested);
}

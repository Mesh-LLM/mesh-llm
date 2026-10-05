# Skippy HF Hub client

This is the Skippy-owned generic HF client, imported from the Apache-2.0
Hugging Face `hf-hub` lineage through the Mesh-LLM fork at
`c47b78835eeb98b7339718f8701adf80b8685446` (`hf-hub/`, published as
`mesh-llm-hf-hub` 1.0.2). The original license is included as `LICENSE`.
`SOURCE_PROVENANCE.json` records original and imported source hashes and local
changes. The import uses workspace versioning; the upstream source version
remains 1.0.2 in that provenance record.

Production changes beyond rustfmt are limited to boxing four inner download
futures before awaiting them and removing an uncalled private HEAD helper.
The public finishers remain async methods; their future representation and
allocation change. This restores the retained future-size tests, which also
failed against the unchanged pinned source on this toolchain. The relative
location predicate remains test-only with its classification test.

Token-precedence tests use
child-process environments instead of mutating the parent test environment;
all eight original precedence cases remain covered. The `hf_hub` library name
keeps consumer import paths and upstream examples intact; the actual Cargo
package identity and path dependency are `skippy-hf-hub`.

This source import does not establish standalone model acquisition acceptance.
Real download, cache reuse/removal, blocking/async and failure-path validation
remain part of the Skippy extraction acceptance gate.

## Original client documentation

# hf-hub

Rust client for the [Hugging Face Hub API](https://huggingface.co/docs/hub/api).

`hf-hub` provides a typed, ergonomic interface for interacting with the Hugging Face Hub from Rust. It is the Rust equivalent of the Python [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library.

Both an **async** interface (`HFClient`, on by default) and a **synchronous** interface (`HFClientSync`, enabled via the `blocking` feature) are provided. The two mirror each other method-for-method.

## Features

- **Repository operations** — query model, dataset, and space metadata; create, delete, update, and move repositories
- **File operations** — upload files and folders, download files, list repository trees, check file existence
- **Commit operations** — create commits with multiple file operations, list commit history, view diffs between revisions
- **Branch and tag management** — create and delete branches and tags, list refs
- **User and organization info** — whoami, user profiles, organization details, followers
- **Streaming pagination** — async list endpoints return `impl Stream<Item = Result<T>>` for lazy, memory-efficient iteration; blocking counterparts collect into `Vec<T>`
- **Bucket operations** — create, delete, list, and move buckets; upload, download, and delete files within buckets
- **Xet high-performance transfers** — support for Hugging Face's Xet storage backend
- **Async or blocking** — use `HFClient` with your own tokio runtime, or `HFClientSync` for synchronous callers (requires the `blocking` feature)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
hf_hub = { package = "skippy-hf-hub", version = "1.0.0" }
```

To use the synchronous interface, enable the `blocking` feature:

```toml
[dependencies]
hf_hub = { package = "skippy-hf-hub", version = "1.0.0", features = ["blocking"] }
```

## CLI Installation

The `hfrs` command-line tool provides a terminal interface to the Hub. Install it with:

```sh
cargo install --git https://github.com/huggingface/hf-hub.git hfrs
```

This builds in release mode by default. Once installed, run `hfrs --help` to see available commands.

## Quick Start

### Async

```rust,no_run
use hf_hub::HFClient;
use hf_hub::repository::RepoInfo;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    // Get model info
    let RepoInfo::Model(info) = client
        .model("openai-community", "gpt2")
        .info()
        .send()
        .await?
    else {
        unreachable!("handle type guarantees the Model variant");
    };
    println!("Model: {} (downloads: {:?})", info.id, info.downloads);

    Ok(())
}
```

### Blocking

Requires the `blocking` feature. `HFClientSync` manages a dedicated tokio runtime internally, so callers do not need their own.

```rust,ignore
use hf_hub::HFClientSync;
use hf_hub::repository::RepoInfo;

fn main() -> hf_hub::HFResult<()> {
    let client = HFClientSync::new()?;

    let RepoInfo::Model(info) = client
        .model("openai-community", "gpt2")
        .info()
        .send()?
    else {
        unreachable!("handle type guarantees the Model variant");
    };
    println!("Model: {} (downloads: {:?})", info.id, info.downloads);

    Ok(())
}
```

The blocking handles (`HFClientSync`, `HFRepositorySync`, `HFSpaceSync`, `HFBucketSync`) mirror their async counterparts method-for-method. See the `blocking_*` examples in `examples/` for runnable programs.

## Usage Examples

### List models by author

```rust,no_run
use futures::StreamExt;
use hf_hub::HFClient;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    let stream = client
        .list_models()
        .author("meta-llama")
        .limit(5_usize)
        .send()?;
    futures::pin_mut!(stream);

    while let Some(model) = stream.next().await {
        let model = model?;
        println!("{}", model.id);
    }

    Ok(())
}
```

### Work with a repository handle

```rust,no_run
use hf_hub::HFClient;
use hf_hub::repository::RepoInfo;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let repo = client.model("openai-community", "gpt2");

    let RepoInfo::Model(model_info) = repo.info().send().await? else {
        println!("error, not a model");
        return Ok(());
    };
    println!("Model: {}", model_info.id);

    let exists = repo
        .file_exists()
        .filename("config.json")
        .send()
        .await?;

    println!("config.json exists: {exists}");
    Ok(())
}
```

### Download a file

```rust,no_run
use std::path::PathBuf;
use hf_hub::HFClient;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let repo = client.model("openai-community", "gpt2");

    let path = repo
        .download_file()
        .filename("config.json")
        .local_dir(PathBuf::from("/tmp/hf-downloads"))
        .send()
        .await?;

    println!("Downloaded to: {}", path.display());
    Ok(())
}
```

### Upload a file

```rust,no_run
use hf_hub::HFClient;
use hf_hub::repository::AddSource;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let repo = client.model("your-username", "your-repo");

    let commit = repo
        .upload_file()
        .source(AddSource::Bytes(b"Hello, world!".to_vec()))
        .path_in_repo("greeting.txt")
        .commit_message("Add greeting file")
        .send()
        .await?;

    println!("Committed: {:?}", commit.oid);
    Ok(())
}
```

### Create a repository

```rust,no_run
use hf_hub::HFClient;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    let url = client
        .create_repo()
        .repo_id("your-username/new-model")
        .private(true)
        .exist_ok(true)
        .send()
        .await?;

    println!("Repository URL: {}", url.url);
    Ok(())
}
```

## Authentication

The client resolves authentication tokens in this order:

1. Explicit token via `HFClientBuilder::token()`
2. `HF_TOKEN` environment variable
3. Token file at the path specified by `HF_TOKEN_PATH`
4. Default token file at `~/.cache/huggingface/token`

Set `HF_HUB_DISABLE_IMPLICIT_TOKEN` to any non-empty value to disable automatic token resolution.

## Configuration

| Environment Variable            | Description                                            |
|---------------------------------|--------------------------------------------------------|
| `HF_ENDPOINT`                   | Hub API endpoint (default: `https://huggingface.co`)   |
| `HF_TOKEN`                      | Authentication token                                   |
| `HF_TOKEN_PATH`                 | Path to token file                                     |
| `HF_HOME`                       | Cache directory root (default: `~/.cache/huggingface`) |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | Disable automatic token loading                        |
| `HF_HUB_USER_AGENT_ORIGIN`      | Custom User-Agent origin string                        |

## Error Handling

All fallible operations return `Result<T, HFError>`. The `HFError` enum provides structured variants for common failure modes:

- `HFError::AuthRequired` — 401 response, token is missing or invalid
- `HFError::RepoNotFound` — repository does not exist or is inaccessible
- `HFError::BucketNotFound` — bucket does not exist or is inaccessible
- `HFError::EntryNotFound` — file or path does not exist in the repository or bucket
- `HFError::RevisionNotFound` — branch, tag, or commit does not exist
- `HFError::Forbidden` — 403 response, insufficient permissions
- `HFError::Conflict` — 409 response, resource already exists or conflicts
- `HFError::RateLimited` — 429 response, too many requests
- `HFError::Http` — other HTTP errors with status code, URL, and response body

## WebAssembly support

`hf-hub` compiles for `wasm32-unknown-unknown`. On wasm the API surface is
the same call shape as native — `client.model(owner, name).download_file_stream()…send().await` —
but only the streaming download path is wired up; methods that touch the
filesystem (`download_file`, `snapshot_download`, `upload_file`, the
`cache` and `buckets` modules, the `blocking` wrappers) are gated behind
`#[cfg(not(target_family = "wasm"))]`. Pure-HTTP modules (`repository`,
`spaces`, `users`) and HTTP-only methods on `HFRepository` (`exists`,
`info`, `list_*`, etc.) are exposed on both targets.

Verify locally with `./scripts/verify_wasm.sh`. CI runs the same check in
the `wasm` job of `.github/workflows/rust.yml`. The `wasm/smoke/` crate is a
tiny `wasm-bindgen` library that exercises the wasm-safe surface end-to-end
(see `wasm/smoke/src/lib.rs`).

When modifying `client`, `error`, `retry`, `xet`, or
`repository/download.rs`, keep the wasm build green — reqwest's wasm
backend has a reduced API (no `is_connect`, no `redirect()` policy) and
its streams are `!Send`, and filesystem APIs must stay behind
`#[cfg(not(target_family = "wasm"))]` gates.

## License

Apache-2.0

//! Search and startup use the real Hub adapter against an offline fixture.
use skippy_model_hf::{remote_catalog, search, startup::resolve_startup_model_source};
use skippy_model_resolver::startup::StartupModelSource;
use std::io::{Read, Write};
use std::path::Path;

#[test]
fn discovery_preserves_startup_precedence_and_search_response_fields() {
    let root = tempfile::tempdir().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        for connection in listener.incoming() {
            let mut connection = connection.unwrap();
            connection
                .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                .unwrap();
            let mut request = [0; 8192];
            let mut length = 0;
            while !request[..length]
                .windows(4)
                .any(|bytes| bytes == b"\r\n\r\n")
            {
                assert!(
                    length < request.len(),
                    "fixture request headers exceed 8192 bytes"
                );
                let read = connection.read(&mut request[length..]).unwrap();
                if read == 0 {
                    assert_eq!(length, 0, "EOF before complete fixture request headers");
                    break;
                }
                length += read;
            }
            if length == 0 {
                break;
            }
            let request = String::from_utf8_lossy(&request[..length]);
            let (status, body) = if request.starts_with("GET /api/models?") {
                let models = if request.contains("MissingOffline") {
                    serde_json::json!([])
                } else if request.contains("OfflineName") {
                    serde_json::json!([{"id": "owner/OfflineName"}])
                } else {
                    serde_json::json!([
                        {"id":"owner/Model-7B", "downloads":42, "likes":7, "siblings":[{"rfilename":"Model-Q4_K_M.gguf", "size":4_000_000_000u64}]},
                        {"id":"owner/Model-13B", "downloads":9, "likes":3, "siblings":[{"rfilename":"Model-Q8_0.gguf", "size":8_000_000_000u64}]},
                        {"id":"owner/WithoutWeights", "siblings":[]}
                    ])
                };
                ("200 OK", models.to_string())
            } else {
                ("404 Not Found", "{}".to_string())
            };
            write!(connection, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        }
    });
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "discovery_child", "--nocapture"])
        .env("SKIPPY_DISCOVERY_FIXTURE", root.path())
        .env("HOME", root.path())
        .env("HF_HOME", root.path().join("hf"))
        .env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        .env("HF_HUB_CACHE", root.path().join("hub"))
        .env("HF_XET_CACHE", root.path().join("xet"))
        .env("HF_ENDPOINT", format!("http://{address}"))
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .env("XDG_CACHE_HOME", root.path().join("cache"))
        .env_remove("HF_TOKEN")
        .env_remove("HF_TOKEN_PATH")
        .env_remove("HUGGING_FACE_HUB_TOKEN")
        .env_remove("HF_HUB_OFFLINE")
        .env_remove("HUGGINGFACE_HUB_CACHE")
        .current_dir(root.path())
        .output()
        .unwrap();
    drop(std::net::TcpStream::connect(address).unwrap());
    server.join().unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}

#[test]
#[ignore = "run only as the discovery fixture subprocess"]
fn discovery_child() {
    let Ok(root) = std::env::var("SKIPPY_DISCOVERY_FIXTURE") else {
        return;
    };
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let _catalog = remote_catalog::set_catalog_entries_for_test(Vec::new());
            check_startup(Path::new(&root)).await;
            check_search().await;
        });
}

async fn check_startup(root: &Path) {
    let package = Path::new("hf://owner/package@rev");
    assert_eq!(
        resolve_startup_model_source(package).await.unwrap(),
        StartupModelSource::Passthrough(package.into())
    );
    let local = root.join("local.gguf");
    std::fs::write(&local, b"gguf").unwrap();
    assert_eq!(
        resolve_startup_model_source(&local).await.unwrap(),
        StartupModelSource::Existing(local.clone())
    );
    let entry = serde_json::from_value(serde_json::json!({
        "schema_version":1, "source_repo":"owner/catalog", "variants":{
            "CatalogModel":{"source":{"repo":"owner/catalog","revision":"pinned","file":"model.gguf"},"curated":{"name":"CatalogModel"}}
        }
    })).unwrap();
    skippy_model_hf::store::local::remember_model_ref_path("CatalogModel", &local);
    {
        let _catalog = remote_catalog::set_catalog_entries_for_test(vec![entry]);
        let source = resolve_startup_model_source(Path::new("CatalogModel"))
            .await
            .unwrap();
        let StartupModelSource::Catalog(model) = source else {
            panic!("catalog must precede installed cache: {source:?}");
        };
        assert_eq!(
            (
                model.repo.as_str(),
                model.revision.as_deref(),
                model.file.as_str(),
                model.name.as_str()
            ),
            (
                "owner/catalog",
                Some("pinned"),
                "model.gguf",
                "CatalogModel"
            )
        );
    }
    assert_eq!(
        resolve_startup_model_source(Path::new("CatalogModel.gguf"))
            .await
            .unwrap(),
        StartupModelSource::Installed(local.clone())
    );
    skippy_model_hf::store::local::remember_model_ref_path("owner/installed", &local);
    assert_eq!(
        resolve_startup_model_source(Path::new("owner/installed"))
            .await
            .unwrap(),
        StartupModelSource::Installed(local)
    );
    assert_eq!(
        resolve_startup_model_source(Path::new("OfflineName:Q4_K_M"))
            .await
            .unwrap(),
        StartupModelSource::Download {
            reference: "owner/OfflineName:Q4_K_M".into(),
            bare_name_fallback: true
        }
    );
    assert_eq!(
        resolve_startup_model_source(Path::new("MissingOffline"))
            .await
            .unwrap(),
        StartupModelSource::MissingBareName
    );
    assert_eq!(
        resolve_startup_model_source(Path::new("owner/download"))
            .await
            .unwrap(),
        StartupModelSource::Download {
            reference: "owner/download".into(),
            bare_name_fallback: false
        }
    );
}

#[derive(Default)]
struct Observer(Vec<search::SearchProgress>);
impl search::SearchObserver for Observer {
    fn progress(&mut self, progress: search::SearchProgress) {
        self.0.push(progress);
    }
    fn inspection_warning(&mut self, error: &anyhow::Error) -> anyhow::Result<()> {
        panic!("unexpected inspection error: {error}")
    }
}

async fn check_search() {
    for (sort, expected) in [
        (
            search::Sort::Trending,
            vec!["owner/Model-7B", "owner/Model-13B"],
        ),
        (
            search::Sort::ParametersDesc,
            vec!["owner/Model-13B", "owner/Model-7B"],
        ),
        (
            search::Sort::ParametersAsc,
            vec!["owner/Model-7B", "owner/Model-13B"],
        ),
    ] {
        let mut observer = Observer::default();
        let results = search::search_huggingface(
            "Model",
            3,
            search::ArtifactFilter::Gguf,
            sort,
            &mut observer,
        )
        .await
        .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|hit| hit.repo_id.as_str())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            observer.0,
            vec![
                search::SearchProgress::SearchingHub,
                search::SearchProgress::InspectingRepos {
                    completed: 0,
                    total: 3
                },
                search::SearchProgress::InspectingRepos {
                    completed: 1,
                    total: 3
                },
                search::SearchProgress::InspectingRepos {
                    completed: 2,
                    total: 3
                },
                search::SearchProgress::InspectingRepos {
                    completed: 3,
                    total: 3
                }
            ]
        );
        let payload = search::projection::search_huggingface_json_payload(
            "Model",
            search::ArtifactFilter::Gguf,
            sort,
            &results,
            10_000_000_000,
        );
        assert_eq!(payload["machine"]["vram_bytes"], 10_000_000_000u64);
        assert_eq!(payload["results"][0]["type"], "gguf");
        assert_eq!(payload["results"][0]["variant_count"], 1);
        assert_eq!(payload["results"][0]["capabilities"]["text"], true);
        assert!(payload["results"][0].get("show").is_none());
    }
    let results = search::search_huggingface(
        "Model",
        1,
        search::ArtifactFilter::Gguf,
        search::Sort::ParametersDesc,
        &mut Observer::default(),
    )
    .await
    .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].repo_id, "owner/Model-13B");
}

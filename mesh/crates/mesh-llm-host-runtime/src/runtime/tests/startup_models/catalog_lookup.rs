use crate::runtime::startup_models::catalog_model_unless_local_file;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn recording_lookup(
    called: &Arc<AtomicBool>,
) -> impl FnOnce(&str) -> Option<crate::models::remote_catalog::RemoteCatalogModel> + Send + 'static
{
    let called = Arc::clone(called);
    move |_| {
        called.store(true, Ordering::SeqCst);
        None
    }
}

#[tokio::test]
async fn a_local_model_file_is_not_looked_up_in_the_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.gguf");
    std::fs::write(&model, b"GGUF").unwrap();
    let called = Arc::new(AtomicBool::new(false));

    let found = catalog_model_unless_local_file(
        model.to_string_lossy().into_owned(),
        recording_lookup(&called),
    )
    .await;

    assert!(found.is_none());
    assert!(!called.load(Ordering::SeqCst), "the catalog lookup ran");
}

#[tokio::test]
async fn a_model_name_is_looked_up_in_the_catalog() {
    let called = Arc::new(AtomicBool::new(false));

    catalog_model_unless_local_file("Qwen3-8B-Q4_K_M".to_string(), recording_lookup(&called)).await;

    assert!(called.load(Ordering::SeqCst));
}

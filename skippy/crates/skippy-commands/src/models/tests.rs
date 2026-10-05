use super::*;
use serial_test::serial;
use std::{
    io::{self, Write},
    sync::{Arc, LazyLock, Mutex},
};

static CAPTURE: LazyLock<Arc<Mutex<Vec<u8>>>> = LazyLock::new(|| Arc::new(Mutex::new(Vec::new())));
struct Capture;
impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        CAPTURE.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn capture() -> Box<dyn Write + Send> {
    Box::new(Capture)
}
fn context(program: &'static str, root: &std::path::Path) -> ModelCommandContext {
    ModelCommandContext {
        program,
        cache_root: root.to_path_buf(),
        fit_budget_bytes: 24_000_000_000,
        terminal_progress: false,
        byte_progress: None,
        console_out: capture,
        console_err: || Box::new(io::sink()),
        machine_out: capture,
    }
}
async fn render(
    command: &cli::ModelsCommand,
    program: &'static str,
    root: &std::path::Path,
) -> String {
    CAPTURE.lock().unwrap().clear();
    run(command, context(program, root)).await.unwrap();
    String::from_utf8(CAPTURE.lock().unwrap().clone()).unwrap()
}
#[tokio::test]
#[serial]
async fn prune_preserves_schema_and_uses_the_invoking_command() {
    let root = tempfile::tempdir().unwrap();
    let command = cli::ModelsCommand::Prune {
        yes: false,
        json: true,
    };
    let mesh: serde_json::Value =
        serde_json::from_str(&render(&command, "mesh-llm", root.path()).await).unwrap();
    let skippy: serde_json::Value =
        serde_json::from_str(&render(&command, "skippy", root.path()).await).unwrap();
    assert_eq!(skippy["apply"], "skippy models prune --yes");
    assert_eq!(mesh["apply"], "mesh-llm models prune --yes");
    assert_eq!(skippy["cache_dir"], mesh["cache_dir"]);
    assert_eq!(skippy["dry_run"], true);
    assert!(!root.path().join("skippy-stages").exists());
}
#[tokio::test]
#[serial]
async fn cleanup_uses_the_same_preview_schema_in_both_products() {
    let root = tempfile::tempdir().unwrap();
    let command = cli::ModelsCommand::Cleanup {
        unused_since: Some("7d".into()),
        yes: false,
        json: true,
    };
    let mesh: serde_json::Value =
        serde_json::from_str(&render(&command, "mesh-llm", root.path()).await).unwrap();
    let skippy: serde_json::Value =
        serde_json::from_str(&render(&command, "skippy", root.path()).await).unwrap();
    assert_eq!(mesh, skippy);
    assert_eq!(skippy["dry_run"], true);
    assert_eq!(skippy["mesh_managed_only"], true);
    assert_eq!(skippy["unused_since"], "7d");
}
#[tokio::test]
#[serial]
async fn human_cleanup_matches_mesh_tables_and_uses_skippy_hints() {
    let root = tempfile::tempdir().unwrap();
    let command = cli::ModelsCommand::Cleanup {
        unused_since: Some("12h".into()),
        yes: false,
        json: false,
    };
    let mesh = render(&command, "mesh-llm", root.path()).await;
    let skippy = render(&command, "skippy", root.path()).await;
    assert_eq!(mesh.replace("mesh-llm models ", "skippy models "), skippy);
    assert!(skippy.contains("skippy models cleanup --unused-since 12h --yes"));
}
#[tokio::test]
#[serial]
async fn installed_model_tables_and_json_keep_the_same_fields() {
    use super::{
        capabilities::ModelCapabilities,
        formatters::{InstalledRow, models_formatter},
    };
    let root = tempfile::tempdir().unwrap();
    let mut row = InstalledRow {
        name: "Demo Q4_K_M".into(),
        model_ref: "org/Demo-GGUF:Q4_K_M".into(),
        show_command: Some("mesh-llm models show org/Demo-GGUF:Q4_K_M".into()),
        download_command: Some("mesh-llm models download org/Demo-GGUF:Q4_K_M".into()),
        delete_command: "mesh-llm models delete org/Demo-GGUF:Q4_K_M".into(),
        path: root.path().join("Demo-Q4_K_M.gguf"),
        size: Some(4_000_000_000),
        layer_count: None,
        catalog_model: None,
        capabilities: ModelCapabilities::default(),
        managed_by_mesh: true,
        last_used_at: None,
    };
    for json in [false, true] {
        let mut results = Vec::new();
        for program in ["mesh-llm", "skippy"] {
            row.show_command = Some(format!("{program} models show org/Demo-GGUF:Q4_K_M"));
            row.download_command = Some(format!("{program} models download org/Demo-GGUF:Q4_K_M"));
            row.delete_command = format!("{program} models delete org/Demo-GGUF:Q4_K_M");
            CAPTURE.lock().unwrap().clear();
            output::scope(context(program, root.path()), async {
                models_formatter(json)
                    .render_installed(std::slice::from_ref(&row))
                    .unwrap();
            })
            .await;
            results.push(String::from_utf8(CAPTURE.lock().unwrap().clone()).unwrap());
        }
        assert_eq!(
            results[0].replace("mesh-llm models ", "skippy models "),
            results[1]
        );
        if json {
            let value: serde_json::Value = serde_json::from_str(&results[1]).unwrap();
            let model = &value["results"][0];
            assert_eq!(model["ref"], "org/Demo-GGUF:Q4_K_M");
            assert_eq!(model["mesh_managed"], true);
            assert!(model.get("capabilities").is_some());
        } else {
            assert!(results[1].contains("skippy models delete org/Demo-GGUF:Q4_K_M"));
            assert!(results[1].contains("size: 4.0GB"));
        }
    }
}

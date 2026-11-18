use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn inventory_binary_generates_and_checks_inventory() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "mesh-cli-inventory-{}-{nonce}.json",
        std::process::id()
    ));
    let binary = env!("CARGO_BIN_EXE_mesh-llm-cli-inventory");

    let generated = Command::new(binary)
        .arg(&path)
        .output()
        .expect("run inventory generator");
    assert!(
        generated.status.success(),
        "generation failed: {}",
        String::from_utf8_lossy(&generated.stderr)
    );

    let checked = Command::new(binary)
        .arg("--check")
        .arg(&path)
        .output()
        .expect("check generated inventory");
    std::fs::remove_file(&path).expect("remove generated inventory");
    assert!(
        checked.status.success(),
        "inventory check failed: {}",
        String::from_utf8_lossy(&checked.stderr)
    );
}

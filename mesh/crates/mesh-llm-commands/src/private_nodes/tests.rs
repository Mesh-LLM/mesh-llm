use super::*;
use tempfile::TempDir;

const STUDIO: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const MINI: &str = "2222222222222222222222222222222222222222222222222222222222222222";

fn config_in(dir: &TempDir) -> std::path::PathBuf {
    dir.path().join("config.toml")
}

fn write_nodes(path: &std::path::Path, nodes: &[String]) -> Result<()> {
    update_nodes(path, |listed| {
        *listed = nodes.to_vec();
        Ok(())
    })
}

fn peers() -> Vec<ConnectedNode> {
    vec![
        ConnectedNode {
            node_id: STUDIO.to_string(),
            label: Some("studio".to_string()),
            hostname: Some("Mics-Studio.local".to_string()),
        },
        ConnectedNode {
            node_id: MINI.to_string(),
            label: None,
            hostname: Some("mini.local".to_string()),
        },
    ]
}

#[test]
fn adding_by_id_writes_the_id_and_keeps_the_rest_of_the_file() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    std::fs::write(&path, "# keep this comment\n[telemetry]\nenabled = true\n").expect("seed");

    add(Some(&path), &STUDIO.to_ascii_uppercase(), None, 3131).expect("add");

    let raw = std::fs::read_to_string(&path).expect("config");
    assert!(raw.contains("# keep this comment"), "{raw}");
    assert!(raw.contains("[private]"), "{raw}");
    assert_eq!(listed_nodes(Some(&path)).expect("list"), vec![STUDIO]);
}

#[test]
fn adding_by_name_saves_the_connected_node_id() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    let peers = peers();

    add(Some(&path), "studio", Some(&peers), 3131).expect("by label");
    add(Some(&path), "mini", Some(&peers), 3131).expect("by hostname without domain");
    add(Some(&path), "STUDIO", Some(&peers), 3131).expect("again is a no-op");

    assert_eq!(listed_nodes(Some(&path)).expect("list"), vec![STUDIO, MINI]);
}

#[test]
fn a_name_needs_a_running_node_and_a_match() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    let peers = peers();

    let offline = add(Some(&path), "studio", None, 3131)
        .unwrap_err()
        .to_string();
    assert!(offline.contains("port 3131"), "{offline}");
    let unknown = add(Some(&path), "laptop", Some(&peers), 3131)
        .unwrap_err()
        .to_string();
    assert!(unknown.contains("no connected node"), "{unknown}");
    assert!(!path.exists(), "nothing written on failure");
}

#[test]
fn a_name_shared_by_two_nodes_is_refused() {
    let mut peers = peers();
    peers[1].label = Some("studio".to_string());
    let error = resolve("studio", Some(&peers), 3131)
        .unwrap_err()
        .to_string();
    assert!(error.contains("more than one"), "{error}");
}

#[test]
fn remove_takes_an_id_a_prefix_or_a_connected_name() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    let peers = peers();
    write_nodes(&path, &[STUDIO.to_string(), MINI.to_string()]).expect("seed");

    remove(Some(&path), "2222", None, 3131).expect("by prefix offline");
    assert_eq!(listed_nodes(Some(&path)).expect("list"), vec![STUDIO]);

    let error = remove(Some(&path), "studio", None, 3131).unwrap_err();
    assert!(format!("{error:#}").contains("port 3131"), "{error:#}");

    remove(Some(&path), "studio", Some(&peers), 3131).expect("by name");
    assert!(listed_nodes(Some(&path)).expect("list").is_empty());
    let parsed = mesh_llm_config::load_config(Some(&path)).expect("still valid");
    assert!(parsed.private.is_empty());
}

#[test]
fn removing_an_unlisted_node_fails_without_writing() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    write_nodes(&path, &[STUDIO.to_string()]).expect("seed");
    let before = std::fs::read_to_string(&path).expect("config");

    assert!(remove(Some(&path), MINI, None, 3131).is_err());
    assert!(remove(Some(&path), "", None, 3131).is_err());

    assert_eq!(std::fs::read_to_string(&path).expect("config"), before);
}

#[test]
fn status_peers_without_a_full_node_id_are_skipped() {
    let status = serde_json::json!({
        "peers": [
            {"id": "111111", "node_id": STUDIO, "connected": true,
             "hostname": "studio.local", "owner": {"node_label": "studio"}},
            {"id": "222222", "node_id": MINI, "connected": false, "hostname": "mini.local"},
            {"id": "333333", "hostname": "old-node"},
            {"id": "444444", "node_id": "444444", "connected": true, "owner": {}},
        ]
    });
    assert_eq!(
        connected_nodes_in(&status),
        vec![ConnectedNode {
            node_id: STUDIO.to_string(),
            label: Some("studio".to_string()),
            hostname: Some("studio.local".to_string()),
        }]
    );
}

#[test]
fn list_works_without_a_running_node() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    list(Some(&path), None, 3131, false).expect("empty list");
    write_nodes(&path, &[STUDIO.to_string()]).expect("seed");
    list(Some(&path), Some(&peers()), 3131, true).expect("json list");
}

#[test]
fn a_prefix_and_a_name_that_name_different_nodes_are_refused() {
    let cafe = format!("cafe{}", "0".repeat(60));
    let peers = vec![ConnectedNode {
        node_id: MINI.to_string(),
        label: Some("cafe".to_string()),
        hostname: None,
    }];
    let nodes = vec![cafe.clone(), MINI.to_string()];

    let error = listed_match(&nodes, "cafe", Some(&peers))
        .unwrap_err()
        .to_string();
    assert!(error.contains("more than one"), "{error}");
    assert_eq!(
        listed_match(&nodes, "cafe", None).expect("prefix only"),
        cafe
    );
}

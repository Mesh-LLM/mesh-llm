use super::*;

fn inputs(paths: &[&str]) -> BTreeSet<String> {
    paths.iter().map(|path| (*path).to_owned()).collect()
}

fn watched(paths: &[&str]) -> Vec<String> {
    paths.iter().map(|path| (*path).to_owned()).collect()
}

#[test]
fn windows_warmer_paths_actual_warmer_watches_every_key_input() {
    check(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
}

#[test]
fn windows_warmer_paths_equal_and_glob_paths_cover_inputs() {
    let covered = unwatched(
        &inputs(&[
            "Cargo.lock",
            "scripts/lib/lld.sh",
            "tools/xtask/src/native_policy/**",
        ]),
        &watched(&["Cargo.lock", "scripts/**", "tools/xtask/src/**"]),
    )
    .unwrap();
    assert!(covered.is_empty(), "{covered:?}");
}

#[test]
fn windows_warmer_paths_report_each_unwatched_input() {
    let missing = unwatched(
        &inputs(&["Cargo.lock", "scripts/**", "tools/xtask/src/**"]),
        &watched(&["scripts/build-llama.sh", "tools/xtask/src/native_policy/**"]),
    )
    .unwrap();
    assert_eq!(missing, ["Cargo.lock", "scripts/**", "tools/xtask/src/**"]);
}

#[test]
fn windows_warmer_paths_need_a_push_path_list() {
    for source in [
        "on:\n  workflow_dispatch: {}\n",
        "on:\n  push:\n    branches: [main]\n",
    ] {
        let document = workflow_yaml::parse(source).unwrap();
        assert!(push_paths(&document).is_err(), "{source}");
    }
}

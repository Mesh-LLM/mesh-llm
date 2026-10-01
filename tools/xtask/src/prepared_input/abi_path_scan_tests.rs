use super::*;

fn arguments(stage: &Path, values: &[&str]) -> Vec<String> {
    std::iter::once(stage.to_str().expect("UTF-8 fixture").to_owned())
        .chain(values.iter().map(|value| (*value).to_owned()))
        .collect()
}

#[test]
fn migration_prepared_inputs_path_scan_rejects_binary_path_when_spanning_read_boundary() {
    let directory = tempfile::tempdir().expect("scratch");
    let mut bytes = vec![0xff; 1024 * 1024 - 3];
    bytes.extend_from_slice(b"/checkout/producer\x00\xfe");
    std::fs::write(directory.path().join("archive.a"), bytes).expect("binary");
    let args = arguments(directory.path(), &["/checkout/producer"]);

    let result = run(&args);

    assert_eq!(
        result.expect_err("leak").0,
        "portable static ABI retained producer-local path \"/checkout/producer\" in archive.a"
    );
}

#[test]
fn migration_prepared_inputs_path_scan_selects_pathlib_component_order_when_multiple_files_leak() {
    let directory = tempfile::tempdir().expect("scratch");
    std::fs::create_dir(directory.path().join("a")).expect("nested");
    std::fs::write(directory.path().join("a/z.a"), b"/second /first").expect("nested file");
    std::fs::write(directory.path().join("a.a"), b"/second").expect("file");
    std::fs::write(directory.path().join("z.a"), b"/first").expect("file");
    let args = arguments(directory.path(), &["/first", "/second"]);

    let result = run(&args);

    assert_eq!(
        result.expect_err("leak").0,
        "portable static ABI retained producer-local path \"/first\" in a/z.a"
    );
}

#[test]
fn migration_prepared_inputs_path_scan_keeps_argument_priority_when_empty_and_duplicate_paths_exist()
 {
    let directory = tempfile::tempdir().expect("scratch");
    std::fs::write(directory.path().join("file"), b"/early /late").expect("binary");
    let args = arguments(directory.path(), &["", "/late", "/late", "/early", ""]);

    let result = run(&args);

    assert_eq!(
        result.expect_err("leak").0,
        "portable static ABI retained producer-local path \"/late\" in file"
    );
}

#[test]
fn migration_prepared_inputs_path_scan_formats_python_repr_when_forbidden_path_has_quotes_and_controls()
 {
    let directory = tempfile::tempdir().expect("scratch");
    std::fs::write(directory.path().join("file"), b"/it's\n\t\\\x01").expect("binary");
    let args = arguments(directory.path(), &["/it's\n\t\\\u{1}"]);

    let result = run(&args);

    assert_eq!(
        result.expect_err("leak").0,
        "portable static ABI retained producer-local path \"/it's\\n\\t\\\\\\u{1}\" in file"
    );
}

#[test]
fn migration_prepared_inputs_path_scan_accepts_arbitrary_binary_when_no_forbidden_bytes_match() {
    let directory = tempfile::tempdir().expect("scratch");
    std::fs::write(directory.path().join("file"), b"\xff\x00/allowed\xfe").expect("binary");
    let args = arguments(directory.path(), &["/producer"]);

    let result = run(&args);

    assert_eq!(result.ok(), Some(String::new()));
}

#[test]
fn migration_prepared_inputs_path_scan_accepts_when_stage_is_missing() {
    let directory = tempfile::tempdir().expect("scratch");
    let args = arguments(&directory.path().join("missing"), &["/producer"]);

    let result = run(&args);

    assert_eq!(result.ok(), Some(String::new()));
}

#[test]
fn migration_prepared_inputs_path_scan_accepts_when_stage_is_a_file() {
    let directory = tempfile::tempdir().expect("scratch");
    let file = directory.path().join("file");
    std::fs::write(&file, b"/producer").expect("file");
    let args = arguments(&file, &["/producer"]);

    let result = run(&args);

    assert_eq!(result.ok(), Some(String::new()));
}

#[test]
fn migration_prepared_inputs_path_scan_accepts_when_forbidden_operands_are_empty() {
    let directory = tempfile::tempdir().expect("scratch");
    std::fs::write(directory.path().join("file"), b"/producer").expect("file");
    let args = arguments(directory.path(), &["", ""]);

    let result = run(&args);

    assert_eq!(result.ok(), Some(String::new()));
}

#[test]
fn migration_prepared_inputs_path_scan_matches_utf8_bytes_when_forbidden_path_is_unicode() {
    let directory = tempfile::tempdir().expect("scratch");
    std::fs::write(directory.path().join("file"), "\u{fffd}/é/producer").expect("file");
    let args = arguments(directory.path(), &["/é/producer"]);

    let result = run(&args);

    assert_eq!(
        result.expect_err("leak").0,
        "portable static ABI retained producer-local path \"/é/producer\" in file"
    );
}

#[test]
fn migration_prepared_inputs_path_scan_accepts_when_no_forbidden_operands_are_supplied() {
    let directory = tempfile::tempdir().expect("scratch");
    std::fs::write(directory.path().join("file"), b"/producer").expect("file");
    let args = arguments(directory.path(), &[]);

    let result = run(&args);

    assert_eq!(result.ok(), Some(String::new()));
}

#[cfg(unix)]
#[test]
fn migration_prepared_inputs_path_scan_follows_file_symlink_when_target_is_outside_stage() {
    let directory = tempfile::tempdir().expect("scratch");
    let stage = directory.path().join("stage");
    std::fs::create_dir(&stage).expect("stage");
    let outside = directory.path().join("outside");
    std::fs::write(&outside, b"/producer").expect("outside");
    std::os::unix::fs::symlink(&outside, stage.join("link.a")).expect("symlink");
    let args = arguments(&stage, &["/producer"]);

    let result = run(&args);

    assert_eq!(
        result.expect_err("leak").0,
        "portable static ABI retained producer-local path \"/producer\" in link.a"
    );
}

#[cfg(unix)]
#[test]
fn migration_prepared_inputs_path_scan_skips_directory_symlink_and_broken_link_when_present() {
    let directory = tempfile::tempdir().expect("scratch");
    let stage = directory.path().join("stage");
    let outside = directory.path().join("outside");
    std::fs::create_dir(&stage).expect("stage");
    std::fs::create_dir(&outside).expect("outside");
    std::fs::write(outside.join("leak"), b"/producer").expect("leak");
    std::os::unix::fs::symlink(&outside, stage.join("directory")).expect("directory symlink");
    std::os::unix::fs::symlink(outside.join("missing"), stage.join("broken"))
        .expect("broken symlink");
    let args = arguments(&stage, &["/producer"]);

    let result = run(&args);

    assert_eq!(result.ok(), Some(String::new()));
}

#[cfg(unix)]
#[test]
fn migration_prepared_inputs_path_scan_descends_when_stage_itself_is_directory_symlink() {
    let directory = tempfile::tempdir().expect("scratch");
    let actual = directory.path().join("actual");
    let stage = directory.path().join("stage");
    std::fs::create_dir(&actual).expect("actual");
    std::fs::write(actual.join("file"), b"/producer").expect("file");
    std::os::unix::fs::symlink(&actual, &stage).expect("stage symlink");
    let args = arguments(&stage, &["/producer"]);

    let result = run(&args);

    assert_eq!(
        result.expect_err("leak").0,
        "portable static ABI retained producer-local path \"/producer\" in file"
    );
}

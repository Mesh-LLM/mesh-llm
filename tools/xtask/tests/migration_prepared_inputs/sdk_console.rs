use crate::support::{Case, Legacy, Scratch, TestResult, assert_output, snapshot};
use std::fs;

#[path = "sdk_console/entities.rs"]
mod entities;
#[path = "sdk_console/failures.rs"]
mod failures;

fn invocation(scratch: &Scratch, command: &str) -> Case {
    let root = scratch.join("console");
    let root = root.to_str().expect("UTF-8 fixture directory");
    let legacy = match command {
        "sdk-console-manifest" => Legacy::Heredoc("scripts/package-sdk-console-assets.sh", 1),
        "sdk-console-verify" => Legacy::Heredoc("scripts/verify-sdk-console-assets.sh", 1),
        _ => panic!("fixture command"),
    };
    Case::same(&[command], &[root], legacy)
}

fn fixture(index: &str, manifest: &str) -> Result<Scratch, Box<dyn std::error::Error>> {
    let scratch = Scratch::new("sdk-console")?;
    scratch.write("console/index.html", index.as_bytes())?;
    scratch.write("console/manifest.txt", manifest.as_bytes())?;
    scratch.write("console/assets/app.js", b"app")?;
    Ok(scratch)
}

fn verified(scratch: &Scratch) -> TestResult {
    let before = snapshot(&scratch.join("console"))?;
    let output = invocation(scratch, "sdk-console-verify").run(scratch.path())?;
    assert_output(
        &output,
        0,
        &format!(
            "verified console assets: {}\n",
            scratch.join("console").display()
        ),
        "",
    );
    assert_eq!(snapshot(&scratch.join("console"))?, before);
    Ok(())
}

#[test]
fn sdk_console_manifest_when_hidden_nested_files_exist() -> TestResult {
    let scratch = fixture("", "stale")?;
    scratch.write("console/.hidden/a", b"hidden")?;
    scratch.write("console/nested/.hidden/a", b"included")?;
    scratch.write("console/nested/manifest.txt", b"included")?;
    let mut before = snapshot(&scratch.join("console"))?;
    before.remove("manifest.txt");
    let output = invocation(&scratch, "sdk-console-manifest")
        .writing(&scratch.join("console/manifest.txt"))
        .run(scratch.path())?;
    assert_output(&output, 0, "", "");
    assert_eq!(
        fs::read_to_string(scratch.join("console/manifest.txt"))?,
        "assets/app.js\nindex.html\nnested/.hidden/a\nnested/manifest.txt\n"
    );
    let mut after = snapshot(&scratch.join("console"))?;
    after.remove("manifest.txt");
    assert_eq!(before, after);
    Ok(())
}

#[test]
fn sdk_console_manifest_when_zero_entries_exist() -> TestResult {
    let scratch = Scratch::new("console-empty")?;
    fs::create_dir(scratch.join("console"))?;
    let output = invocation(&scratch, "sdk-console-manifest")
        .writing(&scratch.join("console/manifest.txt"))
        .run(scratch.path())?;
    assert_output(&output, 0, "", "");
    assert_eq!(fs::read(scratch.join("console/manifest.txt"))?, b"\n");
    Ok(())
}

#[test]
fn sdk_console_verify_when_all_tags_and_duplicate_attrs_exist() -> TestResult {
    let scratch = fixture(
        "<!-- <img src=missing> --><style><img src=missing></style>\
         <script>var text = '<img src=missing>';</script>\
         <IMG SRC=missing src=assets/app.js HREF=missing href=''>\
         <a src=missing src href=assets/app.js></a>\
         <script src=assets/app.js /><img src=assets/app.js>",
        "index.html\nassets/app.js\n",
    )?;
    verified(&scratch)
}

#[test]
fn sdk_console_verify_when_local_external_and_entity_refs_exist() -> TestResult {
    let scratch = fixture(
        "<img src='assets/a&amp;b.js'><a href='assets&#47;app.js?x#f'>\
         <img src=/assets/app.js><a href='//cdn.test/x'><a href=https://cdn.test/x>\
         <a href='custom+scheme:x'><a href='data:a'><a href='mailto:a'><a href='#x'>\
         <a href='?x'><a href='/'><a href='assets/app.js;params?x'>",
        " \tindex.html\r\nassets/app.js\u{85}assets/a&b.js\n",
    )?;
    scratch.write("console/assets/a&b.js", b"entity")?;
    verified(&scratch)
}

#[test]
fn sdk_console_verify_when_css_query_and_fragment_are_present() -> TestResult {
    let scratch = fixture(
        "<link href='/assets/app.css?x#y'>",
        "index.html\nassets/app.css\n",
    )?;
    scratch.write("console/assets/app.css", b"css")?;
    verified(&scratch)
}

#[test]
fn sdk_console_verify_when_javascript_suffix_is_a_directory() -> TestResult {
    let scratch = fixture("", "index.html\n")?;
    fs::remove_file(scratch.join("console/assets/app.js"))?;
    fs::create_dir(scratch.join("console/assets/only.js"))?;
    verified(&scratch)
}

#[test]
fn sdk_console_verify_when_css_suffix_is_a_directory() -> TestResult {
    let scratch = fixture("<link href=app.css>", "index.html\napp.css\n")?;
    scratch.write("console/app.css", b"css")?;
    fs::create_dir(scratch.join("console/assets/only.css"))?;
    verified(&scratch)
}

#[test]
fn sdk_console_verify_when_manifest_contains_dot_components_and_duplicates() -> TestResult {
    let scratch = fixture("", "index.html\n./assets/app.js\nindex.html\n")?;
    verified(&scratch)
}

#[test]
fn sdk_console_verify_when_directory_spelling_needs_pathlib_normalization() -> TestResult {
    let scratch = fixture("", "index.html\n")?;
    let case = Case::same(
        &["sdk-console-verify"],
        &["./console//./"],
        Legacy::Heredoc("scripts/verify-sdk-console-assets.sh", 1),
    );
    let output = case.run(scratch.path())?;
    assert_output(&output, 0, "verified console assets: console\n", "");
    Ok(())
}

#[cfg(unix)]
#[test]
fn sdk_console_manifest_when_file_and_directory_symlinks_exist() -> TestResult {
    let scratch = fixture("<img src=assets/link.js>", "index.html\nassets/link.js\n")?;
    scratch.write("outside/a.js", b"outside")?;
    crate::support::symlink(
        &scratch.join("outside/a.js"),
        &scratch.join("console/assets/link.js"),
    )?;
    crate::support::symlink(&scratch.join("outside"), &scratch.join("console/linked"))?;
    let output = invocation(&scratch, "sdk-console-manifest")
        .writing(&scratch.join("console/manifest.txt"))
        .run(scratch.path())?;
    assert_output(&output, 0, "", "");
    assert_eq!(
        fs::read_to_string(scratch.join("console/manifest.txt"))?,
        "assets/app.js\nassets/link.js\nindex.html\n"
    );
    verified(&scratch)
}

#[cfg(unix)]
#[test]
fn sdk_console_verify_when_html_traverses_to_a_regular_file() -> TestResult {
    let scratch = fixture("<img src=../outside.js>", "index.html\n")?;
    scratch.write("outside.js", b"allowed legacy traversal")?;
    verified(&scratch)
}

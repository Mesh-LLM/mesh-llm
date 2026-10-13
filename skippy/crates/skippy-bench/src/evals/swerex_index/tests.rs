use super::*;

#[test]
fn exact_patch_preserves_surrounding_source_and_is_idempotent() {
    let original = format!("# prefix\n{OLD}\n# suffix\n");
    let patched = replacement(&original, "https://pypi.org/simple")
        .unwrap()
        .unwrap();
    assert_eq!(
        patched,
        original.replace(
            "--no-cache-dir",
            "--index-url 'https://pypi.org/simple' --no-cache-dir"
        )
    );
    assert_eq!(
        replacement(&patched, "https://pypi.org/simple").unwrap(),
        None
    );
    assert!(replacement(&patched, "https://other.invalid/simple").is_err());
    assert!(OLD.contains(r"\n\n"));
    assert!(!OLD.contains('\n'));
}

#[test]
fn unknown_mixed_and_duplicate_shapes_are_refused() {
    let patched = replacement(OLD, "https://pypi.org/simple")
        .unwrap()
        .unwrap();
    for text in [
        "unknown".to_owned(),
        format!("{OLD}{OLD}"),
        format!("{OLD}{patched}"),
        format!("{patched}{patched}"),
    ] {
        assert!(replacement(&text, "https://pypi.org/simple").is_err());
    }
}

#[test]
fn index_is_one_safe_http_token_and_errors_do_not_disclose_credentials() {
    for url in [
        "",
        "https://",
        "file:///tmp/index",
        "https://host/\nRUN evil",
        "https://host/ x",
        "https://host/'",
        "https://host/\"",
        "https://host/\\",
        "https://host/{evil}",
        "https://host/}evil",
    ] {
        assert!(replacement(OLD, url).is_err(), "accepted unsafe URL");
    }
    let secret = "secret-do-not-report";
    let error = replacement(OLD, &format!("https://user:{secret}@host/ bad")).unwrap_err();
    assert!(!error.to_string().contains(secret));
    assert!(replacement(OLD, "http://127.0.0.1:8080/simple").is_ok());
    assert!(replacement(OLD, "https://user:encoded%24pass@host/simple").is_ok());
}

#[test]
fn shell_quoted_query_and_ipv6_urls_preserve_literal_identity() {
    for url in [
        "https://host/simple?one=1&two=2",
        "http://[::1]:8080/simple?selected=[abc]",
        "https://host/$(literal);literal",
    ] {
        let patched = replacement(OLD, url).unwrap().unwrap();
        assert_eq!(
            patched,
            format!(r#"{INDEX_PREFIX}'{url}' --no-cache-dir {{PACKAGE_NAME}}\n\n""#)
        );
        assert_eq!(replacement(&patched, url).unwrap(), None);
    }
}

#[test]
fn url_length_is_bounded_before_parsing_without_disclosing_the_url() {
    let base = "https://host/";
    let at_limit = format!("{base}{}", "x".repeat(MAX_URL_BYTES - base.len()));
    assert!(replacement(OLD, &at_limit).is_ok());
    assert!(replacement(OLD, &format!("{at_limit}x")).is_err());
}

#[cfg(unix)]
mod file_admission {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{PermissionsExt, symlink},
        path::PathBuf,
    };

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "swerex-index-test-{}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&root).unwrap();
            Self(root.canonicalize().unwrap())
        }
        fn module(&self) -> PathBuf {
            self.0.join("docker.py")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn actual_patch_preserves_mode_and_repeat_does_not_rewrite() {
        let fixture = Fixture::new();
        let module = fixture.module();
        fs::write(&module, OLD).unwrap();
        fs::set_permissions(&module, fs::Permissions::from_mode(0o640)).unwrap();
        patch(&module, &fixture.0, "https://pypi.org/simple").unwrap();
        let metadata = fs::metadata(&module).unwrap();
        assert_eq!(metadata.permissions().mode() & 0o777, 0o640);
        let bytes = fs::read(&module).unwrap();
        patch(&module, &fixture.0, "https://pypi.org/simple").unwrap();
        assert_eq!(fs::read(&module).unwrap(), bytes);
        assert_eq!(
            fs::metadata(&module).unwrap().modified().unwrap(),
            metadata.modified().unwrap()
        );
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
    }

    #[test]
    fn admission_refuses_missing_symlink_directory_escape_and_oversize() {
        let fixture = Fixture::new();
        let module = fixture.module();
        assert!(patch(&module, &fixture.0, "https://pypi.org/simple").is_err());
        fs::create_dir(&module).unwrap();
        assert!(patch(&module, &fixture.0, "https://pypi.org/simple").is_err());
        fs::remove_dir(&module).unwrap();
        let target = fixture.0.join("target.py");
        fs::write(&target, OLD).unwrap();
        symlink(&target, &module).unwrap();
        assert!(patch(&module, &fixture.0, "https://pypi.org/simple").is_err());
        assert_eq!(fs::read_to_string(&target).unwrap(), OLD);
        fs::remove_file(&module).unwrap();
        let outside = Fixture::new();
        fs::write(outside.module(), OLD).unwrap();
        assert!(patch(&outside.module(), &fixture.0, "https://pypi.org/simple").is_err());
        let alias = fixture.0.join("alias");
        symlink(&outside.0, &alias).unwrap();
        assert!(
            patch(
                &alias.join("docker.py"),
                &fixture.0,
                "https://pypi.org/simple"
            )
            .is_err()
        );
        fs::write(&module, vec![b'x'; 1024 * 1024 + 1]).unwrap();
        assert!(patch(&module, &fixture.0, "https://pypi.org/simple").is_err());
        assert_eq!(fs::metadata(&module).unwrap().len(), 1024 * 1024 + 1);
    }

    #[test]
    fn changed_source_refuses_replacement_and_cleans_only_owned_stage() {
        let fixture = Fixture::new();
        let module = fixture.module();
        fs::write(&module, OLD).unwrap();
        let admitted = source::AdmittedSource::open(&module, &fixture.0).unwrap();
        fs::write(&module, "changed after admission").unwrap();
        assert!(admitted.replace(b"replacement").is_err());
        assert_eq!(
            fs::read_to_string(&module).unwrap(),
            "changed after admission"
        );
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
    }

    #[test]
    fn source_substitution_and_named_parent_replacement_are_refused() {
        let fixture = Fixture::new();
        let package = fixture.0.join("package");
        fs::create_dir(&package).unwrap();
        let module = package.join("docker.py");
        fs::write(&module, OLD).unwrap();
        let admitted = source::AdmittedSource::open(&module, &fixture.0).unwrap();
        fs::rename(&module, package.join("original.py")).unwrap();
        fs::write(&module, OLD).unwrap();
        assert!(admitted.replace(b"replacement").is_err());
        assert_eq!(fs::read_to_string(&module).unwrap(), OLD);
        let admitted = source::AdmittedSource::open(&module, &fixture.0).unwrap();
        fs::rename(&package, fixture.0.join("moved")).unwrap();
        fs::create_dir(&package).unwrap();
        fs::write(&module, OLD).unwrap();
        assert!(admitted.replace(b"replacement").is_err());
        assert_eq!(fs::read_to_string(&module).unwrap(), OLD);
        assert_eq!(
            fs::read_to_string(fixture.0.join("moved/docker.py")).unwrap(),
            OLD
        );
    }

    #[test]
    fn unsafe_url_and_unknown_shape_leave_actual_file_unchanged() {
        let fixture = Fixture::new();
        let module = fixture.module();
        for (text, url) in [
            (OLD, "https://host/a'evil"),
            ("unknown source", "https://host/simple"),
        ] {
            fs::write(&module, text).unwrap();
            assert!(patch(&module, &fixture.0, url).is_err());
            assert_eq!(fs::read_to_string(&module).unwrap(), text);
            assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
        }
    }

    #[test]
    fn rewritten_size_bound_refuses_before_staging_and_remains_read_admissible() {
        let fixture = Fixture::new();
        let module = fixture.module();
        let original = format!("{}{OLD}", "#".repeat(1024 * 1024 - OLD.len()));
        fs::write(&module, &original).unwrap();
        assert!(patch(&module, &fixture.0, "https://host/simple").is_err());
        assert_eq!(fs::read_to_string(&module).unwrap(), original);
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
        let padding = original.len() - 128;
        let shorter = format!("{}{OLD}", "#".repeat(padding - OLD.len()));
        fs::write(&module, &shorter).unwrap();
        patch(&module, &fixture.0, "https://host/simple?one=1&two=2").unwrap();
        patch(&module, &fixture.0, "https://host/simple?one=1&two=2").unwrap();
    }
}

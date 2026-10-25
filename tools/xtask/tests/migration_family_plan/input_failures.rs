use super::*;

#[test]
fn malformed_manifest_preserves_existing_outputs() {
    for raw in [
        b"{".as_slice(),
        b"\xff",
        b"{\"schema_version\":NaN}",
        b"{} trailing",
    ] {
        let manifest = temp_path("malformed.json");
        let output = temp_path("preserved.json");
        let github = temp_path("preserved.txt");
        fs::write(&manifest, raw).expect("input");
        fs::write(&output, b"plan sentinel").expect("plan");
        fs::write(&github, b"github sentinel").expect("github");
        let result = run(&[
            "--manifest",
            manifest.to_str().expect("path"),
            "--output",
            output.to_str().expect("path"),
            "--github-output",
            github.to_str().expect("path"),
        ]);
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
        assert_eq!(fs::read(&output).expect("plan"), b"plan sentinel");
        assert_eq!(fs::read(&github).expect("github"), b"github sentinel");
        for path in [manifest, output, github] {
            fs::remove_file(path).expect("cleanup");
        }
    }
}

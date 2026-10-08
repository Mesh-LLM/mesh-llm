//! Original handoff safety through the actual typed source and producer owners.
//! Shares the finite pipeline fixture; packaged native/executable bytes never run.
use super::super::admission;
use super::*;

#[test]
fn stale_native_pin_refuses_before_any_restored_checkout_is_created() {
    if isolated(
        "handoff_sealing_tests::stale_native_pin_refuses_before_any_restored_checkout_is_created",
    ) {
        return;
    }
    let fixture = Fixture::new();
    let (package, _) = fixture.pack();
    let root = fixture.consumer();
    let provenance: source::Provenance =
        serde_json::from_slice(&fs::read(package.join("llama-source.json")).unwrap()).unwrap();
    let pin = root.join("skippy/llama_cpp/upstream.txt");
    fs::write(&pin, "c".repeat(40)).unwrap();
    let target = root.join(".deps/llama.cpp");
    let result = admission::restore_native(&root, &package, &provenance, &target);
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("current pinned patch recipe")
    );
    assert!(!root.join(".deps").exists());
    assert!(!target.exists());
    assert_eq!(
        source::prepared(&fixture.root).unwrap().head,
        fixture.native
    );
    assert_eq!(
        process::text(&fixture.root, &["rev-parse", "HEAD"]).unwrap(),
        fixture.base
    );
}

fn admitted_receipt(fixture: &Fixture) -> (PathBuf, Digest) {
    let receipt = fixture.directory.path().join("handoff-receipt.json");
    let digest = producer_receipt::write(&producer_receipt::Input {
        context: fixture.context.clone(),
        root: fixture.root.clone(),
        closure: fixture.closure.clone(),
        output: receipt.clone(),
    })
    .unwrap();
    (receipt, digest)
}

#[test]
fn sealed_handoff_preserves_producer_and_refuses_untracked_source_or_replaced_artifacts() {
    if isolated(
        "handoff_sealing_tests::sealed_handoff_preserves_producer_and_refuses_untracked_source_or_replaced_artifacts",
    ) {
        return;
    }
    for mutation in ["untracked", "artifact", "producer"] {
        let fixture = Fixture::new();
        let (receipt, digest) = admitted_receipt(&fixture);
        let producer = fixture.closure.join("producer.json");
        let original = fs::read(&producer).unwrap();
        let sealed = producer_receipt::consume(
            &receipt,
            &digest,
            &fixture.context,
            &fixture.root,
            &fixture.closure,
            &fixture.base,
        )
        .unwrap();
        let sealed: serde_json::Value = serde_json::from_slice(&sealed).unwrap();
        assert_eq!(sealed["source"]["head"], fixture.base);
        assert_eq!(
            sealed["source"]["worktree_sha256"],
            Digest::of_bytes(b"").as_str()
        );
        assert_eq!(fs::read(&producer).unwrap(), original);
        match mutation {
            "untracked" => {
                fs::write(fixture.root.join("untracked-source.rs"), b"new source").unwrap()
            }
            "artifact" => fs::write(
                fixture.closure.join("cargo/debug/skippy"),
                b"replaced executable",
            )
            .unwrap(),
            "producer" => fs::write(&producer, b"replaced producer").unwrap(),
            _ => unreachable!(),
        }
        let rejected = producer_receipt::consume(
            &receipt,
            &digest,
            &fixture.context,
            &fixture.root,
            &fixture.closure,
            &fixture.base,
        );
        assert!(rejected.is_err(), "admitted handoff accepted {mutation}");
        assert_eq!(
            process::text(&fixture.root, &["rev-parse", "HEAD"]).unwrap(),
            fixture.base
        );
        if mutation != "producer" {
            assert_eq!(fs::read(&producer).unwrap(), original);
        }
    }
}

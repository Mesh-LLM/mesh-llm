#[path = "bindgen/generator.rs"]
mod generator;
#[path = "build/native_link.rs"]
mod native_link;

fn main() {
    bindings_main();
    native_link::native_link_main();
}

// ---------------------------------------------------------------------------
// Deterministic bindgen regeneration for the committed bindings in
// `src/bindings/`. Modes come from `SKIPPY_FFI_BINDGEN`:
//
// - unset: emit rerun inputs only; compile the committed bindings as-is.
// - `regen`: regenerate the committed bindings in place (write-if-changed).
// - `check`: regenerate to memory and fail the build on any diff against the
//   committed files. CI uses this to prove bindings match the pinned
//   headers. With `SKIPPY_FFI_BINDGEN_SELFTEST=1`, also runs a mutation
//   probe proving that changing a consumed native struct's layout changes
//   the generated bindings (the pre-runtime regression gate).
// ---------------------------------------------------------------------------

fn bindings_main() {
    println!("cargo:rerun-if-env-changed=SKIPPY_FFI_BINDGEN");
    println!("cargo:rerun-if-env-changed=SKIPPY_FFI_BINDGEN_SELFTEST");

    let manifest_dir = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"),
    );
    let workspace_root = manifest_dir.join("../../..");
    let wrapper_dir = manifest_dir.join("bindgen/wrapper");
    let bindings_dir = manifest_dir.join("src/bindings");

    // Header and pin/patch identity inputs. A patch-queue edit, an upstream
    // pin move, a wrapper change, or a generator change must all rerun this
    // script so `regen` refreshes the committed bindings deterministically.
    for input in [
        wrapper_dir.join("skippy_abi.h"),
        wrapper_dir.join("llama_upstream.h"),
        wrapper_dir.join("mtmd_abi.h"),
        manifest_dir.join("bindgen/generator.rs"),
        workspace_root.join("skippy/llama_cpp/upstream.txt"),
        workspace_root.join("skippy/llama_cpp/patches/model_support/series"),
        workspace_root.join("skippy/llama_cpp/patches/generated/series"),
    ] {
        println!("cargo:rerun-if-changed={}", input.display());
    }
    let patch_dir = workspace_root.join("skippy/llama_cpp/patches");
    if let Ok(entries) = std::fs::read_dir(&patch_dir) {
        let mut patches: Vec<_> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "patch"))
            .collect();
        patches.sort();
        for patch in patches {
            println!("cargo:rerun-if-changed={}", patch.display());
        }
    }
    // The patched-sha marker is written by prepare-llama.sh after the queue
    // is applied; it is the identity of the header tree bindgen consumes.
    let patched_sha = workspace_root.join(".deps/llama.cpp/.mesh-llm-patched-sha");
    println!("cargo:rerun-if-changed={}", patched_sha.display());

    let mode = std::env::var("SKIPPY_FFI_BINDGEN").unwrap_or_default();
    if mode != "regen" && mode != "check" {
        return;
    }

    let llama_root = std::env::var("LLAMA_WORKDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| workspace_root.join(".deps/llama.cpp"));
    let skippy_header = llama_root.join("include/skippy.h");
    if !skippy_header.exists() {
        panic!(
            "SKIPPY_FFI_BINDGEN={mode} requires the prepared llama.cpp checkout; \
             run `scripts/prepare-llama.sh pinned` from the workspace root \
             (missing {})",
            skippy_header.display()
        );
    }

    let generated = generator::BindingGenerator::new(&llama_root)
        .generate()
        .unwrap_or_else(|error| panic!("{error}"));

    let committed = [
        (bindings_dir.join("skippy.rs"), generated.skippy),
        (bindings_dir.join("llama.rs"), generated.llama),
        (bindings_dir.join("mtmd.rs"), generated.mtmd),
    ];

    match mode.as_str() {
        "regen" => {
            for (path, content) in &committed {
                if generator::write_if_changed(path, content)
                    .unwrap_or_else(|error| panic!("{error}"))
                {
                    println!(
                        "cargo:warning=regenerated {}; rerun the build so the \
                         updated bindings are compiled",
                        path.display()
                    );
                }
            }
        }
        "check" => {
            let mut stale = Vec::new();
            for (path, content) in &committed {
                let current = std::fs::read_to_string(path).unwrap_or_default();
                if current != *content {
                    stale.push(path.file_name().unwrap().to_string_lossy().into_owned());
                }
            }
            if !stale.is_empty() {
                panic!(
                    "skippy-ffi bindings are stale: {} no longer matches the pinned \
                     native headers (llama.cpp pin {}, patched tree {}). Regenerate \
                     with `SKIPPY_FFI_BINDGEN=regen cargo check -p skippy-ffi` after \
                     `scripts/prepare-llama.sh pinned`, review the diff, and commit \
                     the refreshed bindings together with the header change. Never \
                     hand-edit src/bindings/*.rs.",
                    stale.join(", "),
                    std::fs::read_to_string(workspace_root.join("skippy/llama_cpp/upstream.txt"))
                        .map(|pin| pin.trim().to_string())
                        .unwrap_or_else(|_| "unknown".to_string()),
                    std::fs::read_to_string(&patched_sha)
                        .map(|sha| sha.trim().to_string())
                        .unwrap_or_else(|_| "unknown".to_string()),
                );
            }
            if std::env::var("SKIPPY_FFI_BINDGEN_SELFTEST").ok().as_deref() == Some("1") {
                bindings_mutation_probe(&llama_root, &committed);
            }
            println!("cargo:warning=skippy-ffi bindings match the pinned native headers");
        }
        _ => {}
    }
}

/// Regression probe: mutating one consumed native struct's field type must
/// change the generated bindings. This is the property that makes the check
/// mode above a real pre-runtime layout gate rather than a formatting diff.
///
/// The probe copies the consumed header tree into a temporary overlay,
/// narrows one field of `skippy_kv_page_desc` (a struct that crosses the FFI
/// boundary on every KV-page export/import), and regenerates from the
/// overlay. The real checkout is never modified.
fn bindings_mutation_probe(
    llama_root: &std::path::Path,
    committed: &[(std::path::PathBuf, String)],
) {
    let committed_skippy = &committed[0].1;
    let overlay =
        std::env::temp_dir().join(format!("skippy-ffi-bindgen-probe-{}", std::process::id()));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        copy_dir(&llama_root.join("include"), &overlay.join("include"))
            .unwrap_or_else(|error| panic!("mutation probe cannot stage overlay: {error}"));
        copy_dir(
            &llama_root.join("ggml/include"),
            &overlay.join("ggml/include"),
        )
        .unwrap_or_else(|error| panic!("mutation probe cannot stage overlay: {error}"));
        copy_dir(&llama_root.join("tools/mtmd"), &overlay.join("tools/mtmd"))
            .unwrap_or_else(|error| panic!("mutation probe cannot stage overlay: {error}"));

        // Narrow `token_start` from uint64_t to uint32_t: every field after
        // it in `skippy_kv_page_desc` shifts, so the generated layout must
        // change. If this anchor drifts, retarget the probe at a field
        // skippy-ffi actually consumes.
        let state_header = overlay.join("include/skippy/state.h");
        let original = std::fs::read_to_string(&state_header).unwrap_or_else(|error| {
            panic!(
                "mutation probe cannot read {}: {error}",
                state_header.display()
            )
        });
        let needle = "uint64_t token_start;";
        if !original.contains(needle) {
            panic!(
                "mutation probe lost its anchor: `{needle}` not found in {}; \
                 update the probe to a field skippy-ffi actually consumes",
                state_header.display()
            );
        }
        std::fs::write(
            &state_header,
            original.replacen(needle, "uint32_t token_start;", 1),
        )
        .unwrap_or_else(|error| panic!("mutation probe cannot mutate overlay: {error}"));

        let probe = generator::BindingGenerator::new(&overlay)
            .generate()
            .unwrap_or_else(|error| panic!("mutation probe generation failed: {error}"));
        assert_ne!(
            probe.skippy, *committed_skippy,
            "mutation probe: narrowing a consumed native struct field did not \
             change the generated bindings; check mode is no longer sensitive \
             to layout drift"
        );

        // Prove the probe setup is sound: the untouched checkout regenerates
        // the committed bindings byte-for-byte.
        let restored = generator::BindingGenerator::new(llama_root)
            .generate()
            .unwrap_or_else(|error| panic!("mutation probe restore check failed: {error}"));
        assert_eq!(
            restored.skippy, *committed_skippy,
            "the llama.cpp checkout does not regenerate the committed bindings; \
             run check mode without the selftest to see the real diff first"
        );
    }));
    let _ = std::fs::remove_dir_all(&overlay);
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

//! Inert transport fixtures use real GGUF headers, not a quantization oracle.
use super::*;
use contract::{Input, ToolKind, Workflow};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
fn pin_file(path: &Path, bytes: &[u8]) -> admission::Artifact {
    std::fs::write(path, bytes).unwrap();
    admission::Artifact {
        path: path.into(),
        sha256: admission::digest(bytes),
    }
}
fn gguf_bytes() -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes
}
#[cfg(unix)]
fn fixture(mode: &str) -> (tempfile::TempDir, Input, PathBuf) {
    use std::os::unix::fs::PermissionsExt as _;
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().canonicalize().unwrap();
    let source = base.join("source");
    std::fs::create_dir_all(source.join("BF16")).unwrap();
    let root = base.join("evidence");
    std::fs::create_dir(&root).unwrap();
    let target = base.join("target");
    let work = base.join("work");
    let data = gguf_bytes();
    let parts = (1..=2)
        .map(|i| {
            pin_file(
                &source
                    .join("BF16")
                    .join(format!("model-{i:05}-of-00002.gguf")),
                &data,
            )
        })
        .collect();
    let outputs = (1..=2)
        .map(|i| admission::Artifact {
            path: target
                .join("Q4_K_M")
                .join(format!("model-{i:05}-of-00002.gguf")),
            sha256: admission::digest(&data),
        })
        .collect();
    let golden = base.join("independent-golden.gguf");
    pin_file(&golden, &data);
    let recipe = pin_file(
        &base.join("tensor-types.txt"),
        b"blk.0.attn_q.weight=Q4_K\n",
    );
    let manifest=pin_file(&base.join("quant-manifest.json"),&serde_json::to_vec(&json!({"schema_version":1,"kind":"QUANTIZE_GGUF","source":source,"source_prefix":"BF16","target":target,"target_prefix":"Q4_K_M","output_basename":"model","expected_splits":2,"window_size":1,"quant":"Q4_K_M","output_type":null,"tensor_type_file":recipe.path,"tensor_type_recipe":null})).unwrap());
    let runtime = pin_file(
        &base.join("runtime.so"),
        b"finite fixture runtime identity only",
    );
    let loader = pin_file(&base.join("loader"), b"#!/bin/sh\nexit 0\n");
    std::fs::set_permissions(&loader.path, std::fs::Permissions::from_mode(0o700)).unwrap();
    let verify=serde_json::to_string(&json!({"artifact":{"root":target,"prefix":"Q4_K_M","basename":"model","expected_splits":2,"completed_count":2,"complete":true},"llama_load":{"model":target.join("Q4_K_M/model-00001-of-00002.gguf"),"llama_cli":loader.path,"success":true,"status_code":0}})).unwrap();
    let script = format!(
        "#!/bin/sh\nset -eu\ncase \"$1\" in\nrun-quant)\n  printf '%s\\n' \"$@\" >> '{base}/argv'\n  mkdir -p '{work}' '{target}/Q4_K_M'\n  n=1; if test -f '{work}/counter'; then n=2; fi\n  printf '%s' \"$n\" > '{work}/counter'\n  if test '{mode}' = nonzero; then exit 47; fi\n  if test '{mode}' = held; then printf 'owned-held-started\\n'; while :; do sleep 1; done; fi\n  name=$(printf 'model-%05d-of-00002.gguf' \"$n\")\n  cp '{golden}' '{target}/Q4_K_M/'\"$name\"\n  if test '{mode}' = wrong; then printf wrong >> '{target}/Q4_K_M/'\"$name\"; fi\n  printf '{{\"event\":\"quant_window\"}}\\n';;\nstatus)\n  n=$(cat '{work}/counter'); missing=$((2-n)); complete=false; if test \"$n\" = 2; then complete=true; fi\n  printf '{{\"expected_splits\":2,\"completed_count\":%s,\"missing_count\":%s,\"complete\":%s}}\\n' \"$n\" \"$missing\" \"$complete\";;\nverify-job) printf '%s\\n' '{verify}';;\n*) exit 48;;\nesac\n",
        base = base.display(),
        work = work.display(),
        target = target.display(),
        golden = golden.display()
    );
    let tool = pin_file(&base.join("supplied-quantizer"), script.as_bytes());
    std::fs::set_permissions(&tool.path, std::fs::Permissions::from_mode(0o700)).unwrap();
    let tool_source = pin_file(
        &base.join("source-attribution.json"),
        b"fixture source identity, not a real quantizer release",
    );
    (
        temp,
        Input {
            schema_version: 1,
            workflow: Workflow::Quantize,
            tool_kind: ToolKind::SuppliedWindowQuantizer,
            tool,
            tool_source,
            source_revision: "a".repeat(40),
            profile_version: "inert-fixture-only".into(),
            runtime,
            loader,
            manifest,
            tensor_recipe: recipe,
            source_root: source,
            target_root: target,
            work_root: work,
            source_prefix: "BF16".into(),
            target_prefix: "Q4_K_M".into(),
            basename: "model".into(),
            source_parts: parts,
            golden_outputs: outputs,
            timeout_seconds: 30,
        },
        root,
    )
}
#[cfg(unix)]
#[test]
fn quant_probe_consumes_two_owned_windows_status_golden_bytes_and_pinned_verifier() -> DynResult<()>
{
    let (temp, input, root) = fixture("success");
    let mut evidence = json!({});
    execute(
        &input,
        &root,
        Instant::now() + Duration::from_secs(30),
        &Cancellation::default(),
        &mut evidence,
    )?;
    assert_eq!(evidence["progress-1"]["completed_count"], 1);
    assert_eq!(evidence["progress-2"]["completed_count"], 2);
    assert_eq!(evidence["tool_profile_qualified"], false);
    assert_eq!(evidence["remote_mutation_performed"], false);
    let argv = std::fs::read_to_string(temp.path().join("argv"))?;
    assert_eq!(
        argv,
        [input.quant_args().join("\n"), input.quant_args().join("\n")].join("\n") + "\n"
    );
    assert_eq!(input.workflow.job_allowance_seconds(), 259200);
    assert_eq!(Workflow::QuantizeAndPackage.job_allowance_seconds(), 345600);
    Ok(())
}
#[cfg(unix)]
#[test]
fn quant_probe_refuses_current_tool_wrong_golden_nonzero_and_source_drift_without_fallback()
-> DynResult<()> {
    for mode in ["wrong", "nonzero"] {
        let (_temp, input, root) = fixture(mode);
        let mut evidence = json!({});
        assert!(
            execute(
                &input,
                &root,
                Instant::now() + Duration::from_secs(30),
                &Cancellation::default(),
                &mut evidence
            )
            .is_err()
        );
        assert!(evidence.get("window-1").is_some());
        assert!(evidence.get("window-2").is_none());
    }
    let (temp, mut input, root) = fixture("success");
    input.tool_kind = ToolKind::CurrentNativeQuantizer;
    assert!(
        execute(
            &input,
            &root,
            Instant::now() + Duration::from_secs(30),
            &Cancellation::default(),
            &mut json!({})
        )
        .is_err()
    );
    assert!(!temp.path().join("argv").exists());
    assert!(!input.target_root.exists());
    input.tool_kind = ToolKind::SuppliedWindowQuantizer;
    std::fs::write(&input.tensor_recipe.path, b"changed recipe")?;
    assert!(
        execute(
            &input,
            &root,
            Instant::now() + Duration::from_secs(30),
            &Cancellation::default(),
            &mut json!({})
        )
        .is_err()
    );
    assert!(!temp.path().join("argv").exists());
    assert!(!input.target_root.exists());
    Ok(())
}
#[cfg(unix)]
#[test]
fn quant_probe_owned_held_child_cancellation_retains_failure_and_prevents_second_window()
-> DynResult<()> {
    let (_temp, input, root) = fixture("held");
    let cancel = Cancellation::default();
    let peer = cancel.clone();
    let observed = root.clone();
    let trigger = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(5);
        let started = loop {
            if std::fs::read_to_string(observed.join("window-1-stdout.log"))
                .is_ok_and(|s| s.contains("owned-held-started"))
            {
                break true;
            }
            if Instant::now() >= until {
                break false;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        peer.cancel();
        started
    });
    let mut evidence = json!({});
    let result = execute(
        &input,
        &root,
        Instant::now() + Duration::from_secs(15),
        &cancel,
        &mut evidence,
    );
    let started = trigger.join().unwrap();
    assert!(started);
    assert!(result.is_err());
    assert!(evidence.get("window-1").is_some());
    assert!(evidence.get("window-2").is_none());
    assert_eq!(evidence["window-1"]["cleanup"]["complete"], true);
    assert_eq!(evidence["window-1"]["cleanup"]["failure_present"], false);
    assert_eq!(
        evidence["window-1"]["stdout"]["line_capture_complete"],
        true
    );
    assert_eq!(
        evidence["window-1"]["stderr"]["line_capture_complete"],
        true
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn quant_probe_closed_manifest_roster_and_workspace_refuse_before_owned_tool_launch()
-> DynResult<()> {
    let (temp, mut input, _root) = fixture("success");
    input.target_root = input.source_root.join("nested");
    assert!(input.validate().is_err());
    assert!(!temp.path().join("argv").exists());
    let (_temp, input, root) = fixture("success");
    let mut value: Value =
        serde_json::from_slice(&admission::read(&input.manifest.path, 1048576)?)?;
    value["window_size"] = json!(2);
    let bytes = serde_json::to_vec(&value)?;
    std::fs::write(&input.manifest.path, &bytes)?;
    let mut changed = input.clone();
    changed.manifest.sha256 = admission::digest(&bytes);
    assert!(
        execute(
            &changed,
            &root,
            Instant::now() + Duration::from_secs(30),
            &Cancellation::default(),
            &mut json!({})
        )
        .is_err()
    );
    assert!(!changed.target_root.exists());
    assert!(!changed.work_root.exists());
    let mut closed = serde_json::to_value(&changed)?;
    closed["publish_confirmed"] = json!(true);
    assert!(serde_json::from_value::<Input>(closed).is_err());
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(
        execute(
            &changed,
            &root,
            Instant::now() + Duration::from_secs(30),
            &cancel,
            &mut json!({})
        )
        .is_err()
    );
    assert!(
        execute(
            &changed,
            &root,
            Instant::now(),
            &Cancellation::default(),
            &mut json!({})
        )
        .is_err()
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn quant_probe_actual_dispatch_preserves_unqualified_observations_and_current_tool_refusal()
-> DynResult<()> {
    let (temp, input, _root) = fixture("success");
    let request = temp.path().join("request.json");
    std::fs::write(&request, serde_json::to_vec(&input)?)?;
    let output = temp.path().join("dispatch-evidence");
    super::super::run(&[
        "quantizer-window-probe".into(),
        "--input".into(),
        request.to_string_lossy().into(),
        "--output-directory".into(),
        output.to_string_lossy().into(),
    ])?;
    let observation: Value = serde_json::from_slice(&admission::read(
        &output.join("observations.json"),
        1048576,
    )?)?;
    assert_eq!(observation["status"], "OBSERVATIONS_ONLY");
    assert_eq!(observation["tool_profile_qualified"], false);
    assert_eq!(observation["completed_job"], false);
    assert_eq!(observation["phase_error"], false);
    assert_eq!(observation["workflow_allowance_seconds"], 259200);
    let mut unsupported = input;
    unsupported.tool_kind = ToolKind::CurrentNativeQuantizer;
    std::fs::write(&request, serde_json::to_vec(&unsupported)?)?;
    let refused = temp.path().join("refused-evidence");
    assert!(
        super::super::run(&[
            "quantizer-window-probe".into(),
            "--input".into(),
            request.to_string_lossy().into(),
            "--output-directory".into(),
            refused.to_string_lossy().into()
        ])
        .is_err()
    );
    assert!(!refused.exists());
    Ok(())
}

#[test]
fn quant_probe_durable_finalization_retains_observations_on_late_cancel_deadline_and_prior_failure()
{
    use crate::process::{
        self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    };
    for mode in [
        "success",
        "cancel",
        "deadline",
        "prior",
        "write-failure",
        "finish-failure",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let report = process::supervise(&ProcessSpec {
            executable: std::env::current_exe().unwrap(),
            arguments: ["--ignored", "--exact", "automation::hf_certify::quantization_probe::tests::quant_probe_finalization_worker", "--nocapture"].map(|s| Arg::Public(s.into())).to_vec(),
            cwd: root.clone(), environment: [("QUANT_FINAL_ROOT".into(), Arg::Public(root.clone().into_os_string())), ("QUANT_FINAL_MODE".into(), Arg::Public(mode.into()))].into_iter().collect(),
        }, &Limits { execution: Duration::from_secs(10), graceful_shutdown:Duration::from_secs(1), forced_shutdown:Duration::from_secs(1), retained_bytes_per_stream:65536, readiness:Readiness::None, completion:Completion::Exit }, &Cancellation::default(), OutputFiles::default()).unwrap();
        assert!(
            report.success()
                && report.failure.is_none()
                && report.cleanup.complete
                && !report.cleanup.forced
                && !report.cleanup.graceful_signal_failed
                && report.cleanup.failure.is_none()
        );
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert!(
            [&report.stdout, &report.stderr]
                .iter()
                .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
        );
        let durable: Value =
            serde_json::from_slice(&std::fs::read(root.join("observations.json")).unwrap())
                .unwrap();
        assert_eq!(durable["phase_error"], mode != "success");
        assert_eq!(durable["window-1"]["exit_code"], 0);
        assert_eq!(durable["tool_profile_qualified"], false);
        assert_eq!(durable["completed_job"], false);
        assert_eq!(
            durable["interrupt_finish_failed"],
            matches!(mode, "cancel" | "finish-failure")
        );
        if mode == "cancel" {
            assert_eq!(durable["terminal_cancelled"], true);
        }
        if mode == "deadline" {
            assert_eq!(durable["terminal_deadline_expired"], true);
        }
    }
}
#[test]
#[ignore = "owned subprocess helper for actual Interrupt finalization"]
fn quant_probe_finalization_worker() {
    let root = PathBuf::from(std::env::var_os("QUANT_FINAL_ROOT").unwrap());
    let mode = std::env::var("QUANT_FINAL_MODE").unwrap();
    let path = root.join("observations.json");
    let interrupt = Interrupt::install().unwrap();
    let cancel = interrupt.cancellation();
    let until = Instant::now() + Duration::from_millis(if mode == "deadline" { 30 } else { 5000 });
    let mut evidence = json!({"status":"OBSERVATIONS_ONLY", "tool_profile_qualified":false,
        "completed_job":false,"window-1":{"exit_code":0},"remote_mutation_performed":false});
    let phase = if mode == "prior" {
        Err("original failure".into())
    } else {
        Ok(())
    };
    let result = final_publication::finish_with(
        &mut evidence,
        &path,
        until,
        interrupt,
        phase,
        &mut || {
            let pending: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            assert_eq!(pending["window-1"]["exit_code"], 0);
            if mode == "cancel" {
                cancel.cancel();
            }
            if mode == "deadline" {
                std::thread::sleep(Duration::from_millis(40));
            }
            if mode == "write-failure" {
                return Err("local output refused".into());
            }
            Ok(())
        },
        |scope| {
            scope.finish()?;
            if mode == "finish-failure" {
                Err("finish failed".into())
            } else {
                Ok(())
            }
        },
    );
    assert_eq!(result.is_ok(), mode == "success");
}
#[cfg(unix)]
#[test]
fn quant_probe_loader_mutable_workspace_overlap_refuses_before_tool_launch() {
    let (_temp, mut input, root) = fixture("success");
    input.loader.path = input.work_root.join("loader");
    assert!(input.validate().is_err());
    assert!(!input.work_root.exists());
    assert!(!root.join("argv.log").exists());
}

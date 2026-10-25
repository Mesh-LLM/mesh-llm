#![cfg(unix)]
//! Exercise the existing facade with actual typed quant preparation, without submission.
use super::*;
#[test]
fn quant_facade_prepares_both_parent_budgets_without_credentials_or_submission() {
    for combined in [false, true] {
        let (v, m, p) = super::super::quantization::tests::fixture(combined);
        let request = json!({"schema_version":1,"namespace":"owner","worker_input":v,"mounts":m,"cpu_plan":p});
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input.json");
        std::fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
        let output = tmp.path().join("prepared");
        let invoke = |extra: &[&str]| {
            let mut args: Vec<std::ffi::OsString> = vec![
                "model-package-generic-jobs".into(),
                "prepare".into(),
                "--input".into(),
                input.clone().into_os_string(),
                "--output-directory".into(),
                output.clone().into_os_string(),
            ];
            args.extend(extra.iter().map(|s| std::ffi::OsString::from(*s)));
            run_args(args)
        };
        assert!(invoke(&["--confirm-submission"]).is_err());
        assert!(!output.exists());
        assert!(invoke(&[]).unwrap());
        let result: Value =
            serde_json::from_slice(&std::fs::read(output.join("result.json")).unwrap()).unwrap();
        let declaration: Value =
            serde_json::from_slice(&std::fs::read(output.join("declaration.json")).unwrap())
                .unwrap();
        assert_eq!(result["status"], "PREPARED");
        assert_eq!(result["submitted"], false);
        assert_eq!(
            declaration["timeout_seconds"],
            if combined { 345600 } else { 259200 }
        );
        assert!(!output.join("submitted.json").exists());
        assert_eq!(
            std::fs::read(output.join("worker-input.json")).unwrap(),
            serde_json::to_vec(&request["worker_input"]).unwrap()
        );
        assert!(!declaration["image_observed"].as_bool().unwrap());
    }
}

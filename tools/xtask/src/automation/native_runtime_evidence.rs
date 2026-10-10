//! Admission of completed native runtime-event evidence consumed by CI.
use crate::command::DynResult;
use std::{fs::File, io::Read, path::Path};

const INPUT_LIMIT: usize = 1024 * 1024;

fn validate(text: &str) -> DynResult<()> {
    let mut lines = text.lines();
    if lines.next() != Some("executed") {
        return Err("native runtime-event evidence does not claim completed execution".into());
    }
    let mut model_open = false;
    let mut reporter_clear = false;
    for line in lines {
        model_open |= line
            .strip_prefix("model-open:")
            .is_some_and(|detail| !detail.trim().is_empty());
        reporter_clear |= line
            .strip_prefix("reporter-clear:")
            .is_some_and(|detail| !detail.trim().is_empty());
    }
    if !model_open || !reporter_clear {
        return Err(format!(
            "native runtime-event execution requires model-open and reporter-clear checkpoints: model-open={model_open}, reporter-clear={reporter_clear}"
        )
        .into());
    }
    Ok(())
}

fn verify(path: &Path) -> DynResult<()> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take((INPUT_LIMIT + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > INPUT_LIMIT {
        return Err("native runtime-event evidence exceeds 1 MiB input limit".into());
    }
    validate(std::str::from_utf8(&bytes)?)
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let [path] = args else {
        return Err("usage: cargo xtool automation native-runtime-evidence FILE".into());
    };
    verify(Path::new(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_native_checkpoints_accept_with_additional_real_step_evidence() {
        for text in [
            "executed\nmodel-open: single-part real model-open succeeded\nreporter-clear: returned\n",
            "executed\nexact-abi-admission: loaded\nreporter-install: true\nmodel-open: single-part real model-open succeeded\nstructured-production-callbacks: 1\nreporter-clear: returned\n",
        ] {
            validate(text).unwrap();
        }
    }

    #[test]
    fn missing_checkpoints_and_the_original_ungated_execution_claim_reject() {
        for text in [
            "executed\nblocked-when-ungated: no native symbol touched\n",
            "executed\nmodel-open: succeeded\n",
            "executed\nreporter-clear: returned\n",
            "executed\nmodel-open:\nreporter-clear: returned\n",
            "executed\nmodel-open: succeeded\nreporter-clear: \n",
        ] {
            assert!(validate(text).is_err(), "{text}");
        }
    }

    #[test]
    fn empty_blocked_failed_and_embedded_success_do_not_qualify_a_ci_run() {
        for text in [
            "",
            "blocked-when-ungated: no native symbol touched\n",
            "blocked: missing bundle\n",
            "failed: model-open\nexecuted\nmodel-open: attempted\nreporter-clear: returned\n",
            "executed-fake\nmodel-open: succeeded\nreporter-clear: returned\n",
        ] {
            assert!(validate(text).is_err(), "{text}");
        }
    }

    #[test]
    fn actual_files_accept_complete_and_reject_incomplete_markers_without_rewriting() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("native evidence.txt");
        for (text, accepted) in [
            (
                "executed\nmodel-open: succeeded\nreporter-clear: returned\n",
                true,
            ),
            (
                "executed\nblocked-when-ungated: no native symbol touched\n",
                false,
            ),
        ] {
            std::fs::write(&path, text).unwrap();
            assert_eq!(verify(&path).is_ok(), accepted);
            assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        }
        assert!(verify(&directory.path().join("missing")).is_err());
    }

    #[test]
    fn bounded_utf8_input_and_closed_cli_arguments_reject_invalid_evidence() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("evidence.txt");
        for bytes in [vec![b'x'; INPUT_LIMIT + 1], vec![0xff]] {
            std::fs::write(&path, bytes).unwrap();
            assert!(verify(&path).is_err());
        }
        assert!(run(&[]).is_err());
        assert!(run(&["one".into(), "two".into()]).is_err());
    }
}

use super::tests::{matrix, policy};
use super::*;
use crate::automation::codepoint_json::parser;

fn invocation(refs: &[OsString]) -> ReplayInvocation<'_> {
    ReplayInvocation {
        python: OsStr::new("python3"),
        script: Path::new("/checkout/evals/agentic-replay.py"),
        dataset: Path::new("dataset"),
        output: Path::new("out"),
        worktree_root: None,
        refs,
    }
}

pub(super) fn scalar_argv(arguments: Vec<Argument<'_>>) -> Vec<OsString> {
    arguments
        .into_iter()
        .map(|argument| match argument {
            Argument::Os(value) => value,
            Argument::Text(text) => scalar(text.codepoints()),
            Argument::Decoded(Value::Str(text)) => scalar(text.codepoints()),
            Argument::Decoded(_) => panic!("this fixture requires a string digest"),
        })
        .collect()
}

fn scalar(codes: impl Iterator<Item = u32>) -> OsString {
    codes
        .map(char::from_u32)
        .collect::<Option<String>>()
        .expect("this fixture requires Unicode scalar strings")
        .into()
}

#[test]
fn primitive_pin_values_use_legacy_interpolation() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":7,"revision":null,"file":true,"sha256":"digest","class":"dense","native_context_tokens":131072}]"#,
    );
    let model = select(models.get("models"), &policy, &"dense".into()).expect("selection");
    let args = scalar_argv(argv(&model, &policy, &invocation(&[])));
    assert_eq!(args[4], "7@null/true");
    assert!(!args.contains(&"--worktree-root".into()));
}

#[test]
fn container_pin_values_reuse_existing_repr_without_coercing_strings() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":[7,null,true],"revision":{"pin":"a'b"},"file":"raw","sha256":"digest","class":"dense","native_context_tokens":131072}]"#,
    );
    let model = select(models.get("models"), &policy, &"dense".into()).expect("selection");
    let expected = "[7, null, true]@{\"pin\": \"a'b\"}/raw"
        .chars()
        .map(u32::from)
        .collect::<Vec<_>>();
    assert_eq!(model.reference.codepoints().collect::<Vec<_>>(), expected);
}

#[test]
fn missing_repo_precedes_missing_class() {
    let policy = policy();
    let models = matrix(r#"[{"family":"dense","native_context_tokens":131072}]"#);
    let result = select(models.get("models"), &policy, &"dense".into());
    assert_eq!(result.err(), Some(SelectionError::MissingField("repo")));
}

#[test]
fn nonstring_digest_is_retained_in_command_list() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":"r","revision":"v","file":"f","sha256":7,"class":null,"native_context_tokens":131072}]"#,
    );
    let model = select(models.get("models"), &policy, &"dense".into()).expect("selection");
    let args = argv(&model, &policy, &invocation(&[]));
    assert!(matches!(args[10], Argument::Decoded(Value::Int(7))));
    assert!(!model.recurrent);
}

#[test]
fn nonstring_digest_does_not_preempt_missing_class() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":"r","revision":"v","file":"f","sha256":7,"native_context_tokens":131072}]"#,
    );
    let result = select(models.get("models"), &policy, &"dense".into());
    assert_eq!(result.err(), Some(SelectionError::MissingField("class")));
}

#[test]
fn lone_surrogates_survive_requested_family_and_model_strings() {
    assert!(parser::parse(br#""\ud800""#).is_err());
}

#[test]
fn unknown_model_fields_do_not_become_arguments() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":"org/model","revision":"pin","file":"one.gguf","sha256":"digest","class":"dense","native_context_tokens":131072,"unknown":{"extra":[null,1]}}]"#,
    );
    let model = select(models.get("models"), &policy, &"dense".into()).expect("selection");
    let args = scalar_argv(argv(&model, &policy, &invocation(&[])));
    assert_eq!(args.len(), 43);
    assert_eq!(args[4], "org/model@pin/one.gguf");
}

#[test]
fn whitespace_root_and_ordered_refs_remain_single_arguments() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":"r","revision":"v","file":"f","sha256":"digest","class":"dense","native_context_tokens":131072}]"#,
    );
    let model = select(models.get("models"), &policy, &"dense".into()).expect("selection");
    let refs = [
        " main='HEAD'\n".into(),
        "base=old\tvalue".into(),
        " main='HEAD'\n".into(),
    ];
    let invocation = ReplayInvocation {
        worktree_root: Some(OsStr::new(" \t\n")),
        ..invocation(&refs)
    };
    let args = scalar_argv(argv(&model, &policy, &invocation));
    assert_eq!(
        &args[15..23],
        &[
            "--worktree-root",
            " \t\n",
            "--ref",
            " main='HEAD'\n",
            "--ref",
            "base=old\tvalue",
            "--ref",
            " main='HEAD'\n"
        ]
        .map(OsString::from)
    );
}

#[cfg(unix)]
#[test]
fn unix_encoding_rejects_nonstring_digest_only_at_encoding_boundary() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":"r","revision":"v","file":"f","sha256":7,"class":"dense","native_context_tokens":131072}]"#,
    );
    let model = select(models.get("models"), &policy, &"dense".into()).expect("selection");
    let result = encoding::unix_utf8_argv(&argv(&model, &policy, &invocation(&[])));
    assert_eq!(
        result,
        Err(encoding::EncodingError::NonString {
            index: 10,
            kind: "int"
        })
    );
}

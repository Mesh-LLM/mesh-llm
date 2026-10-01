use super::super::types::{ProofFields, Verdict};
use super::{Field, FieldError, ValidateBuilder, quoted_string};

pub(super) fn validate_builder(
    builder: &ValidateBuilder,
    labels: (&str, &str),
    failures: &mut Vec<String>,
) {
    let (file, path) = labels;
    builder
        .file
        .errors(&format!("{path}.file"), false, failures);
    builder
        .constructor
        .errors(&format!("{path}.constructor"), false, failures);
    match &builder.verdict {
        Field::Present(Verdict::Transformable) => transformable(builder, file, failures),
        Field::Present(Verdict::UnsupportedShape) => {
            if !builder
                .unsupported_reason
                .value()
                .is_some_and(|reason| !reason.is_empty())
            {
                failures.push(format!(
                    "{file}: unsupported_shape without unsupported_reason"
                ));
            }
        }
        Field::Present(Verdict::AlreadyTransformed) => {
            ready(builder, file, "already_transformed", failures)
        }
        Field::Present(Verdict::SupportedAuxiliary) => {
            ready(builder, file, "supported_auxiliary", failures);
            scope(
                builder,
                file,
                ("supported_auxiliary", "final_stage_sidecar"),
                failures,
            );
        }
        Field::Present(Verdict::SupportedWholeModel) => {
            ready(builder, file, "supported_whole_model", failures);
            scope(
                builder,
                file,
                ("supported_whole_model", "multiple_sequential_layer_domains"),
                failures,
            );
        }
        Field::Present(Verdict::Error) => {
            failures.push(format!("{file}: error verdict present in report"))
        }
        Field::Invalid(FieldError::UnknownVerdict(verdict)) => failures.push(format!(
            "{file}: unknown verdict {}",
            quoted_string(verdict)
        )),
        Field::Missing => failures.push(format!("{file}: unknown verdict None")),
        Field::Null | Field::Invalid(_) => {
            builder
                .verdict
                .errors(&format!("{path}.verdict"), false, failures)
        }
    }
    builder
        .proof
        .errors(&format!("{path}.proof"), false, failures);
    builder.edits.errors(
        &format!("{path}.edits"),
        matches!(
            builder.verdict.value(),
            Some(
                Verdict::AlreadyTransformed
                    | Verdict::SupportedAuxiliary
                    | Verdict::SupportedWholeModel
            )
        ),
        failures,
    );
    builder
        .unsupported_reason
        .errors(&format!("{path}.unsupported_reason"), true, failures);
}

fn transformable(builder: &ValidateBuilder, file: &str, failures: &mut Vec<String>) {
    if !builder
        .constructor
        .value()
        .is_some_and(|name| !name.0.is_empty())
    {
        failures.push(format!(
            "{file}: transformable without qualified 'constructor' name"
        ));
    }
    match builder.proof.value() {
        Some(proof) => required_proof(proof, file, failures),
        None => failures.push(format!("{file}: transformable without proof block")),
    }
    if !builder.edits.value().is_some_and(|edits| !edits.is_empty()) {
        failures.push(format!("{file}: transformable with empty edit set"));
    }
}

fn required_proof(proof: &ProofFields, file: &str, failures: &mut Vec<String>) {
    for (name, missing) in [
        ("loop", matches!(proof.r#loop, Field::Missing)),
        (
            "activation_in",
            matches!(proof.activation_in, Field::Missing),
        ),
        (
            "activation_out",
            matches!(proof.activation_out, Field::Missing),
        ),
        (
            "embedding_owner",
            matches!(proof.embedding_owner, Field::Missing),
        ),
        ("output_owner", matches!(proof.output_owner, Field::Missing)),
        (
            "terminal_predicates",
            matches!(proof.terminal_predicates, Field::Missing),
        ),
        (
            "nonlocal_exits",
            matches!(proof.nonlocal_exits, Field::Missing),
        ),
        (
            "execution_scope",
            matches!(proof.execution_scope, Field::Missing),
        ),
        (
            "scope_evidence",
            matches!(proof.scope_evidence, Field::Missing),
        ),
    ] {
        if missing {
            failures.push(format!("{file}: proof missing field '{name}'"));
        }
    }
}

fn ready(builder: &ValidateBuilder, file: &str, verdict: &str, failures: &mut Vec<String>) {
    if builder.edits.value().is_some_and(|edits| !edits.is_empty()) {
        failures.push(format!("{file}: {verdict} must carry no edits"));
    }
}

fn scope(
    builder: &ValidateBuilder,
    file: &str,
    expected: (&str, &str),
    failures: &mut Vec<String>,
) {
    let (verdict, scope) = expected;
    match builder.proof.value() {
        None => failures.push(format!("{file}: {verdict} without proof block")),
        Some(proof)
            if !proof
                .execution_scope
                .value()
                .is_some_and(|value| value == scope) =>
        {
            failures.push(format!(
                "{file}: {verdict} requires execution_scope '{scope}'"
            ))
        }
        Some(proof)
            if !proof
                .scope_evidence
                .value()
                .is_some_and(|items| !items.is_empty()) =>
        {
            failures.push(format!("{file}: {verdict} without scope evidence"))
        }
        Some(_) => {}
    }
}

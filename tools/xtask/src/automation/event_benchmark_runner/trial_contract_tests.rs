use super::*;

#[test]
fn decode_only_epsilon_matches_the_component_owned_constant() {
    let syntax = syn::parse_file(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../crates/mesh-llm-commands/src/gpus/tune/benchmark/streaming.rs"
    )))
    .unwrap();
    let constants = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Const(item) if item.ident == "DECODE_ONLY_TOK_S_EPSILON_SECS" => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(constants.len(), 1);
    let syn::Expr::Lit(expression) = constants[0].expr.as_ref() else {
        panic!("component epsilon must be a literal")
    };
    let syn::Lit::Float(value) = &expression.lit else {
        panic!("component epsilon must be a float")
    };
    assert_eq!(
        value.base10_parse::<f64>().unwrap(),
        super::super::stream_metrics::DECODE_EPSILON_SECONDS
    );
}

fn wording() -> std::collections::BTreeMap<String, String> {
    let syntax = syn::parse_file(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../crates/mesh-llm-commands/src/gpus/tune/output_types.rs"
    )))
    .unwrap();
    let functions = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(item) if item.sig.ident == "benchmark_trial_unit_definition" => {
                Some(item)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(functions.len(), 1);
    let Some(syn::Stmt::Expr(syn::Expr::Struct(value), None)) = functions[0].block.stmts.last()
    else {
        panic!("component trial-unit must return its struct")
    };
    value
        .fields
        .iter()
        .map(|field| {
            let syn::Member::Named(name) = &field.member else {
                panic!("trial-unit fields must be named")
            };
            let syn::Expr::MethodCall(call) = &field.expr else {
                panic!("trial-unit fields must own literal strings")
            };
            assert_eq!(call.method, "to_string");
            assert!(call.args.is_empty());
            let syn::Expr::Lit(value) = call.receiver.as_ref() else {
                panic!("trial-unit wording must be literal")
            };
            let syn::Lit::Str(value) = &value.lit else {
                panic!("trial-unit wording must be a string")
            };
            (name.to_string(), value.value())
        })
        .collect()
}

#[test]
fn native_trial_unit_uses_the_component_owned_wording() {
    let words = wording();
    assert_eq!(words.len(), 2);
    assert_eq!(words["trial"], unit().trial);
    assert_eq!(words["pair"], unit().pair);
}

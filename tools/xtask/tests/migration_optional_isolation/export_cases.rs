use super::support::{mutate, valid};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct ExportCase {
    pub id: String,
    pub description: String,
    pub input: Option<Vec<u8>>,
    pub args: Vec<String>,
    pub initial: Vec<(String, Vec<u8>)>,
    pub directories: Vec<String>,
    pub comparison: Comparison,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub enum Comparison {
    Exact,
    InputDiagnostic,
    IoDiagnostic,
    UsageDiagnostic,
    SurrogateGap,
}

fn case(id: &str, description: &str, options: &str) -> ExportCase {
    ExportCase {
        id: id.into(),
        description: description.into(),
        input: Some(valid().into_bytes()),
        args: format!("--matrix matrix.json {options}")
            .split_whitespace()
            .map(str::to_owned)
            .collect(),
        initial: Vec::new(),
        directories: Vec::new(),
        comparison: Comparison::Exact,
    }
}

pub fn roster() -> Vec<ExportCase> {
    let both = "--json-output params.json --github-env github.env --print-shell";
    let json = "--json-output params.json";
    let env = "--github-env github.env";
    let mut cases = vec![
        case("P01", "JSON only, full historical matrix", json),
        case("P02", "environment only", env),
        case("P03", "JSON then environment then shell", both),
        case("P04", "no exports; ignore ambient GITHUB_ENV", ""),
        case("P05", "shell only", "--print-shell"),
        case(
            "P06",
            "unknown fields, recursive sorting and ASCII escaping",
            both,
        ),
        case(
            "P07",
            "float forms, negative zero, nonfinite and bool kinds",
            both,
        ),
        case("P08", "arbitrary integers and ordered concurrency", both),
        case("P09", "duplicate decoded keys, last value wins", json),
        case("P10", "append preserves binary prefix without newline", env),
        case("P11", "JSON truncates old suffix", both),
        case(
            "P12",
            "same output path writes JSON then appends env",
            "--json-output shared --github-env shared --print-shell",
        ),
        case(
            "P13",
            "JSON dash is a literal filename",
            "--json-output - --print-shell",
        ),
        case(
            "P14",
            "environment dash is a literal filename",
            "--github-env -",
        ),
    ];
    cases[0].input =
        Some(include_bytes!("../fixtures/migration/optional_replay/valid.json").to_vec());
    cases[5].input = Some(mutate(&[(r#""mode":"all""#, r#""mode":"all","z":{"z":null,"a":[true,false,"\"\\/\b\f\n\r\t\u0000\u001f\u007f\u0085é\ud83d\ude00"],"\ue000":0,"\ud83d\ude00":1},"a":{"\u0000exact-number":"42"}"#)]).into_bytes());
    cases[6].input = Some(mutate(&[
        (r#""temperature":0"#, r#""temperature":-0.0"#),
        (r#""seed":42"#, r#""seed":42.0"#),
        (r#""mode":"all""#, r#""mode":"all","numbers":[false,0,-0,0.0,-0.0,42.0000000000000001,1e-4000,-1e-4000,NaN,Infinity,-Infinity,1e400,-1e400,1e-5,1e-4,1e15,1e16,1e20,1e21,1.2345678901234567,1.0000000000000002,5e-324,2.2250738585072014e-308,1.7976931348623157e308,1000000000000000128.0,1e23]"#),
    ]).into_bytes());
    cases[7].input = Some(
        mutate(&[
            (r#""passes":2"#, r#""passes":9007199254740993"#),
            (
                r#""sessions_per_concurrency":16"#,
                r#""sessions_per_concurrency":680564733841876926926749214863536422912"#,
            ),
            (
                "[1,2,4,8]",
                "[340282366920938463463374607431768211456,1,9007199254740993,9007199254740992]",
            ),
            (
                r#""mode":"all""#,
                r#""mode":"all","negative":-340282366920938463463374607431768211457"#,
            ),
        ])
        .into_bytes(),
    );
    cases[8].input = Some(
        mutate(&[(
            r#""passes":2"#,
            r#""passes":0,"pa\u0073ses":2,"unknown":{"b":1,"a":0,"b":3}"#,
        )])
        .into_bytes(),
    );
    cases[9]
        .initial
        .push(("github.env".into(), b"PRIOR=1\r\n\xff\x00NO_LF".to_vec()));
    cases[10]
        .initial
        .push(("params.json".into(), vec![b'x'; 2048]));
    let rejected = [
        (
            "R01",
            "positive precedence",
            mutate(&[
                (r#""passes":2"#, r#""passes":false"#),
                ("[1,2,4,8]", "[1,1]"),
            ]),
        ),
        (
            "R02",
            "mode rejection",
            mutate(&[(r#""mode":"all""#, r#""mode":"final""#)]),
        ),
        (
            "R03",
            "concurrency rejection",
            mutate(&[("[1,2,4,8]", "[1,1]")]),
        ),
        (
            "R04",
            "waves rejection",
            mutate(&[(
                r#""sessions_per_concurrency":16"#,
                r#""sessions_per_concurrency":15"#,
            )]),
        ),
        (
            "R05",
            "framework rejection",
            mutate(&[
                (
                    r#""sessions_per_concurrency":16"#,
                    r#""sessions_per_concurrency":2"#,
                ),
                ("[1,2,4,8]", "[1]"),
            ]),
        ),
        (
            "R06",
            "context rejection",
            mutate(&[(
                r#""minimum_context_tokens":131072"#,
                r#""minimum_context_tokens":131071"#,
            )]),
        ),
        (
            "R07",
            "window rejection",
            mutate(&[(r#""max_isl":131072"#, r#""max_isl":32768"#)]),
        ),
        (
            "R08",
            "sampling rejection",
            mutate(&[(r#""seed":42"#, r#""seed":43"#)]),
        ),
        (
            "R09",
            "backend rejection",
            mutate(&[(r#""backend":"metal""#, r#""backend":"cpu""#)]),
        ),
        (
            "R10",
            "late ignored syntax preempts policy",
            r#"{"replay":null,"ignored":!}"#.into(),
        ),
        ("R11", "root array diagnostic difference", "[]".into()),
        ("R12", "invalid UTF8", String::new()),
    ];
    for (id, description, raw) in rejected {
        let mut entry = case(id, description, both);
        entry.input = Some(if id == "R12" {
            vec![0xff]
        } else {
            raw.into_bytes()
        });
        entry.initial = sentinels();
        if matches!(id, "R10" | "R11" | "R12") {
            entry.comparison = Comparison::InputDiagnostic;
        }
        cases.push(entry);
    }
    for (id, description, options, directory) in [
        (
            "F01",
            "JSON missing parent prevents env and stdout",
            "--json-output absent/params.json --github-env github.env --print-shell",
            None,
        ),
        (
            "F02",
            "JSON directory prevents env and stdout",
            both,
            Some("params.json"),
        ),
        (
            "F03",
            "env missing parent preserves completed JSON",
            "--json-output params.json --github-env absent/github.env --print-shell",
            None,
        ),
        (
            "F04",
            "env directory preserves completed JSON",
            both,
            Some("github.env"),
        ),
        (
            "F05",
            "both invalid; JSON attempted first",
            "--json-output absent/params.json --github-env env-dir --print-shell",
            Some("env-dir"),
        ),
        ("F06", "missing input preserves both outputs", both, None),
        (
            "F07",
            "JSON may overwrite input after complete validation",
            "--json-output matrix.json --github-env github.env --print-shell",
            None,
        ),
        (
            "F08",
            "env may append to input after JSON",
            "--json-output params.json --github-env matrix.json --print-shell",
            None,
        ),
    ] {
        let mut entry = case(id, description, options);
        entry.initial = sentinels();
        if let Some(directory) = directory {
            entry.initial.retain(|(path, _)| path != directory);
            entry.directories.push(directory.into());
        }
        if id == "F06" {
            entry.input = None;
            entry.comparison = Comparison::InputDiagnostic;
        } else if !matches!(id, "F07" | "F08") {
            entry.comparison = Comparison::IoDiagnostic;
        }
        cases.push(entry);
    }
    for (id, description, args) in [
        (
            "C01",
            "required matrix missing",
            vec!["--json-output", "params.json", "--github-env", "github.env"],
        ),
        (
            "C02",
            "JSON path missing",
            vec![
                "--matrix",
                "matrix.json",
                "--github-env",
                "github.env",
                "--json-output",
            ],
        ),
    ] {
        let mut entry = case(id, description, "");
        entry.args = args.into_iter().map(str::to_owned).collect();
        entry.initial = sentinels();
        entry.comparison = Comparison::UsageDiagnostic;
        cases.push(entry);
    }
    for (id, description, replacement) in [
        (
            "G01",
            "lone surrogate value remains an explicit gap",
            r#""mode":"all","unknown":"\ud800""#,
        ),
        (
            "G02",
            "lone surrogate key identity remains an explicit gap",
            r#""mode":"all","unknown":{"\ud800":1,"\ufffd":2}"#,
        ),
    ] {
        let mut entry = case(id, description, both);
        entry.input = Some(mutate(&[(r#""mode":"all""#, replacement)]).into_bytes());
        entry.comparison = Comparison::SurrogateGap;
        cases.push(entry);
    }
    assert_eq!(cases.len(), 38);
    cases
}

fn sentinels() -> Vec<(String, Vec<u8>)> {
    vec![
        ("params.json".into(), b"JSON_SENTINEL\n".to_vec()),
        ("github.env".into(), b"ENV_SENTINEL".to_vec()),
    ]
}

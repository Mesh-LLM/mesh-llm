pub fn cases() -> Vec<(&'static str, Vec<u8>, i32)> {
    vec![
        ("utf8-invalid-start", vec![0xff], 1),
        ("utf8-overlong", vec![0xc0, 0x80], 1),
        ("utf8-truncated", vec![0xe2, 0x82], 1),
        ("utf8-surrogate-truncated", vec![0xed, 0xa0], 1),
        ("utf8-out-of-range", vec![0xf4, 0x90, 0x80, 0x80], 1),
        ("utf16-truncated", vec![0xff, 0xfe, 0x7b], 1),
        ("utf32-truncated", vec![0xff, 0xfe, 0, 0, 0x7b], 1),
        (
            "utf32-out-of-range",
            vec![0, 0, 0xfe, 0xff, 0, 0x11, 0, 0],
            1,
        ),
        ("leading-zero", br#"{"ignored":01}"#.to_vec(), 2),
        ("incomplete-fraction", br#"{"ignored":1.}"#.to_vec(), 2),
        ("incomplete-exponent", br#"{"ignored":1e+}"#.to_vec(), 2),
        ("nonfinite-prefix", br#"{"ignored":NaNtail}"#.to_vec(), 2),
        (
            "invalid-overwritten-value",
            br#"{"ignored":"\ud800\uZZZZ","ignored":0}"#.to_vec(),
            2,
        ),
        ("malformed-suffix", br#"{}false"#.to_vec(), 2),
        (
            "integer-limit",
            format!("{{\"ignored\":{}}}", "9".repeat(4301)).into_bytes(),
            1,
        ),
    ]
}

pub struct RejectionCase {
    pub name: &'static str,
    pub baseline: Vec<u8>,
    pub malformed: Vec<u8>,
    pub code: i32,
}

pub fn valid_base_cases(raw: &[u8]) -> Vec<RejectionCase> {
    let text = std::str::from_utf8(raw).expect("ASCII manifest fixture");
    let schema = "\"schema_version\": 1";
    assert!(text.contains(schema));
    let mut cases = Vec::new();
    for (name, valid, malformed) in [
        ("valid-base-leading-zero", "0", "01"),
        ("valid-base-incomplete-fraction", "1", "1."),
        ("valid-base-incomplete-exponent", "1", "1e+"),
        ("valid-base-nonfinite-prefix", "NaN", "NaNtail"),
        (
            "valid-base-invalid-overwritten-value",
            r#""\ud800\u0000""#,
            r#""\ud800\uZZZZ""#,
        ),
    ] {
        let manifest = |token| {
            text.replacen(schema, &format!("\"schema_version\": {token}, {schema}"), 1)
                .into_bytes()
        };
        cases.push(RejectionCase {
            name,
            baseline: manifest(valid),
            malformed: manifest(malformed),
            code: 2,
        });
    }
    cases.push(RejectionCase {
        name: "valid-base-malformed-suffix",
        baseline: raw.to_vec(),
        malformed: [raw, b"false"].concat(),
        code: 2,
    });
    cases.push(RejectionCase {
        name: "valid-base-utf8-invalid-suffix",
        baseline: raw.to_vec(),
        malformed: [raw, &[0xff]].concat(),
        code: 1,
    });
    let mut utf16 = vec![0xff, 0xfe];
    utf16.extend(text.encode_utf16().flat_map(u16::to_le_bytes));
    cases.push(RejectionCase {
        name: "valid-base-utf16-odd-suffix",
        malformed: [utf16.as_slice(), &[0x7b]].concat(),
        baseline: utf16,
        code: 1,
    });
    cases
}

pub struct VerificationCase {
    pub name: &'static str,
    pub manifest: Vec<u8>,
    pub supplied: &'static [u8],
    pub code: i32,
    pub diagnostic: Option<&'static str>,
}

pub fn verification_order_cases(raw: &[u8]) -> Vec<VerificationCase> {
    let text = std::str::from_utf8(raw).expect("ASCII manifest fixture");
    let schema = "\"schema_version\": 1";
    assert!(text.contains(schema));
    let surrogate = br#"{"requested_families":"\ud800","shards":[{}]}"#;
    let mut cases = vec![
        VerificationCase {
            name: "selection-after-valid-manifest",
            manifest: raw.to_vec(),
            supplied: surrogate,
            code: 2,
            diagnostic: Some("--families must contain unique comma-separated family labels"),
        },
        VerificationCase {
            name: "codec-before-selection",
            manifest: vec![0xff],
            supplied: surrogate,
            code: 1,
            diagnostic: None,
        },
        VerificationCase {
            name: "integer-limit-before-selection",
            manifest: text
                .replacen(
                    schema,
                    &format!("\"schema_version\": {}, {schema}", "9".repeat(4301)),
                    1,
                )
                .into_bytes(),
            supplied: surrogate,
            code: 1,
            diagnostic: None,
        },
    ];
    for (name, supplied, diagnostic) in [
        (
            "shards-before-manifest-codec",
            br#"{"requested_families":"\ud800","shards":[]}"#.as_slice(),
            "plan.shards must be a nonempty list",
        ),
        (
            "selection-type-before-shards-and-codec",
            br#"{"requested_families":42,"shards":[]}"#,
            "plan.requested_families must be a string or null",
        ),
    ] {
        cases.push(VerificationCase {
            name,
            manifest: vec![0xff],
            supplied,
            code: 2,
            diagnostic: Some(diagnostic),
        });
    }
    for (name, before, after, diagnostic) in [
        (
            "policy-before-selection",
            "\"oracle\": \"local-monolithic\"",
            "\"oracle\": \"none\"",
            "full profile must use the local-monolithic oracle",
        ),
        (
            "last-model-before-selection",
            "\"family\": \"delta\"",
            "\"family\": \"zeta\"",
            "duplicate family: zeta",
        ),
    ] {
        assert!(text.contains(before));
        cases.push(VerificationCase {
            name,
            manifest: text.replacen(before, after, 1).into_bytes(),
            supplied: surrogate,
            code: 2,
            diagnostic: Some(diagnostic),
        });
    }
    cases
}

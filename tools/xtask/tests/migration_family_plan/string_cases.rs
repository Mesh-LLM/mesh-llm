pub struct StringCase {
    pub name: &'static str,
    pub raw: Vec<u8>,
    pub fragments: Vec<String>,
}

fn replace(raw: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    let start = raw
        .windows(needle.len())
        .position(|bytes| bytes == needle)
        .expect("fixture token");
    [&raw[..start], replacement, &raw[start + needle.len()..]].concat()
}

pub fn cases(base: &[u8]) -> Vec<StringCase> {
    let mut cases = Vec::new();
    cases.extend(note_cases(base));
    cases.extend(encoded_cases(base));
    cases.extend(key_cases(base));
    let evidence = replace(base, b"\"class\": \"causal_generation\", \"architecture\": \"qwen3\", \"profile\": \"full\"",
        br#""class": "embedding", "architecture": "qwen3", "profile": "workload-oracle", "evidence": {"fixture":"\ud800", "comparison":"\udfff"}"#);
    cases.push(StringCase {
        name: "workload-evidence",
        raw: evidence,
        fragments: vec![
            r#""fixture": "\ud800""#.into(),
            r#""comparison": "\udfff""#.into(),
        ],
    });
    cases
}

fn note_cases(base: &[u8]) -> Vec<StringCase> {
    let mut cases = Vec::new();
    for (name, token, rendered) in [
        ("high-surrogate", r#""x\ud800y""#, r#""x\ud800y""#),
        ("low-surrogate", r#""x\udfffy""#, r#""x\udfffy""#),
        (
            "surrogate-boundaries",
            r#""\ud800\ud800\udc00\udfff\udbff\udfff""#,
            r#""\ud800\ud800\udc00\udfff\udbff\udfff""#,
        ),
        (
            "literal-backslash",
            r#""\\ud800\ufffd\u0000\u007f""#,
            r#""\\ud800\ufffd\u0000\u007f""#,
        ),
        (
            "escaped-control-name",
            r#""first", "not\u0065s": "\ud800""#,
            r#""\ud800""#,
        ),
    ] {
        cases.push(StringCase {
            name,
            raw: replace(
                base,
                b"\"notes\": \"synthetic parity only\"",
                format!("\"notes\": {token}").as_bytes(),
            ),
            fragments: vec![format!("\"notes\": {rendered}")],
        });
    }
    cases
}

fn encoded_cases(base: &[u8]) -> Vec<StringCase> {
    let mut cases = Vec::new();
    for (name, codes, expected) in [
        ("raw-high", vec![0xd800], r#""notes": "\ud800""#),
        (
            "raw-adjacent",
            vec![0xd800, 0xdc00],
            r#""notes": "\ud800\udc00""#,
        ),
        ("raw-scalar", vec![0x10000], r#""notes": "\ud800\udc00""#),
    ] {
        for encoding in [0, 1, 2, 3, 4] {
            let template = replace(base, b"synthetic parity only", b"CODEPOINTS");
            let source = String::from_utf8(template).expect("ASCII fixture");
            let (before, after) = source.split_once("CODEPOINTS").expect("replacement");
            let codes = before
                .chars()
                .map(u32::from)
                .chain(codes.iter().copied())
                .chain(after.chars().map(u32::from));
            cases.push(StringCase {
                name,
                raw: encode(codes, encoding),
                fragments: vec![expected.to_owned()],
            });
        }
    }
    cases
}

fn key_cases(base: &[u8]) -> Vec<StringCase> {
    let mut cases = Vec::new();
    let keys = [
        r"\ud800\udc00",
        r"\ufffd",
        r"\ue000",
        r"\udfff",
        r"\ud800a",
        r"\ud800",
        r"\ud7ff",
        "a",
    ];
    let files = keys
        .iter()
        .map(|key| format!("\"{key}\""))
        .collect::<Vec<_>>()
        .join(",");
    let integrity = keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            format!(
                "\"{key}\": {{\"size_bytes\": {}, \"blob_id\": \"{}\"}}",
                index + 1,
                "a".repeat(64)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let artifact = format!(
        "{{\"repo\": \"o/\\ud800\", \"revision\": \"{}\", \"files\": [{files}], \"file_integrity\": {{{integrity}}}, \"selector\": \"\\udfff\"}}",
        "a".repeat(40)
    );
    let text = String::from_utf8(base.to_vec()).expect("ASCII fixture");
    let start = text.find("\"artifact\": ").expect("artifact");
    let end = text[start..].find(",\n").expect("artifact line end") + start;
    cases.push(StringCase {
        name: "codepoint-key-order",
        raw: [
            &text.as_bytes()[..start],
            format!("\"artifact\": {artifact}").as_bytes(),
            &text.as_bytes()[end..],
        ]
        .concat(),
        fragments: keys.iter().map(|key| format!("\"{key}\": {{")).collect(),
    });
    let duplicate = replace(base, b"\"zeta.gguf\": {\"size_bytes\": 1", br#""zeta.gguf": {"size_bytes": 999, "blob_id": "wrong"}, "zeta\u002egguf": {"size_bytes": 1"#);
    cases.push(StringCase {
        name: "decoded-key-last-value",
        raw: duplicate,
        fragments: vec!["\"size_bytes\": 1".into()],
    });
    let scalar_duplicate = String::from_utf8(base.to_vec())
        .expect("ASCII fixture")
        .replace("zeta.gguf", "\\ud800\\udc00");
    let scalar_duplicate = replace(scalar_duplicate.as_bytes(), br#""\ud800\udc00": {"size_bytes": 1"#,
        format!("\"\\ud800\\udc00\": {{\"size_bytes\": 999, \"blob_id\": \"wrong\"}}, \"{}\": {{\"size_bytes\": 1", '\u{10000}').as_bytes());
    cases.push(StringCase {
        name: "pair-and-literal-key-last-value",
        raw: scalar_duplicate,
        fragments: vec![r#""\ud800\udc00": {"#.into(), "\"size_bytes\": 1".into()],
    });
    cases
}

fn encode(codes: impl Iterator<Item = u32>, encoding: u8) -> Vec<u8> {
    let mut raw = match encoding {
        0 => vec![],
        1 => vec![0xff, 0xfe],
        2 => vec![0xfe, 0xff],
        3 => vec![0xff, 0xfe, 0, 0],
        4 => vec![0, 0, 0xfe, 0xff],
        _ => unreachable!("finite encodings"),
    };
    for code in codes {
        match encoding {
            0 => match char::from_u32(code) {
                Some(character) => {
                    raw.extend_from_slice(character.encode_utf8(&mut [0; 4]).as_bytes())
                }
                None => raw.extend_from_slice(&[
                    0xed,
                    0x80 | u8::try_from((code >> 6) & 0x3f).expect("six bits"),
                    0x80 | u8::try_from(code & 0x3f).expect("six bits"),
                ]),
            },
            1 | 2 => {
                let units = match char::from_u32(code) {
                    Some(character) => character.encode_utf16(&mut [0; 2]).to_vec(),
                    None => vec![u16::try_from(code).expect("surrogate")],
                };
                for unit in units {
                    raw.extend_from_slice(&if encoding == 1 {
                        unit.to_le_bytes()
                    } else {
                        unit.to_be_bytes()
                    });
                }
            }
            3 => raw.extend_from_slice(&code.to_le_bytes()),
            4 => raw.extend_from_slice(&code.to_be_bytes()),
            _ => unreachable!("finite encodings"),
        }
    }
    raw
}

//! Canonical digest of an OpenAI request body: float-stringify, RFC 8785 JCS,
//! SHA-256. Pulled out of `openai_exchange.rs` (pure move, no behavior
//! change) because it is a self-contained RFC 8785 port with zero coupling to
//! the envelope types that file otherwise defines.

/// The canonical JSON-DIGEST of a request body: `HEX(SHA-256(JCS(v)))` over
/// the float-stringified body. Byte-for-byte identical to the current
/// `agent_action_capsule.canonical.json_digest` reference and the Python
/// `capsule_sidecar.digest_json` sidecar wrapper it backs
/// (`digest_json(v) = json_digest(_stringify_floats(v))`) — cross-verified
/// against a live run of that reference, not just read off its source.
///
/// Deliberately does **not** apply the profile's absent-field `normalize`
/// step: `agent_action_capsule.canonical` reserves normalization for the
/// vintage format-2 Capsule-ID path only — its current (non-vintage)
/// `json_digest` is plain `JCS(v)`, no normalize — so an OpenAI request body
/// that explicitly carries `null` for an unset optional field (common OpenAI
/// client behavior, e.g. `"stop": null`) must still digest to a DIFFERENT
/// value than the same body with that field omitted; normalizing here would
/// silently collapse the two. (Note for the record: `capsule-emit-mesh`'s own
/// Rust `canonical_body_digest` currently still routes through
/// `capsule_producer::jcs::json_digest`, which normalizes — a drift from the
/// current Python reference that this host digest does not replicate. Flagged
/// separately; out of scope for this crate to fix.)
///
/// It is `HEX(SHA-256(JCS(stringify_floats(body))))`:
///  1. `stringify_floats` — every JSON float becomes its exact decimal string
///     (JCS refuses floats in a digest-bearing value; OpenAI chat bodies are
///     full of them: temperature, top_p, penalties);
///  2. JCS — RFC 8785 canonical serialization (sorted keys, minimal form);
///  3. SHA-256, lowercase hex.
///
/// A self-contained port kept in this crate (the host cannot depend on the
/// plugin's `capsule-producer`), verified against the Python reference on the
/// frozen fixture in the tests below.
///
/// Returns `None` — never a digest of a body the reference would refuse —
/// when `body` contains a JSON integer literal outside the reference's safe
/// range; see [`MAX_SAFE_INTEGER`].
pub fn request_body_digest(body: &serde_json::Value) -> Option<String> {
    if contains_unsafe_integer(body) {
        return None;
    }
    use sha2::{Digest, Sha256};
    let canonical = jcs_bytes(&stringify_floats(body));
    Some(hex::encode(Sha256::digest(&canonical)))
}

/// The reference's safe-integer boundary (`agent_action_capsule.canonical`
/// §5.1): a JSON integer literal outside `+/-(2^53-1)` cannot round-trip
/// through the reference's digest (it raises `UnsafeIntegerError` rather than
/// digest a value it cannot represent losslessly).
const MAX_SAFE_INTEGER: u64 = (1u64 << 53) - 1;

/// Whether `value` contains a JSON integer literal outside the reference's
/// safe range — see [`MAX_SAFE_INTEGER`]. Floats are exempt: a float outside
/// this range already loses precision in the source JSON itself, and
/// [`stringify_floats`] renders it via [`float_repr`], not this check.
fn contains_unsafe_integer(value: &serde_json::Value) -> bool {
    use serde_json::Value;
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.unsigned_abs() > MAX_SAFE_INTEGER
            } else if let Some(u) = n.as_u64() {
                u > MAX_SAFE_INTEGER
            } else {
                false
            }
        }
        Value::Object(map) => map.values().any(contains_unsafe_integer),
        Value::Array(arr) => arr.iter().any(contains_unsafe_integer),
        _ => false,
    }
}

/// Replace every JSON float with its exact decimal-string form via
/// [`float_repr`] (mirrors the Python reference's `_stringify_floats`, which
/// stringifies via `repr(float)`).
///
/// This is a property of the digest context, not a bug this function
/// introduces: once a float and its stringified form are both JSON strings,
/// JCS can no longer tell them apart, so `{"temperature": 0.7}` and
/// `{"temperature": "0.7"}` digest identically. It is inherited unchanged
/// from `agent_action_capsule.canonical`/`capsule_sidecar.digest_json`, the
/// same reference [`request_body_digest`] ports. The current declaration
/// point acknowledging this collision (rather than callers discovering it
/// silently) is the `x-mesh-poc-v1` PoC-only extension block in
/// `capsule-emit-mesh`'s `capsule_sidecar.py`; registering it as a real
/// profile-level property is a separate spec-lane item.
fn stringify_floats(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        // Without the `arbitrary_precision` feature (not enabled anywhere in
        // this workspace), `is_f64()` is already exclusive with
        // `is_i64()`/`is_u64()` — an integer literal never takes this branch.
        Value::Number(n) if n.is_f64() => match n.as_f64() {
            Some(f) => Value::String(float_repr(f)),
            // Unreachable given the guard above; keep the original value
            // rather than a panic site over attacker-supplied JSON.
            None => value.clone(),
        },
        Value::Number(_) => value.clone(),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), stringify_floats(v)))
                .collect(),
        ),
        Value::Array(arr) => Value::Array(arr.iter().map(stringify_floats).collect()),
        other => other.clone(),
    }
}

/// Port of Python's `repr(float)` / `str(float)` / `json.dumps` float
/// formatting: the shortest decimal digit sequence that round-trips to `f`,
/// rendered in fixed-point form when the decimal exponent is in `[-4, 16)`
/// and in exponent form (`d[.ddd]e+NN`/`e-NN`, sign always present, at least
/// two exponent digits) otherwise. Rust's own `Display` for `f64` already
/// computes the shortest round-trip digit sequence but — unlike Python —
/// never switches to exponent form, so this reuses `Display`'s fixed-point
/// string purely as a source of those digits and re-derives the placement.
fn float_repr(f: f64) -> String {
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    let sign = if f.is_sign_negative() { "-" } else { "" };
    let fixed = format!("{}", f.abs());
    let (int_part, frac_part) = match fixed.split_once('.') {
        Some((i, f)) => (i, f),
        None => (fixed.as_str(), ""),
    };
    // The decimal exponent of the most significant digit, and the
    // significant digits themselves (no leading zeros).
    let (digits, exponent): (String, i32) = if int_part != "0" {
        // A double this large (>= 10^16, once `exponent` disqualifies fixed
        // form below) has no fractional precision left — it is exactly an
        // integer — so `int_part` alone carries every real digit. Whatever
        // follows the last nonzero digit is a positional placeholder, not a
        // significant digit, and must be trimmed for the mantissa (`2e17`,
        // not `2.00000000000000000e+17`).
        let exponent = int_part.len() as i32 - 1;
        let trimmed = int_part.trim_end_matches('0');
        let trimmed = if trimmed.is_empty() { "0" } else { trimmed };
        (trimmed.to_string(), exponent)
    } else {
        let leading_zeros = frac_part.chars().take_while(|c| *c == '0').count();
        (frac_part[leading_zeros..].to_string(), -(leading_zeros as i32) - 1)
    };
    if (-4..16).contains(&exponent) {
        // Fixed notation: the source string already places the digits
        // correctly; just guarantee the trailing `.0` Python always shows for
        // a whole number.
        let s = if fixed.contains('.') {
            fixed
        } else {
            format!("{fixed}.0")
        };
        format!("{sign}{s}")
    } else {
        let mantissa = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exponent.abs())
    }
}

/// RFC 8785 JCS serialization (mirror of `agent_action_capsule.canonical.jcs`).
/// Floats are already stringified before this runs, so a bare float here is a
/// programmer error, serialized via serde's default rather than panicking.
fn jcs_bytes(v: &serde_json::Value) -> Vec<u8> {
    let mut out = String::new();
    jcs_value(v, &mut out);
    out.into_bytes()
}

fn jcs_value(v: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::String(s) => jcs_string(s, out),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::Array(arr) => {
            out.push('[');
            for (i, x) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                jcs_value(x, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // RFC 8785 §3.2.3: object members sorted by UTF-16 code-unit sequence.
            let mut items: Vec<(&String, &Value)> = map.iter().collect();
            items.sort_by(|(a, _), (b, _)| {
                let au: Vec<u16> = a.encode_utf16().collect();
                let bu: Vec<u16> = b.encode_utf16().collect();
                au.cmp(&bu).then_with(|| a.cmp(b))
            });
            out.push('{');
            for (i, (k, val)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                jcs_string(k, out);
                out.push(':');
                jcs_value(val, out);
            }
            out.push('}');
        }
    }
}

fn jcs_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        let o = ch as u32;
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ if o == 0x08 => out.push_str("\\b"),
            _ if o == 0x09 => out.push_str("\\t"),
            _ if o == 0x0A => out.push_str("\\n"),
            _ if o == 0x0C => out.push_str("\\f"),
            _ if o == 0x0D => out.push_str("\\r"),
            _ if o < 0x20 => out.push_str(&format!("\\u{o:04x}")),
            _ => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host's `request_body_digest` is byte-for-byte the value a live run
    /// of the Python reference, `capsule_sidecar.digest_json`, produces over
    /// the identical JSON value:
    ///
    ///   python3 -c "
    ///   from capsule_sidecar import digest_json
    ///   print(digest_json({
    ///       'model': 'hermes-2-pro-mistral-7b',
    ///       'messages': [{'role': 'user', 'content': 'hello'}],
    ///       'temperature': 0.7,
    ///       'top_p': 1.0,
    ///       'max_tokens': 512,
    ///   }))"
    ///
    /// `top_p: 1.0` exercises the whole-number-float edge case
    /// (`stringify_floats` must emit "1.0", not "1"). This value has no
    /// null/absent fields, so it does not by itself distinguish a
    /// normalizing digest from a non-normalizing one — see the next test for
    /// that.
    #[test]
    fn request_body_digest_matches_python_reference() {
        let body = serde_json::json!({
            "model": "hermes-2-pro-mistral-7b",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 512
        });
        let expected = "a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5";
        assert_eq!(request_body_digest(&body).as_deref(), Some(expected));
    }

    /// The same body, plus two explicit-`null` optional fields (as real
    /// OpenAI clients routinely send, e.g. `"stop": null`), must digest to a
    /// DIFFERENT value than the null-free body above — proving this digest
    /// does NOT apply the profile's absent-field `normalize` step. Expected
    /// value from the same live Python reference invocation with the two
    /// extra `None` fields added to the dict. Pins the current
    /// `agent_action_capsule.canonical.json_digest` contract (normalize is
    /// vintage-format-2-only) against a future accidental reintroduction of
    /// normalization here.
    #[test]
    fn request_body_digest_does_not_normalize_absent_fields() {
        let body = serde_json::json!({
            "model": "hermes-2-pro-mistral-7b",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 512,
            "stop": null,
            "user": null
        });
        let expected = "ee8aeb450ccf8c8017caae0d3733d3dcd62ec88752053894118d28cea0d176fe";
        assert_eq!(request_body_digest(&body).as_deref(), Some(expected));
    }

    /// `float_repr`'s fixed-vs-exponent switch and digit sequence, each
    /// checked directly against the Python `repr()` values ndizazzo's review
    /// tabulated (`1e-05`, `1e-07`, `1e+16`, `1e+20`), plus the boundary the
    /// table didn't cover (`1e-4` stays fixed) and `-0.0`.
    #[test]
    fn float_repr_matches_python_repr_at_and_around_the_exponent_boundaries() {
        assert_eq!(float_repr(1e-5), "1e-05");
        assert_eq!(float_repr(1e-7), "1e-07");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(1e20), "1e+20");
        assert_eq!(float_repr(2e17), "2e+17");
        // exp == -4 is the last value that stays fixed; exp == -5 switches.
        assert_eq!(float_repr(1e-4), "0.0001");
        assert_eq!(float_repr(-0.0), "-0.0");
        assert_eq!(float_repr(0.0), "0.0");
        assert_eq!(float_repr(0.7), "0.7");
        assert_eq!(float_repr(1.0), "1.0");
    }

    /// Each vector cross-checked against a live run of the Python reference,
    /// `capsule_sidecar.digest_json`, over the identical JSON value:
    ///
    ///   PYTHONPATH=.../agent-action-capsule/python python3 -c "
    ///   from capsule_sidecar import digest_json
    ///   print(digest_json({'v': <value>}))"
    ///
    /// These are exactly the float values the previous digest silently got
    /// wrong (Rust's bare `Display` never emits exponent notation), plus the
    /// `1e-9` determinism trick and the `0.1 + 0.2` imprecision case.
    #[test]
    fn request_body_digest_matches_python_reference_for_float_edge_cases() {
        assert_eq!(
            request_body_digest(&serde_json::json!({"v": 1e-5_f64})).as_deref(),
            Some("8edda8e740022353cd222db1ff9c56e6bef76663a10814bd6aaf579f8d0e2332")
        );
        assert_eq!(
            request_body_digest(&serde_json::json!({"v": 1e-7_f64})).as_deref(),
            Some("bbe43d466a7e53136ff444be1b91529154b547b43774b99d7a6cf41bb4a3ae34")
        );
        assert_eq!(
            request_body_digest(&serde_json::json!({"v": 1e16_f64})).as_deref(),
            Some("318fda488ff6a31c5710e73d0ad02340677be5ed8d553869596ff8b2e27b3b94")
        );
        assert_eq!(
            request_body_digest(&serde_json::json!({"v": 1e20_f64})).as_deref(),
            Some("edf56ab854860723e8a400417d7605b1405964f0dbd4289bfe8c2373238496f5")
        );
        assert_eq!(
            request_body_digest(&serde_json::json!({"v": -0.0_f64})).as_deref(),
            Some("7c9018d8078566c67ebff7a4fa6be32f7b4f0f1dbd4e5d3abcaca596e57d6831")
        );
        assert_eq!(
            request_body_digest(&serde_json::json!({"v": 1e-9_f64})).as_deref(),
            Some("f253528520b4878440038edf53d7a32b00a08bd216c2a0c0b60ffbf5cda133b0")
        );
        assert_eq!(
            request_body_digest(&serde_json::json!({"v": 0.1_f64 + 0.2_f64})).as_deref(),
            Some("5204642c42382100bd6fb098cb429a45a4f42c994b67f2de909274e597756b4b")
        );
    }

    /// A JSON integer literal beyond the reference's safe range
    /// (`+/-(2^53-1)`, `agent_action_capsule.canonical` §5.1) makes the
    /// reference raise `UnsafeIntegerError` rather than digest a value it
    /// cannot represent losslessly — verified live:
    ///
    ///   PYTHONPATH=.../agent-action-capsule/python python3 -c "
    ///   from capsule_sidecar import digest_json
    ///   digest_json({'v': 9007199254740993})"
    ///   # -> UnsafeIntegerError: integer 9007199254740993 is outside the
    ///   #    safe range +/-9007199254740991 (...)
    ///
    /// The host must omit `request_digest` entirely on such a body — never
    /// digest what the reference would refuse.
    #[test]
    fn request_body_digest_omits_when_body_has_an_unsafe_integer() {
        let safe = serde_json::json!({"v": 9_007_199_254_740_991_i64});
        assert!(request_body_digest(&safe).is_some(), "2^53-1 is safe");

        let one_over = serde_json::json!({"v": 9_007_199_254_740_993_i64});
        assert!(
            request_body_digest(&one_over).is_none(),
            "2^53+1 exceeds the reference's safe-integer range"
        );

        let nested = serde_json::json!({"a": [1, {"b": 9_007_199_254_740_993_i64}]});
        assert!(
            request_body_digest(&nested).is_none(),
            "an unsafe integer nested inside an array/object must still be caught"
        );

        let negative_over = serde_json::json!({"v": -9_007_199_254_740_993_i64});
        assert!(
            request_body_digest(&negative_over).is_none(),
            "the safe range is symmetric"
        );
    }

    /// A nested array, a non-object top-level value, a non-BMP key, and a
    /// control character in a string all digest without panicking, and match
    /// the same live Python reference run as the other vectors here.
    #[test]
    fn request_body_digest_matches_python_reference_for_structural_edge_cases() {
        assert_eq!(
            request_body_digest(&serde_json::json!({"v": [1, [2, 3], {"a": 1.5}]})).as_deref(),
            Some("1cb385d061fc163f6663b80217f5c262795c669da48c0b788dcc9b328b888226")
        );
        assert_eq!(
            request_body_digest(&serde_json::json!({"\u{1F600}": 1})).as_deref(),
            Some("763606c9e0046348cc185a6e829e12ae6c0f3565b940f8184901a4d83dda6c33")
        );
        assert_eq!(
            request_body_digest(&serde_json::json!({"v": "a\tb"})).as_deref(),
            Some("595711cef0e6e4d037e1fae2b1ece32702c442c0501d6362791e31cd1a6c866d")
        );
    }
}

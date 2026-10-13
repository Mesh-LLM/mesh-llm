//! Typed projections consumed by the existing optional latency research wrappers.
use crate::command::DynResult;
use serde_json::Value;

pub(super) fn tensor_split(raw: &str) -> DynResult<String> {
    if raw.is_empty() || raw.len() > 5 || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return Err("tensor split requires a bounded positive node count".into());
    }
    let nodes: u32 = raw.parse()?;
    if !(1..=10000).contains(&nodes) {
        return Err("tensor split node count exceeds four-decimal resolution".into());
    }
    let mut unit = 10000 / nodes;
    let remainder = 10000 % nodes;
    if remainder * 2 > nodes || (remainder * 2 == nodes && unit % 2 == 1) {
        unit += 1;
    }
    let tail = 10000_u32
        .checked_sub((nodes - 1) * unit)
        .filter(|v| *v != 0)
        .ok_or("rounded equal tensor split has no positive remainder")?;
    let parts = (0..nodes)
        .map(|index| {
            let value = if index + 1 == nodes { tail } else { unit };
            format!("{}.{:04}", value / 10000, value % 10000)
        })
        .collect::<Vec<_>>();
    Ok(format!("{}\n", parts.join(",")))
}

pub(super) fn latency_summary(bytes: &[u8]) -> DynResult<String> {
    if bytes.len() > super::JSON_LIMIT {
        return Err("latency observation exceeds input bound".into());
    }
    let value: Value = serde_json::from_slice(bytes)?;
    let object = value
        .as_object()
        .ok_or("latency observation must be an object")?;
    let mut fields = Vec::new();
    for name in ["ttft_ms", "total_ms", "tok_s"] {
        let field = match object.get(name) {
            None | Some(Value::Null) => "err".into(),
            Some(Value::Number(number)) => {
                let number = number
                    .as_f64()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or("latency metric must be finite and nonnegative")?;
                if number == 0.0 {
                    "err".into()
                } else if name == "tok_s" {
                    format!("{number:.1}")
                } else {
                    format!("{number:.0}ms")
                }
            }
            _ => return Err("latency metric must be a number or null".into()),
        };
        fields.push(field);
    }
    Ok(format!("{}\n", fields.join("\t")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_parts_preserve_order_count_and_exact_sum() {
        for (nodes, expected) in [
            (1, "1.0000\n"),
            (2, "0.5000,0.5000\n"),
            (3, "0.3333,0.3333,0.3334\n"),
            (6, "0.1667,0.1667,0.1667,0.1667,0.1667,0.1665\n"),
        ] {
            assert_eq!(tensor_split(&nodes.to_string()).unwrap(), expected);
        }
        for nodes in [1, 2, 3, 6, 8, 16, 100, 10000] {
            let row = tensor_split(&nodes.to_string()).unwrap();
            let units = row
                .trim()
                .split(',')
                .map(|v| v.replace('.', "").parse::<u32>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(units.len(), nodes as usize);
            assert_eq!(units.iter().sum::<u32>(), 10000);
            assert!(units.iter().all(|v| *v > 0));
            assert!(units[..units.len() - 1].windows(2).all(|p| p[0] == p[1]));
        }
        for invalid in [
            "0",
            "-1",
            "",
            "1.5",
            "10001",
            "999999999999999999",
            " 2",
            "512",
        ] {
            assert!(tensor_split(invalid).is_err(), "{invalid}");
        }
    }
    #[test]
    fn latency_projection_preserves_partial_missing_and_error_rows() {
        assert_eq!(
            latency_summary(br#"{"ttft_ms":12.6,"total_ms":55.4,"tok_s":7.25,"unconsumed":true}"#)
                .unwrap(),
            "13ms\t55ms\t7.2\n"
        );
        assert_eq!(
            latency_summary(br#"{"ttft_ms":null,"total_ms":0,"tok_s":3}"#).unwrap(),
            "err\terr\t3.0\n"
        );
        assert_eq!(
            latency_summary(br#"{"error":"timeout"}"#).unwrap(),
            "err\terr\terr\n"
        );
        for invalid in [
            "{",
            "[]",
            r#"{"ttft_ms":true}"#,
            r#"{"tok_s":"3"}"#,
            r#"{"total_ms":-1}"#,
            r#"{"tok_s":1e999}"#,
        ] {
            assert!(latency_summary(invalid.as_bytes()).is_err());
        }
        assert!(latency_summary(&vec![b' '; super::super::JSON_LIMIT + 1]).is_err());
    }
}

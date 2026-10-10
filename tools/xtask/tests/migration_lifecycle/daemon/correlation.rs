use super::{RequestId, classify};
use crate::process::{LineEnding, ObservedLine, Stream};

fn record(id: RequestId) -> String {
    format!(
        "{{\"request_id\":\"{}\",\"source\":\"direct_http\",\"route\":\"models\",\"method\":\"GET\",\"request_kind\":\"model_listing\",\"status_code\":200,\"event\":\"request_completed\",\"outcome\":\"completed\"}}",
        id.header()
    )
}

fn observed(bytes: &[u8]) -> ObservedLine<'_> {
    ObservedLine {
        stream: Stream::Stderr,
        bytes,
        ending: LineEnding::Lf,
    }
}

macro_rules! mismatch {
    ($name:ident, $from:literal, $to:literal) => {
        #[test]
        fn $name() {
            let id = RequestId::generate().unwrap();
            let record = record(id).replace($from, $to);
            assert_eq!(classify(observed(record.as_bytes()), id), None);
        }
    };
}
mismatch!(d10_route, "models", "chat");
mismatch!(d10_source, "direct_http", "mesh");
mismatch!(d10_method, "GET", "POST");
mismatch!(d10_kind, "model_listing", "chat_completion");
mismatch!(d10_admitted, "request_completed", "request_admitted");
mismatch!(
    d10_outcome,
    "\"outcome\":\"completed\"",
    "\"outcome\":\"failed\""
);
mismatch!(d10_wrong_status_class, ":200", ":302");
mismatch!(d10_string_status, ":200", ":\"200\"");
mismatch!(
    d10_duplicate_consumed,
    "\"method\":\"GET\"",
    "\"method\":\"POST\",\"method\":\"GET\""
);

#[test]
fn d10_old_uuid() {
    let record = record(RequestId::generate().unwrap());
    assert_eq!(
        classify(observed(record.as_bytes()), RequestId::generate().unwrap()),
        None
    );
}

#[test]
fn d12_lf_boundary_includes_cr() {
    let id = RequestId::generate().unwrap();
    let mut record = record(id);
    record.push_str(&" ".repeat(8191 - record.len()));
    record.push('\r');
    assert_eq!(classify(observed(record.as_bytes()), id), Some(200));
    record.push(' ');
    assert_eq!(classify(observed(record.as_bytes()), id), None);
}

#[test]
fn d12_eof_is_not_lf() {
    let id = RequestId::generate().unwrap();
    let record = record(id);
    let line = ObservedLine {
        ending: LineEnding::Eof,
        ..observed(record.as_bytes())
    };
    assert_eq!(classify(line, id), None);
}

#[test]
fn d12_malformed_json() {
    assert_eq!(
        classify(observed(b"{"), RequestId::generate().unwrap()),
        None
    );
}

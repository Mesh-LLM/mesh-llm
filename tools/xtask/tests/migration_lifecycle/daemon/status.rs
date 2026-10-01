use super::identify;
use crate::automation::daemon_readiness::Rejection;

macro_rules! malformed {
    ($name:ident, $bytes:expr) => {
        #[test]
        fn $name() {
            assert_eq!(
                identify($bytes, (42, 1234)),
                Err(Rejection::MalformedStatus)
            );
        }
    };
}
malformed!(d05_truncated, b"{\"api_port\":");
malformed!(d05_utf8, b"\xff");
malformed!(
    d05_utf8_in_additive_field,
    b"{\"api_port\":1234,\"extra\":\"\xff\",\"local_instances\":[{\"pid\":42,\"is_self\":true}]}"
);
malformed!(d05_scalar, b"42");
malformed!(
    d05_struct_as_array,
    br#"[1234, [{"pid":42,"is_self":true}]]"#
);
malformed!(
    d05_instance_as_array,
    br#"{"api_port":1234,"local_instances":[[42,true]]}"#
);
malformed!(d05_missing, br#"{"api_port":1234}"#);
malformed!(
    d05_string_pid,
    br#"{"api_port":1234,"local_instances":[{"pid":"42","is_self":true}]}"#
);
malformed!(
    d05_duplicate_port,
    br#"{"api_port":1234,"api_port":1234,"local_instances":[]}"#
);
malformed!(
    d05_duplicate_pid,
    br#"{"api_port":1234,"local_instances":[{"pid":42,"pid":42,"is_self":true}]}"#
);
malformed!(
    d05_duplicate_self,
    br#"{"api_port":1234,"local_instances":[{"pid":42,"is_self":true,"is_self":true}]}"#
);
malformed!(d05_trailing, br#"{"api_port":1234,"local_instances":[]}x"#);
malformed!(
    d05_wrong_port_type,
    br#"{"api_port":"1234","local_instances":[]}"#
);

macro_rules! mismatch {
    ($name:ident, $body:expr) => {
        #[test]
        fn $name() {
            assert_eq!(
                identify($body, (42, 1234)),
                Err(Rejection::OwnershipMismatch)
            );
        }
    };
}
mismatch!(
    d05_zero,
    br#"{"api_port":1234,"local_instances":[{"pid":0,"is_self":true}]}"#
);
mismatch!(
    d05_two_self,
    br#"{"api_port":1234,"local_instances":[{"pid":42,"is_self":true},{"pid":42,"is_self":true}]}"#
);
mismatch!(
    d05_wrong_port,
    br#"{"api_port":1235,"local_instances":[{"pid":42,"is_self":true}]}"#
);
mismatch!(
    d05_nonself_only,
    br#"{"api_port":1234,"local_instances":[{"pid":42,"is_self":false}]}"#
);

#[test]
fn d06_additive_metadata_variations() {
    for metadata in ["1234", "3131", "null"] {
        let body = format!(
            r#"{{"api_port":1234,"local_instances":[{{"pid":42,"is_self":true,"api_port":{metadata},"started_at_unix":0,"runtime_dir":""}}],"extra":true}}"#
        );
        assert_eq!(identify(body.as_bytes(), (42, 1234)), Ok(()));
    }
}

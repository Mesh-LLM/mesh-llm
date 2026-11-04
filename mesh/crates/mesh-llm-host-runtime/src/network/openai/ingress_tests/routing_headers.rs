use super::super::tests::test_endpoint_id;
use super::*;
use mesh_llm_events::logging::identifiers::RequestId;

#[test]
fn parse_mesh_target_header_absent_is_a_no_op() {
    assert_eq!(parse_mesh_target_header(&[]), Ok(None));
}

#[test]
fn parse_mesh_target_header_parses_one_valid_value() {
    let id = test_endpoint_id(0x42);
    let value = hex::encode(id.as_bytes());
    assert_eq!(parse_mesh_target_header(&[value]), Ok(Some(id)));
}

#[test]
fn parse_mesh_target_header_rejects_malformed_value() {
    assert!(parse_mesh_target_header(&["not-a-hex-endpoint-id".to_string()]).is_err());
}

#[test]
fn parse_mesh_target_header_rejects_multiple_values_as_ambiguous() {
    let value = hex::encode(test_endpoint_id(0x11).as_bytes());
    assert!(parse_mesh_target_header(&[value.clone(), value]).is_err());
}

#[test]
fn parse_twin_bracket_header_copies_one_valid_value() {
    assert_eq!(parse_twin_bracket_header(&[]), Ok(None));
    assert_eq!(
        parse_twin_bracket_header(&[" run-2026.09:pair_7 ".to_string()]),
        Ok(Some("run-2026.09:pair_7".to_string()))
    );
}

#[test]
fn parse_twin_bracket_header_rejects_malformed_or_repeated_values() {
    let too_long = "x".repeat(129);
    for bad in ["", "has space", "slash/no", too_long.as_str()] {
        assert!(
            parse_twin_bracket_header(&[bad.to_string()]).is_err(),
            "{bad:?} must be rejected"
        );
    }
    assert!(parse_twin_bracket_header(&["a".to_string(), "b".to_string()]).is_err());
}

#[test]
fn parse_twin_bracket_header_trims_only_http_whitespace() {
    for padded in [" pair-7", "pair-7 ", "\tpair-7\t", " \t pair-7 \t "] {
        assert_eq!(
            parse_twin_bracket_header(&[padded.to_string()]),
            Ok(Some("pair-7".to_string())),
            "{padded:?} is SP/HTAB-padded and must be trimmed"
        );
    }
    for ws in ['\u{00A0}', '\u{2003}', '\u{3000}'] {
        for bad in [
            format!("{ws}pair-7"),
            format!("pair-7{ws}"),
            format!("pa{ws}ir-7"),
            format!(" {ws}pair-7 "),
        ] {
            assert!(
                parse_twin_bracket_header(std::slice::from_ref(&bad)).is_err(),
                "{bad:?} carries non-HTTP whitespace and must be rejected"
            );
        }
    }
}

#[test]
fn twin_bracket_header_values_keep_non_http_whitespace_for_the_parser() {
    let raw = "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nx-mesh-twin-bracket: \u{00A0}pair-7\u{3000}\r\nContent-Length: 2\r\n\r\n{}";
    let request = proxy::BufferedHttpRequest {
        raw: raw.as_bytes().to_vec(),
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: 2,
        completion_tokens: None,
        stream: None,
        model_name: None,
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
        exchange_observation_id: None,
        trust: crate::network::request_trust::RequestTrust::UNRESTRICTED,
    };

    let values = request.twin_bracket_header_values().unwrap();
    assert_eq!(values, vec!["\u{00A0}pair-7\u{3000}".to_string()]);
    assert!(
        twin_bracket_id(&request).is_err(),
        "must be a 400, not accepted"
    );
}

#[test]
fn parse_mesh_exclude_header_absent_is_empty() {
    assert_eq!(parse_mesh_exclude_header(&[]), Ok(vec![]));
}

#[test]
fn parse_mesh_exclude_header_splits_comma_separated_list() {
    let id_a = test_endpoint_id(0x11);
    let id_b = test_endpoint_id(0x22);
    let combined = format!(
        "{},{}",
        hex::encode(id_a.as_bytes()),
        hex::encode(id_b.as_bytes())
    );
    assert_eq!(parse_mesh_exclude_header(&[combined]), Ok(vec![id_a, id_b]));
}

#[test]
fn parse_mesh_exclude_header_rejects_malformed_entry() {
    assert!(parse_mesh_exclude_header(&["not-a-hex-endpoint-id".to_string()]).is_err());
}

#[test]
fn parse_mesh_exclude_header_rejects_an_empty_entry() {
    assert!(parse_mesh_exclude_header(&["".to_string()]).is_err());
}

#[test]
fn parse_mesh_exclude_header_rejects_an_empty_entry_between_valid_ones() {
    let id_a = hex::encode(test_endpoint_id(0x11).as_bytes());
    let id_b = hex::encode(test_endpoint_id(0x22).as_bytes());
    assert!(parse_mesh_exclude_header(&[format!("{id_a},,{id_b}")]).is_err());
}

#[test]
fn parse_mesh_exclude_header_rejects_a_trailing_comma() {
    let id_a = hex::encode(test_endpoint_id(0x11).as_bytes());
    assert!(parse_mesh_exclude_header(&[format!("{id_a},")]).is_err());
}

// --- the local-first bypass: `mesh_headers_force_remote` must be consulted
// before a request is ever handed to the local-candidate path ---

#[test]
fn mesh_headers_force_remote_is_false_with_no_headers() {
    let self_id = test_endpoint_id(0x01);
    assert!(!mesh_headers_force_remote(self_id, None, &[]));
}

#[test]
fn mesh_headers_force_remote_is_true_when_target_names_another_peer() {
    let self_id = test_endpoint_id(0x01);
    let peer_id = test_endpoint_id(0x02);
    assert!(
        mesh_headers_force_remote(self_id, Some(peer_id), &[]),
        "a target naming a remote peer must force routing away from local candidates"
    );
}

#[test]
fn mesh_headers_force_remote_is_false_when_target_names_this_node() {
    let self_id = test_endpoint_id(0x01);
    assert!(
        !mesh_headers_force_remote(self_id, Some(self_id), &[]),
        "a target naming this node is allowed to serve locally"
    );
}

#[test]
fn mesh_headers_force_remote_is_true_when_excluding_this_node() {
    let self_id = test_endpoint_id(0x01);
    let other_id = test_endpoint_id(0x02);
    assert!(
        mesh_headers_force_remote(self_id, None, &[other_id, self_id]),
        "excluding this node must remove the local candidate"
    );
}

#[test]
fn mesh_headers_force_remote_is_false_when_excluding_a_different_peer() {
    let self_id = test_endpoint_id(0x01);
    let other_id = test_endpoint_id(0x02);
    assert!(!mesh_headers_force_remote(self_id, None, &[other_id]));
}

use super::*;

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_failed_recipient_preserves_healthy_stream_and_delivery_receipt() {
    for fault in ["queue_overflow", "disconnect_on_response_stream"] {
        let other = "z-failing-observer";
        let roles = [LiveRecipient {
            name: other,
            admission: false,
            deny_model: "",
            fault,
            max_queue_bytes: if fault == "queue_overflow" {
                1
            } else {
                4 * 1024 * 1024
            },
        }];
        let host = LiveHost::start_with_recipients(false, true, false, false, &roles).await;
        let body=br#"{"model":"allowed-model","stream":true,"messages":[{"role":"user","content":"hello"}]}"#;
        let baseline = host.unobserved_baseline("/v1/chat/completions", body).await;
        let response = host.request("/v1/chat/completions", body).await;
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert_eq!(response_entity(&response), response_entity(&baseline));
        let events = host.events();
        assert_terminal(&events, "completed", &response);
        let terminal = events.last().unwrap();
        assert!(terminal.get("observer_response_delivery").is_none());
        assert_eq!(terminal["observer_evidence_complete"], true);
        assert_eq!(
            terminal["response_wire_commitment"]["side_stream_complete"],
            true
        );
        assert_eq!(
            terminal["response_wire_commitment"]["byte_count"],
            response_entity(&response).len()
        );
        assert_receiver_receipt(
            &host,
            terminal,
            "openai_exchange_response",
            &response_entity(&response),
        );
        assert_eq!(host.requests.lock().await.len(), 2);
        if fault == "queue_overflow" {
            let bad = host.recipient_events(other);
            assert_terminal(&bad, "completed", &response);
            assert_eq!(
                bad.last().unwrap()["response_wire_commitment"]["side_stream_complete"],
                false
            );
            assert_eq!(bad.last().unwrap()["observer_evidence_complete"], false);
        } else {
            // The child can log a terminal callback before exiting on the
            // concurrent stream-open RPC. A log write is not an ACK or receipt.
            for event in host.recipient_events(other) {
                if event["phase"] == "exchange_finished" {
                    assert_eq!(event["observer_evidence_complete"], false);
                    assert_eq!(
                        event["response_wire_commitment"]["side_stream_complete"],
                        false
                    );
                }
            }
            let streams =
                std::fs::read_to_string(host.root.path().join(format!("{other}.streams.jsonl")))
                    .unwrap();
            assert!(
                streams
                    .lines()
                    .map(|line| serde_json::from_str::<Value>(line).unwrap())
                    .any(
                        |metadata| metadata["exchange_id"] == terminal["exchange_id"]
                            && metadata["kind"] == "openai_exchange_response"
                    )
            );
            let receipts =
                std::fs::read_to_string(host.root.path().join(format!("{other}.receipts.jsonl")))
                    .unwrap_or_default();
            assert!(
                !receipts
                    .lines()
                    .map(|line| serde_json::from_str::<Value>(line).unwrap())
                    .any(
                        |receipt| receipt["metadata"]["exchange_id"] == terminal["exchange_id"]
                            && receipt["metadata"]["kind"] == "openai_exchange_response"
                            && receipt["complete"] == true
                    ),
                "exited observer unexpectedly completed its response stream"
            );
        }
        host.stop().await;
    }
}

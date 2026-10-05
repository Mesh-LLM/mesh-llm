//! Host lifecycle integration for extracted management HTTP framing.

mod tests {
    use std::{
        pin::Pin,
        sync::Arc,
        task::{Context, Poll},
    };

    use mesh_llm_events::logging::identifiers::RequestId;
    use tokio::{
        io::{AsyncReadExt, AsyncWrite, AsyncWriteExt},
        net::TcpListener,
    };

    use mesh_llm_control_api::http::{
        managed_response_head, respond_bytes_cached, write_managed_response_head,
    };

    #[derive(Default)]
    struct FailAfterResponseHead {
        writes: usize,
        head: Vec<u8>,
    }

    impl AsyncWrite for FailAfterResponseHead {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            self.writes += 1;
            if self.writes == 1 {
                self.head.extend_from_slice(buffer);
                Poll::Ready(Ok(buffer.len()))
            } else {
                Poll::Ready(Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "induced body write failure",
                )))
            }
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    async fn assert_scoped_response_lifecycle(status: u16, expected_state: &str) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
        let (mut server, _) = listener.accept().await.unwrap();
        let request_id = RequestId::new();
        let service = Arc::new(crate::logging::LoggingService::new_disabled(
            Default::default(),
        ));
        let lifecycle = crate::logging::ManagementRequestLifecycle::register(
            Arc::clone(&service),
            request_id,
            "management_post",
        );

        crate::api::management_lifecycle::scope(lifecycle, async {
            write_managed_response_head(
                &mut server,
                format!(
                    "HTTP/1.1 {status} Test\r\nX-Request-Id: upstream\r\nContent-Length: 0\r\n\r\n"
                )
                .into_bytes(),
            )
            .await
            .expect("write managed response header");
        })
        .await;
        server.shutdown().await.unwrap();

        let mut response = Vec::new();
        client.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with(&format!("HTTP/1.1 {status} Test\r\n")));
        assert!(response.contains(&format!("x-request-id: {}\r\n", request_id.as_uuid())));
        assert!(!response.contains("upstream"));
        assert_eq!(response.matches("x-request-id:").count(), 1);
        let entry = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal lifecycle entry");
        assert_eq!(entry.state, expected_state);
    }

    #[tokio::test]
    async fn managed_response_head_replaces_a_conflicting_request_id() {
        let request_id = RequestId::new();
        let service = Arc::new(crate::logging::LoggingService::new_disabled(
            Default::default(),
        ));
        let lifecycle = crate::logging::ManagementRequestLifecycle::register(
            service,
            request_id,
            "management_post",
        );
        let (_, status) = crate::api::management_lifecycle::scope(lifecycle, async {
            let (head, status) = managed_response_head(
                b"HTTP/1.1 201 Created\r\nX-Request-Id: upstream-id\r\nx-request-id: second-upstream-id\r\nContent-Length: 0\r\n\r\n".to_vec(),
            )
            .expect("response head");
            let head = String::from_utf8(head).expect("UTF-8 response head");
            assert!(head.contains(&format!("x-request-id: {}", request_id.as_uuid())));
            assert!(!head.contains("upstream-id"));
            assert!(!head.contains("second-upstream-id"));
            assert_eq!(head.matches("x-request-id:").count(), 1);
            ((), status)
        })
        .await;

        assert_eq!(status, 201);
    }

    #[tokio::test]
    async fn managed_response_writer_records_exact_terminal_status_classes() {
        assert_scoped_response_lifecycle(201, "completed").await;
        assert_scoped_response_lifecycle(404, "rejected").await;
        assert_scoped_response_lifecycle(503, "failed").await;
    }

    #[tokio::test]
    async fn committed_response_head_keeps_status_when_body_write_fails() {
        let request_id = RequestId::new();
        let service = Arc::new(crate::logging::LoggingService::new_disabled(
            Default::default(),
        ));
        let lifecycle = crate::logging::ManagementRequestLifecycle::register(
            Arc::clone(&service),
            request_id,
            "management_post",
        );
        let mut writer = FailAfterResponseHead::default();

        let result = crate::api::management_lifecycle::scope(lifecycle, async {
            respond_bytes_cached(
                &mut writer,
                200,
                "OK",
                "application/octet-stream",
                "no-cache",
                b"body",
            )
            .await
        })
        .await;

        assert!(result.is_err());
        assert!(writer.head.starts_with(b"HTTP/1.1 200 OK\r\n"));
        let entry = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal lifecycle entry");
        assert_eq!(entry.state, "completed");
        let exact_status = service
            .bus_ref()
            .replay_window()
            .records
            .into_iter()
            .filter_map(|record| {
                let envelope =
                    serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
                let payload = envelope.get("payload")?.as_str()?;
                serde_json::from_str::<mesh_llm_events::logging::events::LifecycleEvent>(payload)
                    .ok()
            })
            .find_map(|event| match event {
                mesh_llm_events::logging::events::LifecycleEvent::Completed {
                    status_code, ..
                } => status_code,
                _ => None,
            });
        assert_eq!(exact_status, Some(200));
    }
}

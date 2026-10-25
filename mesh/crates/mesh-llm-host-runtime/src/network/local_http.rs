/// A client for this node's own plain-HTTP endpoints on 127.0.0.1 (API,
/// console, planner). It trusts no roots: it never talks TLS, so building it
/// doesn't load the system's CA certificates. `reqwest::Client::new()` loads
/// them and panics without any, which stopped `mesh-llm serve` at startup on
/// systems without a CA bundle. Those endpoints never redirect, so it doesn't
/// follow redirects either: requests stay on the address they were sent to.
pub(crate) fn local_http_client() -> reqwest::Client {
    mesh_llm_system::local_http::local_http_client_builder()
        .build()
        .expect("a client without TLS roots builds without system certificates")
}

#[cfg(test)]
mod tests {
    use super::local_http_client;

    const LOCAL_HTTP_CLIENT_CHILD: &str = "MESH_LLM_TEST_LOCAL_HTTP_CLIENT_CHILD";

    // SSL_CERT_FILE and SSL_CERT_DIR point the system root loading at an empty
    // file and directory on Linux; they are set on a child test process only.
    #[test]
    fn local_http_client_needs_no_ca_certificates() {
        let temp = tempfile::tempdir().unwrap();
        let empty_file = temp.path().join("empty.pem");
        std::fs::write(&empty_file, "").unwrap();
        let empty_dir = temp.path().join("certs");
        std::fs::create_dir(&empty_dir).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "network::local_http::tests::local_http_client_child",
                "--nocapture",
            ])
            .env(LOCAL_HTTP_CLIENT_CHILD, "1")
            .env("SSL_CERT_FILE", &empty_file)
            .env("SSL_CERT_DIR", &empty_dir)
            .output()
            .expect("spawn local HTTP client child");
        assert!(
            output.status.success(),
            "local HTTP client child failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child test filter must execute its assertion"
        );
    }

    // The responder answers its first request with a redirect to itself. A
    // client that followed it would connect again, which the listener sees as
    // a second pending connection.
    #[tokio::test]
    async fn local_http_client_does_not_follow_redirects() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let responder = listener.try_clone().unwrap();
        let location = url.clone();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = responder.accept().unwrap();
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while std::io::BufRead::read_line(&mut reader, &mut line).unwrap() > 2 {
                line.clear();
            }
            let response = format!(
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            std::io::Write::write_all(&mut stream, response.as_bytes()).unwrap();
        });

        let response = local_http_client()
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .expect("the redirect response itself is the result");
        server.join().unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::TEMPORARY_REDIRECT);
        listener.set_nonblocking(true).unwrap();
        assert_eq!(
            listener.accept().map(|_| ()).unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock,
            "the client must not send a second request"
        );
    }

    #[test]
    fn local_http_client_child() {
        if std::env::var_os(LOCAL_HTTP_CLIENT_CHILD).is_none() {
            return;
        }
        local_http_client();
    }
}

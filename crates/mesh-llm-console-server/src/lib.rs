use mesh_llm_ui::{ConsoleAssetProvider, FileSystemConsoleAssets};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
};

#[derive(Clone, Debug)]
pub struct ConsoleServerOptions {
    pub asset_dir: PathBuf,
    pub port: u16,
    pub listen_all: bool,
}

#[derive(Debug)]
pub struct ConsoleServerHandle {
    url: String,
    shutdown_tx: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl ConsoleServerHandle {
    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
    }
}

pub async fn start_file_console(
    options: ConsoleServerOptions,
) -> anyhow::Result<ConsoleServerHandle> {
    let assets = Arc::new(FileSystemConsoleAssets::new(options.asset_dir));
    if assets.index().is_none() {
        anyhow::bail!("console asset directory must contain index.html");
    }
    start_console(options.port, options.listen_all, assets).await
}

pub async fn start_console(
    port: u16,
    listen_all: bool,
    assets: Arc<dyn ConsoleAssetProvider>,
) -> anyhow::Result<ConsoleServerHandle> {
    let bind_addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = TcpListener::bind(format!("{bind_addr}:{port}")).await?;
    let addr = listener.local_addr()?;
    let url = console_url(addr, listen_all);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task = tokio::spawn(run(listener, assets, shutdown_rx));
    Ok(ConsoleServerHandle {
        url,
        shutdown_tx: Some(shutdown_tx),
        task,
    })
}

async fn run(
    listener: TcpListener,
    assets: Arc<dyn ConsoleAssetProvider>,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    loop {
        tokio::select! {
            result = listener.accept() => {
                let Ok((stream, _)) = result else {
                    continue;
                };
                let assets = assets.clone();
                tokio::spawn(async move {
                    let _ = handle_connection(stream, assets).await;
                });
            }
            _ = &mut shutdown_rx => break,
        }
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    assets: Arc<dyn ConsoleAssetProvider>,
) -> anyhow::Result<()> {
    let Some(request) = read_request(&mut stream).await? else {
        return Ok(());
    };
    let (method, path) = parse_request_line(&request).unwrap_or(("", ""));
    let path_only = path.split('?').next().unwrap_or(path);
    if method != "GET" {
        respond_text(&mut stream, 405, "Method Not Allowed", "method not allowed").await?;
        return Ok(());
    }

    if is_index_route(path_only) {
        respond_asset(&mut stream, assets.index(), 500, "console bundle missing").await?;
    } else if is_static_asset_route(path_only) {
        respond_asset(&mut stream, assets.asset(path_only), 404, "not found").await?;
    } else {
        respond_text(&mut stream, 404, "Not Found", "not found").await?;
    }
    Ok(())
}

fn is_index_route(path: &str) -> bool {
    matches!(
        path,
        "/" | "/dashboard"
            | "/dashboard/"
            | "/reserves"
            | "/reserves/"
            | "/chat"
            | "/chat/"
            | "/configuration"
            | "/configuration/"
            | "/__playground"
            | "/__meshviz-perf"
    ) || path.starts_with("/chat/")
        || path.starts_with("/configuration/")
}

fn is_static_asset_route(path: &str) -> bool {
    path.starts_with("/assets/")
        || matches!(path.rsplit('.').next(), Some("png" | "ico" | "webmanifest"))
        || (path.ends_with(".json") && !path.starts_with("/api/"))
}

async fn read_request(stream: &mut TcpStream) -> anyhow::Result<Option<Vec<u8>>> {
    let mut buffer = vec![0_u8; 16 * 1024];
    let read = tokio::time::timeout(std::time::Duration::from_secs(5), stream.read(&mut buffer))
        .await
        .ok()
        .transpose()?;
    let Some(read) = read else {
        return Ok(None);
    };
    buffer.truncate(read);
    Ok(Some(buffer))
}

fn parse_request_line(request: &[u8]) -> Option<(&str, &str)> {
    let line_end = request.windows(2).position(|window| window == b"\r\n")?;
    let line = std::str::from_utf8(&request[..line_end]).ok()?;
    let mut parts = line.split_whitespace();
    Some((parts.next()?, parts.next()?))
}

async fn respond_asset(
    stream: &mut TcpStream,
    asset: Option<mesh_llm_ui::UiAsset>,
    missing_code: u16,
    missing_message: &str,
) -> anyhow::Result<()> {
    let Some(asset) = asset else {
        return respond_text(
            stream,
            missing_code,
            status_text(missing_code),
            missing_message,
        )
        .await;
    };
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: {}\r\n\r\n",
        asset.content_type,
        asset.contents.len(),
        asset.cache_control
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(asset.contents.as_ref()).await?;
    Ok(())
}

async fn respond_text(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    body: &str,
) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-cache\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    Ok(())
}

fn status_text(code: u16) -> &'static str {
    match code {
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

fn console_url(addr: SocketAddr, listen_all: bool) -> String {
    if listen_all && addr.ip().is_unspecified() {
        format!("http://127.0.0.1:{}", addr.port())
    } else {
        format!("http://{addr}")
    }
}

#[cfg(test)]
mod tests {
    use super::{start_file_console, ConsoleServerOptions};
    use std::{fs, io::Write};

    #[tokio::test]
    async fn serves_index_and_assets_from_directory() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-console-server-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("assets")).expect("create asset root");
        fs::write(root.join("index.html"), "<html>console</html>").expect("write index");
        fs::write(root.join("assets/app.js"), "console.log('ok')").expect("write app");

        let handle = start_file_console(ConsoleServerOptions {
            asset_dir: root.clone(),
            port: 0,
            listen_all: false,
        })
        .await
        .expect("start console");

        let index = blocking_get(handle.url().to_string(), "/".to_string()).await;
        assert!(index.contains("200 OK"));
        assert!(index.contains("<html>console</html>"));

        let asset = blocking_get(handle.url().to_string(), "/assets/app.js".to_string()).await;
        assert!(asset.contains("200 OK"));
        assert!(asset.contains("text/javascript"));

        handle.stop().await;
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn serves_index_for_console_deep_links() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-console-server-deep-link-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("assets")).expect("create asset root");
        fs::write(root.join("index.html"), "<html>console</html>").expect("write index");

        let handle = start_file_console(ConsoleServerOptions {
            asset_dir: root.clone(),
            port: 0,
            listen_all: false,
        })
        .await
        .expect("start console");

        for path in [
            "/configuration",
            "/configuration/defaults",
            "/configuration/local-deployment",
            "/reserves",
            "/chat/thread",
        ] {
            let response = blocking_get(handle.url().to_string(), path.to_string()).await;
            assert!(
                response.contains("200 OK"),
                "expected {path} to serve index, got {response}"
            );
            assert!(response.contains("<html>console</html>"));
        }

        handle.stop().await;
        let _ = fs::remove_dir_all(root);
    }

    async fn blocking_get(base: String, path: String) -> String {
        tokio::task::spawn_blocking(move || {
            let url = base.strip_prefix("http://").expect("test server uses http");
            let mut stream = std::net::TcpStream::connect(url).expect("connect");
            write!(stream, "GET {path} HTTP/1.1\r\nHost: {url}\r\n\r\n").expect("write request");
            let mut response = String::new();
            std::io::Read::read_to_string(&mut stream, &mut response).expect("read response");
            response
        })
        .await
        .expect("blocking get")
    }
}

use super::l3_contract::{Operation, Status};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::time::Duration;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub(super) struct Client {
    pub base: String,
    pub timeout: Duration,
    pub interval: Duration,
    pub cancellation: Option<crate::process::Cancellation>,
}
impl Client {
    async fn json(
        &self,
        method: &str,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        self.poll(async {
            let uri: hyper::Uri=format!("{}{}",self.base.trim_end_matches('/'),path).parse()?;
            let host=uri.host().ok_or("missing management host")?;
            if uri.scheme_str()!=Some("http") || !(host=="localhost" || host.parse::<std::net::IpAddr>().is_ok_and(|ip|ip.is_loopback())) { return Err("disk-L3 management must use loopback HTTP".into()); }
            let socket=tokio::net::TcpStream::connect((host,uri.port_u16().unwrap_or(80))).await?;
            let (mut sender, connection)=hyper::client::conn::http1::handshake(TokioIo::new(socket)).await?;
            let request=hyper::Request::builder().method(method).uri(uri.path_and_query().ok_or("missing path")?.as_str()).header("host",uri.authority().ok_or("missing authority")?.as_str()).header("content-type","application/json").body(Full::new(Bytes::from(body.map(|v|serde_json::to_vec(&v)).transpose()?.unwrap_or_default())))?;
            let response=async {
                let mut response=sender.send_request(request).await?;
                if !response.status().is_success() { return Err(format!("management HTTP {}",response.status()).into()); }
                let mut bytes=Vec::new();
                while let Some(frame)=response.body_mut().frame().await {
                    if let Some(data)=frame?.data_ref() { if bytes.len().saturating_add(data.len())>8*1024*1024 { return Err("management response exceeds 8 MiB".into()); } bytes.extend_from_slice(data); }
                }
                Ok(serde_json::from_slice(&bytes)?)
            };
            tokio::pin!(response);
            tokio::select! { result=&mut response=>result, result=connection=>{ result?; response.await } }
        }).await
    }
    pub async fn status(&self) -> Result<Status> {
        let status: Status =
            serde_json::from_value(self.json("GET", "/api/runtime/kv-cache", None).await?)?;
        if status.version != 1 {
            return Err("unsupported cache status version".into());
        }
        Ok(status)
    }
    pub async fn prune(&self) -> Result<Operation> {
        Ok(serde_json::from_value(
            self.json(
                "POST",
                "/api/runtime/kv-cache/prune",
                Some(serde_json::json!({"target_bytes":0})),
            )
            .await?,
        )?)
    }
    pub async fn clear(&self) -> Result<Operation> {
        Ok(serde_json::from_value(
            self.json(
                "DELETE",
                "/api/runtime/kv-cache",
                Some(serde_json::json!({})),
            )
            .await?,
        )?)
    }
    pub async fn committed(&self, minimum_writes: u64) -> Result<Status> {
        self.poll(async {
            let mut previous = None;
            loop {
                let status = self.status().await?;
                let usage = status.usage.as_ref().ok_or("cache usage unavailable")?;
                let activity = status
                    .activity
                    .as_ref()
                    .ok_or("cache activity unavailable")?;
                let snapshot = (
                    activity.writes,
                    usage.reserved_inflight_bytes,
                    usage.used_bytes,
                );
                if snapshot.0 >= minimum_writes
                    && snapshot.1 == 0
                    && snapshot.2 > 0
                    && previous == Some(snapshot)
                {
                    return Ok(status);
                }
                previous = Some(snapshot);
                tokio::time::sleep(self.interval).await;
            }
        })
        .await
    }
    pub async fn empty(&self) -> Result<Operation> {
        self.poll(async {
            let mut stable = 0;
            loop {
                let operation = self.clear().await?;
                let usage = operation
                    .status
                    .usage
                    .as_ref()
                    .ok_or("cache usage unavailable")?;
                if usage.manifests == 0 && usage.reserved_inflight_bytes == 0 {
                    stable += 1;
                } else {
                    stable = 0;
                }
                if stable == 4 {
                    return Ok(operation);
                }
                tokio::time::sleep(self.interval).await;
            }
        })
        .await
    }
    async fn poll<T>(&self, future: impl std::future::Future<Output = Result<T>>) -> Result<T> {
        if self.interval.is_zero() || self.timeout.is_zero() {
            return Err("management polling durations must be positive".into());
        }
        let cancelled = async {
            if let Some(cancellation) = &self.cancellation {
                while !cancellation.is_cancelled() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            } else {
                std::future::pending::<()>().await;
            }
        };
        tokio::select! { result=tokio::time::timeout(self.timeout,future)=>result?, ()=cancelled=>Err("disk-L3 management interrupted".into()) }
    }
}

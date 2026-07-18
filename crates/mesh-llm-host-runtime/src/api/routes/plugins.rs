use super::super::{
    MeshApi,
    http::{respond_error, respond_json},
};
use crate::plugin::stapler;
use serde_json::{Map, Value};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use url::form_urlencoded;

mod web_ui;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HttpBindingTransferMode {
    Buffered,
    StreamedRequest,
    StreamedResponse,
    StreamedBidirectional,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PluginApiRoute<'a> {
    List,
    Endpoints,
    Providers,
    Provider(&'a str),
    Manifest(&'a str),
    Tools(&'a str),
    ToolCall(&'a str),
    WebUiMetadata(&'a str),
    WebUiEnabled(&'a str),
    WebUiAsset(&'a str),
    WebUiConfig(&'a str),
    StapledHttp,
    Unmatched,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PluginWebUiSubroute<'a> {
    Metadata,
    Enabled,
    Config,
    Asset(&'a str),
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    path_only: &str,
    body: &str,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    match classify_plugin_route(method, path_only) {
        PluginApiRoute::List => handle_list(stream, state).await,
        PluginApiRoute::Endpoints => handle_endpoints(stream, state).await,
        PluginApiRoute::Providers => handle_providers(stream, state).await,
        PluginApiRoute::Provider(route_path) => handle_provider(stream, state, route_path).await,
        PluginApiRoute::Manifest(route_path) => handle_manifest(stream, state, route_path).await,
        PluginApiRoute::Tools(route_path) => handle_tools(stream, state, route_path).await,
        PluginApiRoute::ToolCall(route_path) => handle_call(stream, state, route_path, body).await,
        PluginApiRoute::WebUiMetadata(route_path) => {
            web_ui::handle_metadata(stream, state, route_path).await
        }
        PluginApiRoute::WebUiEnabled(route_path) => {
            web_ui::handle_enabled(stream, state, route_path, body).await
        }
        PluginApiRoute::WebUiAsset(route_path) => {
            web_ui::handle_asset(stream, state, route_path).await
        }
        PluginApiRoute::WebUiConfig(route_path) => {
            web_ui::handle_config(stream, state, method, route_path, body).await
        }
        PluginApiRoute::StapledHttp => {
            handle_stapled_http(stream, state, method, path, path_only, body, raw_request).await
        }
        PluginApiRoute::Unmatched => Ok(()),
    }
}

fn classify_plugin_route<'a>(method: &str, path: &'a str) -> PluginApiRoute<'a> {
    match method {
        "GET" => classify_plugin_get_route(path),
        "POST" if is_plugin_tool_call_route(path) => PluginApiRoute::ToolCall(path),
        "PATCH" if is_plugin_web_ui_enabled_route(path) => PluginApiRoute::WebUiEnabled(path),
        "PATCH" if is_plugin_web_ui_config_route(path) => PluginApiRoute::WebUiConfig(path),
        _ if is_plugin_web_ui_namespace(path) => PluginApiRoute::Unmatched,
        "POST" | "PUT" | "PATCH" | "DELETE" if is_plugin_route(path) => PluginApiRoute::StapledHttp,
        _ => PluginApiRoute::Unmatched,
    }
}

fn classify_plugin_get_route(path: &str) -> PluginApiRoute<'_> {
    match path {
        "/api/plugins" => PluginApiRoute::List,
        "/api/plugins/endpoints" => PluginApiRoute::Endpoints,
        "/api/plugins/providers" => PluginApiRoute::Providers,
        _ if path.starts_with("/api/plugins/providers/") => PluginApiRoute::Provider(path),
        _ if is_plugin_manifest_route(path) => PluginApiRoute::Manifest(path),
        _ if is_plugin_tools_route(path) => PluginApiRoute::Tools(path),
        _ if is_plugin_web_ui_metadata_route(path) => PluginApiRoute::WebUiMetadata(path),
        _ if is_plugin_web_ui_asset_route(path) => PluginApiRoute::WebUiAsset(path),
        _ if is_plugin_web_ui_config_route(path) => PluginApiRoute::WebUiConfig(path),
        _ if is_plugin_web_ui_namespace(path) => PluginApiRoute::Unmatched,
        _ if is_plugin_route(path) => PluginApiRoute::StapledHttp,
        _ => PluginApiRoute::Unmatched,
    }
}

fn is_plugin_route(path: &str) -> bool {
    path.starts_with("/api/plugins/")
}

fn is_plugin_manifest_route(path: &str) -> bool {
    plugin_route_suffix(path) == Some("manifest")
}

fn is_plugin_tools_route(path: &str) -> bool {
    plugin_route_suffix(path) == Some("tools")
}

fn is_plugin_tool_call_route(path: &str) -> bool {
    plugin_route_suffix(path).is_some_and(|suffix| {
        suffix
            .strip_prefix("tools/")
            .is_some_and(|operation| !operation.is_empty() && !operation.contains('/'))
    })
}

fn plugin_route_suffix(path: &str) -> Option<&str> {
    let rest = path.strip_prefix("/api/plugins/")?;
    let (plugin_name, suffix) = rest.split_once('/')?;
    (!plugin_name.is_empty()).then_some(suffix)
}

fn is_plugin_web_ui_namespace(path: &str) -> bool {
    plugin_route_suffix(path)
        .is_some_and(|suffix| suffix == "web-ui" || suffix.starts_with("web-ui/"))
}

pub(super) fn parse_plugin_web_ui_route(path: &str) -> Option<(&str, PluginWebUiSubroute<'_>)> {
    let rest = path.strip_prefix("/api/plugins/")?;
    let (plugin_name, suffix) = rest.split_once('/')?;
    if plugin_name.is_empty() {
        return None;
    }
    match suffix {
        "web-ui" => Some((plugin_name, PluginWebUiSubroute::Metadata)),
        "web-ui/enabled" => Some((plugin_name, PluginWebUiSubroute::Enabled)),
        "web-ui/config" => Some((plugin_name, PluginWebUiSubroute::Config)),
        _ => suffix
            .strip_prefix("web-ui/assets/")
            .filter(|asset_path| !asset_path.is_empty())
            .map(|asset_path| (plugin_name, PluginWebUiSubroute::Asset(asset_path))),
    }
}

fn is_plugin_web_ui_metadata_route(path: &str) -> bool {
    matches!(
        parse_plugin_web_ui_route(path),
        Some((_, PluginWebUiSubroute::Metadata))
    )
}

fn is_plugin_web_ui_enabled_route(path: &str) -> bool {
    matches!(
        parse_plugin_web_ui_route(path),
        Some((_, PluginWebUiSubroute::Enabled))
    )
}

fn is_plugin_web_ui_asset_route(path: &str) -> bool {
    matches!(
        parse_plugin_web_ui_route(path),
        Some((_, PluginWebUiSubroute::Asset(_)))
    )
}

fn is_plugin_web_ui_config_route(path: &str) -> bool {
    matches!(
        parse_plugin_web_ui_route(path),
        Some((_, PluginWebUiSubroute::Config))
    )
}

async fn handle_list(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let plugins = state.plugins().await;
    respond_json(stream, 200, &plugins).await?;
    Ok(())
}

async fn handle_endpoints(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match state.runtime_endpoints().await {
        Ok(endpoints) => respond_json(stream, 200, &endpoints).await?,
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
    Ok(())
}

async fn handle_providers(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match state.plugin_capability_providers().await {
        Ok(providers) => respond_json(stream, 200, &providers).await?,
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
    Ok(())
}

async fn handle_provider(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let capability = &path["/api/plugins/providers/".len()..];
    let capability = urlencoding::decode(capability)
        .map(|value| value.into_owned())
        .unwrap_or_else(|_| capability.to_string());
    match state.plugin_provider_for_capability(&capability).await {
        Ok(Some(provider)) => respond_json(stream, 200, &provider).await?,
        Ok(None) => {
            respond_error(
                stream,
                404,
                &format!("No provider for capability '{}'", capability),
            )
            .await?
        }
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
    Ok(())
}

async fn handle_tools(stream: &mut TcpStream, state: &MeshApi, path: &str) -> anyhow::Result<()> {
    let rest = &path["/api/plugins/".len()..];
    let plugin_name = rest.trim_end_matches("/tools");
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.tools(plugin_name).await {
        Ok(tools) => {
            let json = serde_json::to_string(&tools)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        Err(e) => {
            respond_error(stream, 404, &e.to_string()).await?;
        }
    }
    Ok(())
}

async fn handle_manifest(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let rest = &path["/api/plugins/".len()..];
    let plugin_name = rest.trim_end_matches("/manifest");
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.manifest_json(plugin_name).await {
        Ok(Some(manifest)) => {
            let json = serde_json::to_string(&manifest)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        Ok(None) => {
            respond_error(stream, 404, "Plugin did not publish a manifest").await?;
        }
        Err(e) => {
            respond_error(stream, 500, &e.to_string()).await?;
        }
    }
    Ok(())
}

async fn handle_call(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    let rest = &path["/api/plugins/".len()..];
    if let Some((plugin_name, tool_name)) = rest.split_once("/tools/") {
        let payload = if body.trim().is_empty() { "{}" } else { body };
        let plugin_manager = state.inner.lock().await.plugin_manager.clone();
        match plugin_manager
            .invoke_operation(plugin_name, tool_name, payload)
            .await
        {
            Ok(result) if !result.is_error => {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    result.content_json.len(),
                    result.content_json
                );
                stream.write_all(resp.as_bytes()).await?;
            }
            Ok(result) => {
                respond_error(stream, 502, &result.content_json).await?;
            }
            Err(e) => {
                respond_error(stream, 502, &e.to_string()).await?;
            }
        }
    } else {
        respond_error(stream, 404, "Not found").await?;
    }
    Ok(())
}

async fn handle_stapled_http(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    path_only: &str,
    body: &str,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    let Some((plugin_name, route_path)) = parse_stapled_http_path(path_only) else {
        respond_error(stream, 404, "Not found").await?;
        return Ok(());
    };

    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    let manifest = match plugin_manager.manifest(plugin_name).await {
        Ok(Some(manifest)) => manifest,
        Ok(None) => {
            respond_error(stream, 404, "Plugin did not publish a manifest").await?;
            return Ok(());
        }
        Err(err) => {
            respond_error(stream, 500, &err.to_string()).await?;
            return Ok(());
        }
    };

    let Some(binding) = manifest.http_bindings.iter().find(|binding| {
        stapler::http_binding_route(plugin_name, binding)
            .map(|route| route.method == method && route.route_path == route_path)
            .unwrap_or(false)
    }) else {
        respond_error(stream, 404, "No matching plugin HTTP binding").await?;
        return Ok(());
    };

    if binding_transfer_mode(binding) != HttpBindingTransferMode::Buffered {
        return handle_streamed_http_binding(
            stream,
            &plugin_manager,
            plugin_name,
            binding,
            raw_request,
        )
        .await;
    }

    let Some(operation_name) = binding.operation_name.as_deref() else {
        respond_error(
            stream,
            501,
            "HTTP binding does not declare an operation_name yet",
        )
        .await?;
        return Ok(());
    };

    let args = match build_http_arguments(path, body) {
        Ok(args) => args,
        Err(err) => {
            respond_error(stream, 400, &err).await?;
            return Ok(());
        }
    };

    match plugin_manager
        .invoke_operation(
            plugin_name,
            operation_name,
            &Value::Object(args).to_string(),
        )
        .await
    {
        Ok(result) if !result.is_error => match serde_json::from_str::<Value>(&result.content_json)
        {
            Ok(value) => respond_json(stream, 200, &value).await?,
            Err(_) => {
                respond_error(
                    stream,
                    502,
                    "Plugin returned a non-JSON response for a buffered HTTP binding",
                )
                .await?;
            }
        },
        Ok(result) => {
            respond_error(stream, 502, &result.content_json).await?;
        }
        Err(err) => {
            respond_error(stream, 502, &err.to_string()).await?;
        }
    }

    Ok(())
}

async fn handle_streamed_http_binding(
    client_stream: &mut TcpStream,
    plugin_manager: &crate::plugin::PluginManager,
    plugin_name: &str,
    binding: &crate::plugin::proto::HttpBindingManifest,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    let forwarded_request = rewrite_http_request_path(raw_request, &binding.path)?;
    let stream_id = format!("http-{}-{}", std::process::id(), rand::random::<u64>());
    let request = crate::plugin::proto::OpenStreamRequest {
        stream_id,
        purpose: crate::plugin::proto::StreamPurpose::Generic as i32,
        mode: crate::plugin::proto::StreamMode::Http1 as i32,
        bidirectional: true,
        content_type: Some("application/http".into()),
        correlation_id: None,
        metadata_json: Some(
            serde_json::json!({
                "binding_id": binding.binding_id,
                "method": method_name(binding.method),
                "path": binding.path,
            })
            .to_string(),
        ),
        expected_bytes: Some(forwarded_request.len() as u64),
        idle_timeout_ms: Some(30_000),
    };
    let mut plugin_stream = plugin_manager.connect_stream(plugin_name, request).await?;
    plugin_stream.write_all(&forwarded_request).await?;
    plugin_stream.shutdown().await?;

    let mut buf = [0u8; 16 * 1024];
    loop {
        let read = plugin_stream.read(&mut buf).await?;
        if read == 0 {
            break;
        }
        client_stream.write_all(&buf[..read]).await?;
    }
    Ok(())
}

fn parse_stapled_http_path(path_only: &str) -> Option<(&str, &str)> {
    let rest = path_only.strip_prefix("/api/plugins/")?;
    let (plugin_name, remainder) = rest.split_once("/http")?;
    if plugin_name.is_empty() || remainder.is_empty() {
        return None;
    }
    Some((
        plugin_name,
        &path_only[.."/api/plugins/".len() + plugin_name.len() + "/http".len() + remainder.len()],
    ))
}

fn build_http_arguments(path: &str, body: &str) -> Result<Map<String, Value>, String> {
    let mut args = query_arguments(path);
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Ok(args);
    }
    let body_value: Value =
        serde_json::from_str(trimmed).map_err(|err| format!("Invalid JSON body: {err}"))?;
    let Value::Object(body_map) = body_value else {
        return Err("Buffered plugin HTTP bindings currently require a JSON object body".into());
    };
    args.extend(body_map);
    Ok(args)
}

fn rewrite_http_request_path(raw_request: &[u8], path: &str) -> anyhow::Result<Vec<u8>> {
    let header_end = raw_request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .ok_or_else(|| anyhow::anyhow!("HTTP request is missing a header terminator"))?;
    let mut headers_buf = [httparse::EMPTY_HEADER; 64];
    let mut req = httparse::Request::new(&mut headers_buf);
    req.parse(raw_request)
        .map_err(|err| anyhow::anyhow!("HTTP parse error while rewriting request path: {err}"))?;

    let method = req.method.unwrap_or("GET");
    let version = req.version.unwrap_or(1);
    let original_path = req.path.unwrap_or("/");
    let query = original_path
        .find('?')
        .map(|i| &original_path[i..])
        .unwrap_or("");
    let mut rebuilt = format!(
        "{method} {}{} HTTP/1.{version}\r\n",
        normalized_http_path(path),
        query
    );

    for header in req.headers.iter() {
        let name = header.name;
        if name.eq_ignore_ascii_case("connection") {
            continue;
        }
        let value = std::str::from_utf8(header.value).unwrap_or("");
        rebuilt.push_str(&format!("{name}: {value}\r\n"));
    }
    rebuilt.push_str("Connection: close\r\n\r\n");

    let mut forwarded = rebuilt.into_bytes();
    forwarded.extend_from_slice(&raw_request[header_end..]);
    Ok(forwarded)
}

fn normalized_http_path(path: &str) -> &str {
    if path.is_empty() { "/" } else { path }
}

fn method_name(value: i32) -> &'static str {
    match crate::plugin::proto::HttpMethod::try_from(value)
        .unwrap_or(crate::plugin::proto::HttpMethod::Unspecified)
    {
        crate::plugin::proto::HttpMethod::Get => "GET",
        crate::plugin::proto::HttpMethod::Post => "POST",
        crate::plugin::proto::HttpMethod::Put => "PUT",
        crate::plugin::proto::HttpMethod::Patch => "PATCH",
        crate::plugin::proto::HttpMethod::Delete => "DELETE",
        crate::plugin::proto::HttpMethod::Unspecified => "UNSPECIFIED",
    }
}

fn binding_transfer_mode(
    binding: &crate::plugin::proto::HttpBindingManifest,
) -> HttpBindingTransferMode {
    let request_streamed = matches!(
        crate::plugin::proto::HttpBodyMode::try_from(binding.request_body_mode)
            .unwrap_or(crate::plugin::proto::HttpBodyMode::Unspecified),
        crate::plugin::proto::HttpBodyMode::Streamed
    );
    let response_streamed = matches!(
        crate::plugin::proto::HttpBodyMode::try_from(binding.response_body_mode)
            .unwrap_or(crate::plugin::proto::HttpBodyMode::Unspecified),
        crate::plugin::proto::HttpBodyMode::Streamed
    );
    match (request_streamed, response_streamed) {
        (false, false) => HttpBindingTransferMode::Buffered,
        (true, false) => HttpBindingTransferMode::StreamedRequest,
        (false, true) => HttpBindingTransferMode::StreamedResponse,
        (true, true) => HttpBindingTransferMode::StreamedBidirectional,
    }
}

fn query_arguments(path: &str) -> Map<String, Value> {
    let mut args = Map::new();
    let Some((_, query)) = path.split_once('?') else {
        return args;
    };
    for (key, value) in form_urlencoded::parse(query.as_bytes()) {
        let json_value = if value == "true" {
            Value::Bool(true)
        } else if value == "false" {
            Value::Bool(false)
        } else if let Ok(n) = value.parse::<i64>() {
            Value::Number(n.into())
        } else if let Ok(f) = value.parse::<f64>() {
            // NaN and Infinity are not valid JSON numbers; keep the raw string.
            match serde_json::Number::from_f64(f) {
                Some(n) => Value::Number(n),
                None => Value::String(value.into_owned()),
            }
        } else {
            Value::String(value.into_owned())
        };
        args.insert(key.into_owned(), json_value);
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::MeshApi;
    use crate::mesh::{Node, NodeRole};
    use crate::network::affinity;
    use crate::plugin::{self, PluginSummary, PluginWebUiStateInput, derive_plugin_web_ui_state};
    use mesh_llm_plugin_manager::store::{
        InstalledPluginWebUiBundleMetadata, InstalledPluginWebUiConfigSectionMetadata,
        InstalledPluginWebUiMetadata, InstalledPluginWebUiPageMetadata,
        InstalledPluginWebUiValidation, InstalledPluginWebUiValidationStatus,
    };
    use mesh_llm_plugin_manager::{
        InstalledPluginApplyMode, InstalledPluginConfigSchema, InstalledPluginManifestMetadata,
        InstalledPluginMetadata, InstalledPluginRestartScope, InstalledPluginSettingSchema,
        InstalledPluginValueKind, InstalledPluginValueSchema, InstalledPluginVisibility,
        PluginStore, SUPPORTED_PLUGIN_SCHEMA_VERSION,
    };
    use serial_test::serial;
    use std::path::{Path, PathBuf};
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    #[test]
    fn parses_stapled_http_path() {
        let parsed = parse_stapled_http_path("/api/plugins/demo/http/feed").unwrap();
        assert_eq!(parsed.0, "demo");
        assert_eq!(parsed.1, "/api/plugins/demo/http/feed");
    }

    #[test]
    fn classifies_plugin_web_ui_routes_by_exact_suffix() {
        assert_eq!(
            classify_plugin_route("GET", "/api/plugins/demo/web-ui"),
            PluginApiRoute::WebUiMetadata("/api/plugins/demo/web-ui")
        );
        assert_eq!(
            classify_plugin_route("PATCH", "/api/plugins/demo/web-ui/enabled"),
            PluginApiRoute::WebUiEnabled("/api/plugins/demo/web-ui/enabled")
        );
        assert_eq!(
            classify_plugin_route("GET", "/api/plugins/demo/web-ui/config"),
            PluginApiRoute::WebUiConfig("/api/plugins/demo/web-ui/config")
        );
        assert_eq!(
            classify_plugin_route("GET", "/api/plugins/demo/web-ui/assets/app.js"),
            PluginApiRoute::WebUiAsset("/api/plugins/demo/web-ui/assets/app.js")
        );
        assert_eq!(
            classify_plugin_route("GET", "/api/plugins/demo/http/web-ui/assets/app.js"),
            PluginApiRoute::StapledHttp
        );
        assert_eq!(
            classify_plugin_route("GET", "/api/plugins/demo/web-ui/assets"),
            PluginApiRoute::Unmatched
        );
        assert_eq!(
            classify_plugin_route("DELETE", "/api/plugins/demo/web-ui/assets/app.js"),
            PluginApiRoute::Unmatched
        );
        assert_eq!(
            classify_plugin_route("GET", "/api/plugins/demo/http/manifest"),
            PluginApiRoute::StapledHttp
        );
        assert_eq!(
            classify_plugin_route("POST", "/api/plugins/demo/http/tools/run"),
            PluginApiRoute::StapledHttp
        );
        assert_eq!(
            classify_plugin_route("PATCH", "/api/plugins/demo/http/web-ui/enabled"),
            PluginApiRoute::StapledHttp
        );
    }

    #[test]
    fn query_arguments_decode_values() {
        let args = query_arguments("/api/plugins/demo/http/feed?name=hello%20world&limit=10");
        assert_eq!(args.get("name"), Some(&Value::String("hello world".into())));
        assert_eq!(args.get("limit"), Some(&Value::Number(10.into())));
    }

    #[test]
    fn build_http_arguments_merges_query_and_body() {
        let args = build_http_arguments(
            "/api/plugins/demo/http/feed?from=alice",
            r#"{"limit":10,"from":"bob"}"#,
        )
        .unwrap();
        assert_eq!(args.get("limit"), Some(&Value::Number(10.into())));
        assert_eq!(args.get("from"), Some(&Value::String("bob".into())));
    }

    #[test]
    fn rewrite_http_request_path_updates_request_line_only() {
        let raw = b"POST /api/plugins/demo/http/feed?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\nConnection: keep-alive\r\n\r\n{\"a\":1}";
        let rewritten = rewrite_http_request_path(raw, "/feed").unwrap();
        let text = String::from_utf8(rewritten).unwrap();
        assert!(text.starts_with("POST /feed?x=1 HTTP/1.1\r\n"));
        assert!(text.contains("Host: localhost\r\n"));
        assert!(text.contains("Connection: close\r\n"));
        assert!(text.ends_with("\r\n\r\n{\"a\":1}"));
    }

    #[test]
    fn rewrite_http_request_path_without_query_string() {
        let raw = b"GET /api/plugins/demo/http/items HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let rewritten = rewrite_http_request_path(raw, "/items").unwrap();
        let text = String::from_utf8(rewritten).unwrap();
        assert!(text.starts_with("GET /items HTTP/1.1\r\n"));
    }

    #[test]
    fn binding_transfer_mode_covers_all_streaming_combinations() {
        let mut binding = crate::plugin::proto::HttpBindingManifest {
            binding_id: "demo".into(),
            method: crate::plugin::proto::HttpMethod::Post as i32,
            path: "/demo".into(),
            operation_name: Some("demo".into()),
            request_body_mode: crate::plugin::proto::HttpBodyMode::Buffered as i32,
            response_body_mode: crate::plugin::proto::HttpBodyMode::Buffered as i32,
            request_schema_json: None,
            response_schema_json: None,
        };
        assert_eq!(
            binding_transfer_mode(&binding),
            HttpBindingTransferMode::Buffered
        );

        binding.request_body_mode = crate::plugin::proto::HttpBodyMode::Streamed as i32;
        assert_eq!(
            binding_transfer_mode(&binding),
            HttpBindingTransferMode::StreamedRequest
        );

        binding.request_body_mode = crate::plugin::proto::HttpBodyMode::Buffered as i32;
        binding.response_body_mode = crate::plugin::proto::HttpBodyMode::Streamed as i32;
        assert_eq!(
            binding_transfer_mode(&binding),
            HttpBindingTransferMode::StreamedResponse
        );

        binding.request_body_mode = crate::plugin::proto::HttpBodyMode::Streamed as i32;
        assert_eq!(
            binding_transfer_mode(&binding),
            HttpBindingTransferMode::StreamedBidirectional
        );
    }

    async fn connected_tcp_streams() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).await.unwrap();
        let (server, _) = listener.accept().await.unwrap();
        (client, server)
    }

    async fn build_test_api_with_plugin_manager(plugin_manager: plugin::PluginManager) -> MeshApi {
        let config_dir = tempfile::tempdir().unwrap();
        build_test_api_with_plugin_manager_and_config_path(
            plugin_manager,
            &config_dir.keep().join("config.toml"),
        )
        .await
    }

    async fn build_test_api_with_plugin_manager_and_config_path(
        plugin_manager: plugin::PluginManager,
        config_path: &Path,
    ) -> MeshApi {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        node.replace_config_state_for_test(config_path)
            .await
            .unwrap();
        let runtime_data_collector = node.runtime_data_collector();
        let runtime_data_producer =
            runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
                scope: "runtime",
                plugin_data_key: None,
                plugin_endpoint_key: None,
            });
        MeshApi::new(crate::api::MeshApiConfig {
            node,
            model_name: "test-model".into(),
            api_port: 3131,
            model_size_bytes: 0,
            owner_key_path: None,
            plugin_manager,
            affinity_router: affinity::AffinityRouter::default(),
            runtime_data_collector,
            runtime_data_producer,
        })
    }

    struct PluginStoreEnvGuard {
        previous: Option<std::ffi::OsString>,
    }

    impl PluginStoreEnvGuard {
        fn install(path: &Path) -> Self {
            let previous = std::env::var_os("MESH_LLM_PLUGIN_DIR");
            unsafe { std::env::set_var("MESH_LLM_PLUGIN_DIR", path) };
            Self { previous }
        }
    }

    impl Drop for PluginStoreEnvGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(previous) => unsafe { std::env::set_var("MESH_LLM_PLUGIN_DIR", previous) },
                None => unsafe { std::env::remove_var("MESH_LLM_PLUGIN_DIR") },
            }
        }
    }

    fn installed_metadata(
        name: &str,
        install_path: PathBuf,
        validation_status: InstalledPluginWebUiValidationStatus,
    ) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: name.to_string(),
            source_repository: format!("https://github.com/mesh-llm/{name}"),
            installed_version: "v1.0.0".to_string(),
            target_triple: "test-target".to_string(),
            downloaded_asset_name: format!("{name}.tar.gz"),
            install_path,
            enabled: true,
            manifest: Some(InstalledPluginManifestMetadata {
                config_schema: Some(InstalledPluginConfigSchema {
                    plugin_name: name.to_string(),
                    schema_version: SUPPORTED_PLUGIN_SCHEMA_VERSION,
                    allow_unvalidated_config: false,
                    settings: vec![InstalledPluginSettingSchema {
                        key: "retention_days".to_string(),
                        value_schema: InstalledPluginValueSchema {
                            kind: InstalledPluginValueKind::Integer,
                            enum_values: Vec::new(),
                            items: None,
                            object_properties: Vec::new(),
                            allow_additional_properties: false,
                        },
                        required: false,
                        default_json: Some("30".to_string()),
                        constraints: Vec::new(),
                        apply_mode: InstalledPluginApplyMode::DynamicApply,
                        restart_scope: InstalledPluginRestartScope::None,
                        visibility: InstalledPluginVisibility::User,
                        description: Some("Retention days".to_string()),
                        presentation: None,
                        control_behavior: None,
                    }],
                }),
                web_ui: Some(InstalledPluginWebUiMetadata {
                    pages: vec![InstalledPluginWebUiPageMetadata {
                        id: "home".to_string(),
                        label: "Home".to_string(),
                        icon: None,
                        route: "home".to_string(),
                        bundle_id: "main".to_string(),
                        entry_script: "assets/app.js".to_string(),
                    }],
                    config_sections: vec![InstalledPluginWebUiConfigSectionMetadata {
                        id: "settings".to_string(),
                        title: "Settings".to_string(),
                        entry_script: "assets/settings.js".to_string(),
                        parent_tab: Some("integrations".to_string()),
                        bundle_id: "main".to_string(),
                    }],
                    bundles: vec![InstalledPluginWebUiBundleMetadata {
                        id: "main".to_string(),
                        root_path: "web".to_string(),
                    }],
                    asset_root: Some(PathBuf::from("web")),
                    validation: InstalledPluginWebUiValidation {
                        status: validation_status,
                        reason: Some("bundle failed validation".to_string()),
                    },
                }),
            }),
            last_protocol_version: Some(1),
            last_status: Some("running".to_string()),
            last_error: None,
        }
    }

    fn summary_from_metadata(
        metadata: &InstalledPluginMetadata,
        web_ui_enabled: Option<bool>,
        status: &str,
        error: Option<String>,
    ) -> PluginSummary {
        PluginSummary {
            name: metadata.name.clone(),
            kind: "installed".to_string(),
            enabled: status == "running",
            status: status.to_string(),
            pid: None,
            version: Some(metadata.installed_version.clone()),
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            web_ui: derive_plugin_web_ui_state(PluginWebUiStateInput {
                plugin_name: &metadata.name,
                live_manifest: None,
                installed_metadata: Some(metadata),
                web_ui_enabled,
                runtime_available: status == "running",
                runtime_unavailable_reason: error.as_deref(),
            }),
            startup: None,
            error,
        }
    }

    fn nondeclaring_summary(name: &str) -> PluginSummary {
        PluginSummary {
            name: name.to_string(),
            kind: "external".to_string(),
            enabled: true,
            status: "running".to_string(),
            pid: None,
            version: None,
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            web_ui: crate::plugin::PluginWebUiState::default(),
            startup: None,
            error: None,
        }
    }

    async fn call_plugins_route(state: &MeshApi, method: &str, path: &str, body: &str) -> String {
        let (mut observed_client, mut response_stream) = connected_tcp_streams().await;
        let raw_request = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        handle(
            &mut response_stream,
            state,
            method,
            path,
            path,
            body,
            raw_request.as_bytes(),
        )
        .await
        .unwrap();
        response_stream.shutdown().await.unwrap();
        let mut response_bytes = Vec::new();
        observed_client
            .read_to_end(&mut response_bytes)
            .await
            .unwrap();
        String::from_utf8(response_bytes).unwrap()
    }

    fn response_body(response: &str) -> &str {
        response.split("\r\n\r\n").nth(1).unwrap_or("")
    }

    fn json_body(response: &str) -> Value {
        serde_json::from_str(response_body(response)).unwrap()
    }

    fn prepare_installed_plugin(
        store: &PluginStore,
        temp: &Path,
        name: &str,
        validation_status: InstalledPluginWebUiValidationStatus,
    ) -> InstalledPluginMetadata {
        let install_path = temp.join("installed").join(name);
        std::fs::create_dir_all(install_path.join("web/assets")).unwrap();
        std::fs::write(
            install_path.join("web/assets/app.js"),
            "export const ok = true;",
        )
        .unwrap();
        std::fs::write(install_path.join("web/assets/settings.js"), "export {};").unwrap();
        let metadata = installed_metadata(name, install_path, validation_status);
        store.save(&metadata).unwrap();
        metadata
    }

    #[tokio::test]
    #[serial]
    async fn plugin_web_ui_api_serves_metadata_toggle_assets_and_updated_summary() {
        let temp = tempfile::tempdir().unwrap();
        let _env = PluginStoreEnvGuard::install(temp.path());
        let store = PluginStore::new(temp.path());
        let metadata = prepare_installed_plugin(
            &store,
            temp.path(),
            "demo",
            InstalledPluginWebUiValidationStatus::Valid,
        );
        let plugin_manager =
            plugin::PluginManager::for_test_summaries(vec![summary_from_metadata(
                &metadata, None, "running", None,
            )]);
        let config_path = temp.path().join("config.toml");
        let state =
            build_test_api_with_plugin_manager_and_config_path(plugin_manager, &config_path).await;

        let metadata_response =
            call_plugins_route(&state, "GET", "/api/plugins/demo/web-ui", "").await;
        let metadata_body = json_body(&metadata_response);
        assert!(metadata_response.starts_with("HTTP/1.1 200 OK"));
        assert_eq!(metadata_body["state"], "ready");
        assert_eq!(
            metadata_body["asset_base_url"],
            "/api/plugins/demo/web-ui/assets/"
        );

        let asset_response = call_plugins_route(
            &state,
            "GET",
            "/api/plugins/demo/web-ui/assets/assets/app.js",
            "",
        )
        .await;
        assert!(asset_response.starts_with("HTTP/1.1 200 OK"));
        assert!(asset_response.contains("Content-Type: text/javascript; charset=utf-8"));
        assert!(asset_response.contains("Cache-Control: no-cache"));
        assert_eq!(response_body(&asset_response), "export const ok = true;");

        let config_response = call_plugins_route(
            &state,
            "PATCH",
            "/api/plugins/demo/web-ui/config",
            r#"{"plugin":"demo","settings":{"retention_days":14}}"#,
        )
        .await;
        let config_body = json_body(&config_response);
        assert!(config_response.starts_with("HTTP/1.1 200 OK"));
        assert_eq!(config_body["plugin"], "demo");
        assert_eq!(config_body["settings"]["retention_days"], 14);
        let persisted_config: crate::plugin::MeshConfig =
            toml::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(persisted_config.plugins[0].enabled, None);
        assert_eq!(persisted_config.plugins[0].web_ui_enabled, None);
        assert_eq!(
            persisted_config.plugins[0].settings.get("retention_days"),
            Some(&toml::Value::Integer(14))
        );

        let invalid_config_response = call_plugins_route(
            &state,
            "PATCH",
            "/api/plugins/demo/web-ui/config",
            r#"{"plugin":"demo","settings":{"retention_days":"invalid"}}"#,
        )
        .await;
        assert!(invalid_config_response.starts_with("HTTP/1.1 422 Unprocessable Content"));

        let disable_response = call_plugins_route(
            &state,
            "PATCH",
            "/api/plugins/demo/web-ui/enabled",
            r#"{"enabled":false}"#,
        )
        .await;
        let disable_body = json_body(&disable_response);
        assert!(disable_response.starts_with("HTTP/1.1 200 OK"));
        assert_eq!(disable_body["state"], "disabled");
        let persisted = std::fs::read_to_string(&config_path).unwrap();
        assert!(persisted.contains("name = \"demo\""));
        assert!(persisted.contains("web_ui_enabled = false"));
        let persisted_config: crate::plugin::MeshConfig = toml::from_str(&persisted).unwrap();
        assert_eq!(persisted_config.plugins[0].enabled, None);
        assert_eq!(persisted_config.plugins[0].web_ui_enabled, Some(false));
        assert_eq!(
            persisted_config.plugins[0].settings.get("retention_days"),
            Some(&toml::Value::Integer(14))
        );

        let summary_response = call_plugins_route(&state, "GET", "/api/plugins", "").await;
        let summary_body = json_body(&summary_response);
        assert_eq!(summary_body[0]["status"], "running");
        assert_eq!(summary_body[0]["enabled"], true);
        assert_eq!(summary_body[0]["web_ui"]["state"], "disabled");

        let enable_response = call_plugins_route(
            &state,
            "PATCH",
            "/api/plugins/demo/web-ui/enabled",
            r#"{"enabled":true}"#,
        )
        .await;
        let enable_body = json_body(&enable_response);
        assert!(enable_response.starts_with("HTTP/1.1 200 OK"));
        assert_eq!(enable_body["state"], "ready");
    }

    struct WebUiFailureFixture {
        _temp: tempfile::TempDir,
        _env: PluginStoreEnvGuard,
        state: MeshApi,
        config_path: PathBuf,
    }

    async fn web_ui_failure_fixture() -> WebUiFailureFixture {
        let temp = tempfile::tempdir().unwrap();
        let env = PluginStoreEnvGuard::install(temp.path());
        let store = PluginStore::new(temp.path());
        let ready = prepare_installed_plugin(
            &store,
            temp.path(),
            "ready",
            InstalledPluginWebUiValidationStatus::Valid,
        );
        let disabled = prepare_installed_plugin(
            &store,
            temp.path(),
            "disabled",
            InstalledPluginWebUiValidationStatus::Valid,
        );
        let invalid = prepare_installed_plugin(
            &store,
            temp.path(),
            "invalid",
            InstalledPluginWebUiValidationStatus::Invalid,
        );
        let stopped = prepare_installed_plugin(
            &store,
            temp.path(),
            "stopped",
            InstalledPluginWebUiValidationStatus::Valid,
        );
        let plugin_manager = plugin::PluginManager::for_test_summaries(vec![
            summary_from_metadata(&ready, None, "running", None),
            summary_from_metadata(&disabled, Some(false), "running", None),
            summary_from_metadata(&invalid, None, "running", None),
            summary_from_metadata(
                &stopped,
                None,
                "disabled",
                Some("plugin process is disabled".to_string()),
            ),
            nondeclaring_summary("plain"),
        ]);
        let config_path = temp.path().join("config.toml");
        let state =
            build_test_api_with_plugin_manager_and_config_path(plugin_manager, &config_path).await;
        WebUiFailureFixture {
            _temp: temp,
            _env: env,
            state,
            config_path,
        }
    }

    #[tokio::test]
    #[serial]
    async fn plugin_web_ui_api_reports_metadata_failure_states() {
        let fixture = web_ui_failure_fixture().await;
        let state = &fixture.state;
        let ordered = call_plugins_route(state, "GET", "/api/plugins/ready/web-ui", "").await;
        assert!(ordered.starts_with("HTTP/1.1 200 OK"));
        assert_eq!(json_body(&ordered)["state"], "ready");

        let disabled_metadata =
            call_plugins_route(state, "GET", "/api/plugins/disabled/web-ui", "").await;
        assert!(disabled_metadata.starts_with("HTTP/1.1 200 OK"));
        let disabled_body = json_body(&disabled_metadata);
        assert_eq!(disabled_body["state"], "disabled");
        assert_eq!(disabled_body["available"], false);
        assert_eq!(
            disabled_body["unavailable_reason"],
            "web UI disabled by configuration"
        );

        let invalid_metadata =
            call_plugins_route(state, "GET", "/api/plugins/invalid/web-ui", "").await;
        assert!(invalid_metadata.starts_with("HTTP/1.1 200 OK"));
        let invalid_body = json_body(&invalid_metadata);
        assert_eq!(invalid_body["state"], "invalid");
        assert_eq!(invalid_body["available"], false);
        assert_eq!(
            invalid_body["unavailable_reason"],
            "bundle failed validation"
        );

        let stopped_metadata =
            call_plugins_route(state, "GET", "/api/plugins/stopped/web-ui", "").await;
        assert!(stopped_metadata.starts_with("HTTP/1.1 200 OK"));
        let stopped_body = json_body(&stopped_metadata);
        assert_eq!(stopped_body["state"], "plugin_not_running");
        assert_eq!(stopped_body["available"], false);
        assert_eq!(
            stopped_body["unavailable_reason"],
            "plugin process is disabled"
        );

        let plain_metadata =
            call_plugins_route(state, "GET", "/api/plugins/plain/web-ui", "").await;
        assert!(plain_metadata.starts_with("HTTP/1.1 200 OK"));
        let plain_body = json_body(&plain_metadata);
        assert_eq!(plain_body["state"], "none");
        assert_eq!(plain_body["declared"], false);
        assert_eq!(plain_body["available"], false);
    }

    #[tokio::test]
    #[serial]
    async fn plugin_web_ui_api_rejects_unavailable_and_unsafe_assets() {
        let fixture = web_ui_failure_fixture().await;
        let state = &fixture.state;
        let disabled_asset = call_plugins_route(
            state,
            "GET",
            "/api/plugins/disabled/web-ui/assets/assets/app.js",
            "",
        )
        .await;
        assert!(disabled_asset.starts_with("HTTP/1.1 404 Not Found"));

        let invalid_asset = call_plugins_route(
            state,
            "GET",
            "/api/plugins/invalid/web-ui/assets/assets/app.js",
            "",
        )
        .await;
        assert!(invalid_asset.starts_with("HTTP/1.1 409 Conflict"));
        assert_eq!(
            json_body(&invalid_asset)["error"],
            "bundle failed validation"
        );

        let stopped_asset = call_plugins_route(
            state,
            "GET",
            "/api/plugins/stopped/web-ui/assets/assets/app.js",
            "",
        )
        .await;
        assert!(stopped_asset.starts_with("HTTP/1.1 409 Conflict"));
        assert_eq!(
            json_body(&stopped_asset)["error"],
            "plugin process is disabled"
        );

        let traversal = call_plugins_route(
            state,
            "GET",
            "/api/plugins/ready/web-ui/assets/%2e%2e/secret.txt",
            "",
        )
        .await;
        assert!(traversal.starts_with("HTTP/1.1 404 Not Found"));

        let missing_asset = call_plugins_route(
            state,
            "GET",
            "/api/plugins/ready/web-ui/assets/assets/missing.js",
            "",
        )
        .await;
        assert!(missing_asset.starts_with("HTTP/1.1 404 Not Found"));

        let unknown_asset = call_plugins_route(
            state,
            "GET",
            "/api/plugins/unknown/web-ui/assets/assets/app.js",
            "",
        )
        .await;
        assert!(unknown_asset.starts_with("HTTP/1.1 404 Not Found"));
    }

    #[tokio::test]
    #[serial]
    async fn plugin_web_ui_api_rejects_invalid_toggles_and_config_without_stapled_fallback() {
        let fixture = web_ui_failure_fixture().await;
        let state = &fixture.state;
        let nondeclaring_toggle = call_plugins_route(
            state,
            "PATCH",
            "/api/plugins/plain/web-ui/enabled",
            r#"{"enabled":false}"#,
        )
        .await;
        assert!(nondeclaring_toggle.starts_with("HTTP/1.1 400 Bad Request"));
        assert_eq!(
            json_body(&nondeclaring_toggle)["error"],
            "Plugin does not declare a web UI"
        );
        assert!(!fixture.config_path.exists());

        let stapled_asset_path = call_plugins_route(
            state,
            "GET",
            "/api/plugins/ready/http/web-ui/assets/assets/app.js",
            "",
        )
        .await;
        assert!(!stapled_asset_path.starts_with("HTTP/1.1 200 OK"));
        assert!(!response_body(&stapled_asset_path).contains("export const ok = true;"));

        let plugin_mismatch = call_plugins_route(
            state,
            "PATCH",
            "/api/plugins/ready/web-ui/config",
            r#"{"plugin":"other","settings":{"retention_days":14}}"#,
        )
        .await;
        assert!(plugin_mismatch.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(
            json_body(&plugin_mismatch)["error"]
                .as_str()
                .unwrap()
                .contains("does not match")
        );

        let host_owned_key = call_plugins_route(
            state,
            "PATCH",
            "/api/plugins/ready/web-ui/config",
            r#"{"settings":{"enabled":false}}"#,
        )
        .await;
        assert!(host_owned_key.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(
            json_body(&host_owned_key)["error"]
                .as_str()
                .unwrap()
                .contains("host-owned")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streamed_http_bindings_proxy_all_transfer_modes_over_side_streams() {
        struct NoopBridge;
        impl plugin::PluginRpcBridge for NoopBridge {
            fn handle_request(
                &self,
                _plugin_name: String,
                _method: String,
                _params_json: String,
            ) -> plugin::BridgeFuture<Result<plugin::RpcResult, crate::plugin::proto::ErrorResponse>>
            {
                Box::pin(async {
                    Err(crate::plugin::proto::ErrorResponse {
                        code: rmcp::model::ErrorCode::INTERNAL_ERROR.0,
                        message: "unexpected request".into(),
                        data_json: String::new(),
                    })
                })
            }

            fn handle_notification(
                &self,
                _plugin_name: String,
                _method: String,
                _params_json: String,
            ) -> plugin::BridgeFuture<()> {
                Box::pin(async {})
            }
        }

        let plugin_manager =
            plugin::PluginManager::for_test_bridge(&["demo"], std::sync::Arc::new(NoopBridge));
        let transfer_modes = [
            (
                crate::plugin::proto::HttpBodyMode::Buffered,
                crate::plugin::proto::HttpBodyMode::Streamed,
            ),
            (
                crate::plugin::proto::HttpBodyMode::Streamed,
                crate::plugin::proto::HttpBodyMode::Buffered,
            ),
            (
                crate::plugin::proto::HttpBodyMode::Streamed,
                crate::plugin::proto::HttpBodyMode::Streamed,
            ),
        ];
        for (request_mode, response_mode) in transfer_modes {
            plugin_manager
                .set_test_manifests(std::collections::BTreeMap::from([(
                    "demo".into(),
                    crate::plugin::proto::PluginManifest {
                        http_bindings: vec![crate::plugin::proto::HttpBindingManifest {
                            binding_id: "stream".into(),
                            method: crate::plugin::proto::HttpMethod::Post as i32,
                            path: "/stream".into(),
                            operation_name: Some("stream".into()),
                            request_body_mode: request_mode as i32,
                            response_body_mode: response_mode as i32,
                            request_schema_json: None,
                            response_schema_json: None,
                        }],
                        ..Default::default()
                    },
                )]))
                .await;
            plugin_manager
                .set_test_stream_handler("demo", move |request| {
                    Box::pin(async move {
                        let mut request = request;
                        request.stream_id = "s".into();
                        let listener =
                            mesh_llm_plugin::bind_side_stream("demo", &request.stream_id).await?;
                        let response = listener.open_stream_response(&request);
                        let endpoint = response.endpoint.clone().unwrap();
                        let transport_kind = response.transport_kind;
                        tokio::spawn(async move {
                            let mut plugin_stream = listener.accept().await.unwrap();
                            let mut request_bytes =
                                vec![0u8; request.expected_bytes.unwrap_or_default() as usize];
                            plugin_stream
                                .read_exact_bytes(&mut request_bytes)
                                .await
                                .unwrap();
                            let request_text = String::from_utf8_lossy(&request_bytes);
                            assert!(request_text.starts_with("POST /stream HTTP/1.1\r\n"));
                            assert!(request_text.contains("Connection: close\r\n"));
                            plugin_stream
                                .write_all_bytes(
                                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 12\r\n\r\n{\"ok\":true}\n",
                                )
                                .await
                                .unwrap();
                        });
                        crate::plugin::connect_test_side_stream(&endpoint, transport_kind).await
                    })
                })
                .await;
            let state = build_test_api_with_plugin_manager(plugin_manager.clone()).await;
            let (mut observed_client, mut response_stream) = connected_tcp_streams().await;
            let raw_request = b"POST /api/plugins/demo/http/stream HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 7\r\nConnection: keep-alive\r\n\r\n{\"a\":1}";
            handle_stapled_http(
                &mut response_stream,
                &state,
                "POST",
                "/api/plugins/demo/http/stream",
                "/api/plugins/demo/http/stream",
                "{\"a\":1}",
                raw_request,
            )
            .await
            .unwrap();
            response_stream.shutdown().await.unwrap();
            let mut response_bytes = Vec::new();
            observed_client
                .read_to_end(&mut response_bytes)
                .await
                .unwrap();
            let response_text = String::from_utf8_lossy(&response_bytes);
            assert!(response_text.starts_with("HTTP/1.1 200 OK\r\n"));
            assert!(response_text.contains("{\"ok\":true}"));
        }
    }
}

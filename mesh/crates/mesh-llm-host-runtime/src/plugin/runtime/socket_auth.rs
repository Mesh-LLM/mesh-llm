//! Authenticate new lifecycle capabilities to the OS peer of the launched child.

use anyhow::{Context, Result, bail};

use super::super::transport::LocalStream;

pub(super) fn authenticate(stream: &LocalStream, child_pid: Option<u32>) -> Result<()> {
    let expected = child_pid.context("launched plugin process ID unavailable")?;
    let actual = peer_process_id(stream)?;
    if actual != expected {
        bail!("plugin control peer does not match the launched process");
    }
    Ok(())
}

pub(super) fn is_authenticated(
    stream: &LocalStream,
    child_pid: Option<u32>,
    host_owned_runner: bool,
) -> bool {
    // Builtins use a host-created duplex stream and cannot be raced locally.
    if matches!(stream, LocalStream::Memory(_)) {
        return host_owned_runner;
    }
    #[cfg(test)]
    if matches!(stream, LocalStream::Tcp(_)) {
        // TCP exists solely for explicitly constructed host-runtime test fixtures.
        return true;
    }
    authenticate(stream, child_pid).is_ok()
}

pub(super) fn validate_lifecycle_declaration(declared: bool, authenticated: bool) -> Result<()> {
    if declared && !authenticated {
        bail!("OpenAI lifecycle declarations require an OS-authenticated launched plugin process");
    }
    Ok(())
}

fn peer_process_id(stream: &LocalStream) -> Result<u32> {
    match stream {
        #[cfg(unix)]
        LocalStream::Unix(stream) => {
            let pid = stream
                .peer_cred()?
                .pid()
                .context("OS plugin peer process ID unavailable")?;
            u32::try_from(pid).context("invalid OS plugin peer process ID")
        }
        #[cfg(windows)]
        LocalStream::PipeServer(server) => {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::System::Pipes::GetNamedPipeClientProcessId;
            let mut pid = 0;
            // SAFETY: the connected server owns this live handle and pid points
            // to writable storage for the duration of the synchronous OS call.
            if unsafe { GetNamedPipeClientProcessId(server.as_raw_handle() as _, &mut pid) } == 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(pid)
        }
        _ => bail!("plugin control transport has no OS peer process identity"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unverified_connections_keep_legacy_protocol_but_cannot_declare_new_capability() {
        assert!(validate_lifecycle_declaration(false, false).is_ok());
        assert!(validate_lifecycle_declaration(true, false).is_err());
        assert!(validate_lifecycle_declaration(true, true).is_ok());
        let (host, _) = tokio::io::duplex(64);
        let memory = LocalStream::Memory(host);
        assert!(authenticate(&memory, None).is_err());
        assert!(!is_authenticated(&memory, None, false));
        assert!(is_authenticated(&memory, None, true));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn launched_control_client() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let Ok(endpoint) = std::env::var("PLUGIN_PEER_AUTH_TEST_ENDPOINT") else {
            if std::env::var_os("PLUGIN_PEER_AUTH_TEST_HOLD").is_some() {
                std::future::pending::<()>().await;
            }
            return;
        };
        let mut stream = tokio::net::UnixStream::connect(endpoint).await.unwrap();
        stream.write_all(b"connected").await.unwrap();
        let mut finish = [0];
        let _ = stream.read(&mut finish).await;
    }

    #[cfg(unix)]
    fn process_spec() -> super::super::ExternalPluginSpec {
        super::super::ExternalPluginSpec {
            name: "peer-auth-observer".into(),
            command: std::env::current_exe().unwrap().to_str().unwrap().into(),
            args: vec![
                "--exact".into(),
                "plugin::runtime::socket_auth::tests::launched_control_client".into(),
            ],
            url: None,
            env: [("PLUGIN_PEER_AUTH_TEST_HOLD".into(), "1".into())].into(),
            startup: Default::default(),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            installed_metadata: None,
            openai_exchange_grant: Some(Box::default()),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unverified_lifecycle_manifest_is_rejected_before_and_after_late_grants() {
        use std::sync::atomic::Ordering;
        let mut spec = process_spec();
        spec.openai_exchange_grant = None;
        let mut plugin = super::super::tests::plugin_for_spec(spec);
        plugin.authenticated_peer.store(false, Ordering::Release);
        let mut init = super::super::proto::InitializeResponse {
            plugin_id: plugin.spec.name.clone(),
            plugin_protocol_version: super::super::PROTOCOL_VERSION,
            ..Default::default()
        };
        assert!(plugin.validate_initialize_response(&init).is_ok());
        init.manifest = Some(super::super::proto::PluginManifest {
            openai_exchange_hook: Some(Box::new(
                mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe"),
            )),
            ..Default::default()
        });
        assert!(
            plugin
                .validate_initialize_response(&init)
                .unwrap_err()
                .to_string()
                .contains("OS-authenticated")
        );
        plugin.spec.openai_exchange_grant = Some(Box::default());
        assert!(
            plugin
                .validate_initialize_response(&init)
                .unwrap_err()
                .to_string()
                .contains("OS-authenticated")
        );
        plugin.authenticated_peer.store(true, Ordering::Release);
        plugin.spec.openai_exchange_grant = None;
        assert!(plugin.validate_initialize_response(&init).is_ok());
        plugin
            .handle_runtime_failure(None, "current connection failed".into())
            .await;
        assert!(!plugin.authenticated_peer.load(Ordering::Acquire));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn real_wrong_process_first_connection_is_rejected_before_runtime_installation() {
        use super::super::super::transport::LocalListener;
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("control.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        // A same-user process queues first, before the configured child starts.
        let _attacker = tokio::net::UnixStream::connect(&path).await.unwrap();
        let plugin = super::super::tests::plugin_for_spec(process_spec());
        let mut child = plugin
            .spawn_child_process(path.to_str().unwrap(), "unix")
            .unwrap();
        let error = plugin
            .await_plugin_connection(LocalListener::Unix(listener, path), child.id())
            .await
            .err()
            .expect("wrong peer must fail");
        assert!(
            error
                .to_string()
                .contains("does not match the launched process")
        );
        assert!(plugin.runtime.lock().await.is_none());
        child.kill().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn real_launched_child_is_verified_even_without_initial_grants() {
        use super::super::super::transport::LocalListener;
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("control.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let mut spec = process_spec();
        spec.openai_exchange_grant = None;
        spec.env.insert(
            "PLUGIN_PEER_AUTH_TEST_ENDPOINT".into(),
            path.to_str().unwrap().into(),
        );
        let plugin = super::super::tests::plugin_for_spec(spec);
        let mut child = plugin
            .spawn_child_process(path.to_str().unwrap(), "unix")
            .unwrap();
        let stream = plugin
            .await_plugin_connection(LocalListener::Unix(listener, path), child.id())
            .await
            .unwrap();
        assert!(is_authenticated(&stream, child.id(), false));
        assert!(
            validate_lifecycle_declaration(true, is_authenticated(&stream, child.id(), false))
                .is_ok()
        );
        assert!(authenticate(&stream, None).is_err());
        drop(stream);
        assert!(child.wait().await.unwrap().success());
    }
}

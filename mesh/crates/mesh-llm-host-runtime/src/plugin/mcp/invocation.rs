//! Exact callback ownership for capable plugins and serialized legacy calls.

use rmcp::{RoleServer, service::Peer};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

#[derive(Clone, Default)]
pub(super) struct Invocations {
    plugins: Arc<Mutex<HashMap<String, Arc<PerPlugin>>>>,
}

#[derive(Default)]
struct PerPlugin {
    serial: Arc<AsyncMutex<()>>,
    current: Arc<Mutex<Weak<Invocation>>>,
    live: Mutex<HashMap<u64, Weak<Invocation>>>,
    correlated: std::sync::atomic::AtomicBool,
}

pub(super) struct Invocation {
    pub(super) peer: Peer<RoleServer>,
    pub(super) id: Option<u64>,
    plugin: Arc<PerPlugin>,
    _serial: Option<OwnedMutexGuard<()>>,
}

/// The call owns registration; an already-running callback may retain its peer
/// without keeping the completed call eligible for new callbacks.
pub(super) struct InvocationGuard(Arc<Invocation>);

impl std::ops::Deref for InvocationGuard {
    type Target = Arc<Invocation>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

static NEXT_INVOCATION: AtomicU64 = AtomicU64::new(1);
tokio::task_local! { static FORWARDED_INVOCATION: u64; }

pub(crate) fn forwarded_id() -> Option<u64> {
    FORWARDED_INVOCATION.try_with(|id| *id).ok()
}

impl Invocation {
    pub(super) async fn scope<T>(&self, future: impl std::future::Future<Output = T>) -> T {
        match self.id {
            Some(id) => FORWARDED_INVOCATION.scope(id, future).await,
            None => future.await,
        }
    }
}

impl Drop for InvocationGuard {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            self.plugin
                .live
                .lock()
                .expect("MCP invocation lock poisoned")
                .remove(&id);
        }
    }
}

impl Invocations {
    #[cfg(test)]
    pub(super) async fn bind(&self, plugin_name: &str, peer: Peer<RoleServer>) -> InvocationGuard {
        self.bind_with_capability(plugin_name, peer, false).await
    }

    pub(super) async fn bind_with_capability(
        &self,
        plugin_name: &str,
        peer: Peer<RoleServer>,
        correlated: bool,
    ) -> InvocationGuard {
        let plugin = self
            .plugins
            .lock()
            .expect("MCP plugin invocation lock poisoned")
            .entry(plugin_name.to_owned())
            .or_default()
            .clone();
        plugin.correlated.store(correlated, Ordering::Release);
        let serial = if correlated {
            None
        } else {
            Some(plugin.serial.clone().lock_owned().await)
        };
        let id = correlated.then(|| NEXT_INVOCATION.fetch_add(1, Ordering::Relaxed));
        let invocation = Arc::new(Invocation {
            peer,
            id,
            plugin: plugin.clone(),
            _serial: serial,
        });
        if let Some(id) = id {
            plugin
                .live
                .lock()
                .expect("MCP invocation lock poisoned")
                .insert(id, Arc::downgrade(&invocation));
        } else {
            *plugin.current.lock().expect("MCP invocation lock poisoned") =
                Arc::downgrade(&invocation);
        }
        InvocationGuard(invocation)
    }

    #[cfg(test)]
    pub(super) fn current(&self, plugin_name: &str) -> Option<Arc<Invocation>> {
        self.resolve(plugin_name, None)
    }

    pub(super) fn resolve(&self, plugin_name: &str, id: Option<u64>) -> Option<Arc<Invocation>> {
        let plugin = self
            .plugins
            .lock()
            .expect("MCP plugin invocation lock poisoned")
            .get(plugin_name)
            .cloned()?;
        match id {
            Some(id) => plugin
                .live
                .lock()
                .expect("MCP invocation lock poisoned")
                .get(&id)
                .and_then(Weak::upgrade),
            None if plugin.correlated.load(Ordering::Acquire) => None,
            None => plugin
                .current
                .lock()
                .expect("MCP invocation lock poisoned")
                .upgrade(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;

    #[tokio::test]
    async fn correlated_callbacks_reject_missing_unknown_finished_and_foreign_ids() {
        use rmcp::ServiceExt;
        struct Server;
        impl rmcp::ServerHandler for Server {}
        let (server, client) = tokio::io::duplex(4096);
        let server = tokio::spawn(async move { Server.serve(server).await.unwrap() });
        let client = ().serve(client).await.unwrap();
        let server = server.await.unwrap();
        let invocations = Invocations::default();
        let first = invocations
            .bind_with_capability("adapter", server.peer().clone(), true)
            .await;
        let id = first.id.unwrap();
        let callback = invocations.resolve("adapter", Some(id)).unwrap();
        let second = invocations
            .bind_with_capability("adapter", server.peer().clone(), true)
            .now_or_never()
            .expect("capable invocations must not serialize");
        assert!(invocations.resolve("adapter", None).is_none());
        assert!(invocations.resolve("adapter", Some(u64::MAX)).is_none());
        assert!(invocations.resolve("other", Some(id)).is_none());
        assert!(Arc::ptr_eq(&first, &callback));

        drop(first);

        assert!(invocations.resolve("adapter", Some(id)).is_none());
        assert!(Arc::ptr_eq(
            &second,
            &invocations.resolve("adapter", second.id).unwrap()
        ));
        drop(callback);
        drop(client);
        drop(server);
    }

    #[tokio::test]
    async fn invocation_ownership_is_released_only_after_callbacks_finish() {
        use rmcp::ServiceExt;
        let (server, client) = tokio::io::duplex(4096);
        struct Server;
        impl rmcp::ServerHandler for Server {}
        let server = tokio::spawn(async move { Server.serve(server).await.unwrap() });
        let client = ().serve(client).await.unwrap();
        let server = server.await.unwrap();
        let invocations = Invocations::default();
        let first = invocations.bind("adapter", server.peer().clone()).await;
        let callback = invocations.current("adapter").unwrap();
        let second = invocations.bind("adapter", server.peer().clone());
        tokio::pin!(second);

        assert!(second.as_mut().now_or_never().is_none());
        drop(first);
        assert!(second.as_mut().now_or_never().is_none());
        drop(callback);

        let second = second.await;
        assert!(Arc::ptr_eq(
            &second,
            &invocations.current("adapter").unwrap()
        ));
        drop(second);
        assert!(invocations.current("adapter").is_none());
        drop(client);
        drop(server);
    }

    #[tokio::test]
    async fn different_plugins_can_bind_while_an_invocation_is_held() {
        use rmcp::ServiceExt;
        struct Server;
        impl rmcp::ServerHandler for Server {}
        let (server, client) = tokio::io::duplex(4096);
        let server = tokio::spawn(async move { Server.serve(server).await.unwrap() });
        let client = ().serve(client).await.unwrap();
        let server = server.await.unwrap();
        let invocations = Invocations::default();
        let first = invocations.bind("plugin-a", server.peer().clone()).await;

        let second = invocations
            .bind("plugin-b", server.peer().clone())
            .now_or_never()
            .expect("another plugin must not wait");

        assert!(Arc::ptr_eq(
            &first,
            &invocations.current("plugin-a").unwrap()
        ));
        assert!(Arc::ptr_eq(
            &second,
            &invocations.current("plugin-b").unwrap()
        ));
        drop(client);
        drop(server);
    }

    #[tokio::test]
    async fn callback_from_another_plugin_cannot_select_an_active_invocation() {
        use crate::plugin::PluginRpcBridge;
        use rmcp::ServiceExt;
        struct Server;
        impl rmcp::ServerHandler for Server {}
        let (server, client) = tokio::io::duplex(4096);
        let server = tokio::spawn(async move { Server.serve(server).await.unwrap() });
        let client = ().serve(client).await.unwrap();
        let server = server.await.unwrap();
        let bridge = super::super::ActiveBridge::default();
        let _first = bridge
            .bind_invocation("plugin-a", server.peer().clone())
            .await;

        let result = bridge
            .handle_request("plugin-b".into(), "ping".into(), "{}".into())
            .await;

        assert!(result.is_err());
        assert!(bridge.current_peer("plugin-a").is_ok());
        drop(client);
        drop(server);
    }
}

// A join token that names this node's own identity: issue #2352, papercut 2.
//
// A join file holding this node's own invite token (e.g. one token file
// pushed to every node in the mesh) makes the 60 s rejoin pass report the
// #1699 duplicate-key error forever. When no other local instance is
// running, the token is simply this node's own: say so, and stop retrying.

use std::path::Path;

/// Points `MESH_LLM_RUNTIME_ROOT` at an empty temporary directory so the
/// local-instance scan is deterministic instead of depending on whatever is
/// running on the developer's machine. The enclosing test must be `#[serial]`:
/// the mutation is process-wide.
struct RuntimeRootGuard {
    original: Option<String>,
}

impl RuntimeRootGuard {
    const KEY: &'static str = "MESH_LLM_RUNTIME_ROOT";

    fn set(root: &Path) -> Self {
        let original = std::env::var(Self::KEY).ok();
        #[allow(deprecated)]
        // SAFETY: the enclosing test is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var(Self::KEY, root.to_str().unwrap()) };
        Self { original }
    }
}

impl Drop for RuntimeRootGuard {
    fn drop(&mut self) {
        match self.original.take() {
            Some(value) => {
                #[allow(deprecated)]
                unsafe { std::env::set_var(Self::KEY, &value) };
            }
            None => {
                #[allow(deprecated)]
                unsafe { std::env::remove_var(Self::KEY) };
            }
        }
    }
}

#[tokio::test]
#[serial]
async fn joining_this_nodes_own_invite_token_is_classified_as_own_invite_token() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let _guard = RuntimeRootGuard::set(root.path());

    let node = make_test_node(super::NodeRole::Worker).await?;
    let own_token = node.invite_token().await;

    let err = node.join(&own_token).await.unwrap_err().to_string();
    assert!(
        err.contains("own invite token"),
        "a token that names this node's own id, with no other local instance running, must be \
         classified as the node's own invite token: {err}"
    );
    assert!(
        !err.contains("another process on this machine"),
        "the duplicate-key message is for two processes sharing one key (#1699), not a node \
         joining itself: {err}"
    );
    Ok(())
}

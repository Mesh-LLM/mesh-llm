//! Provider asset requests and their pending host-response lifetimes.

use super::{
    PendingHostResponseGuard, PluginContext, insert_pending_host_response, next_host_request_id,
};
use crate::models::{ASSET_CAPABILITY, AssetPlan, PreparedModelAssets};
use crate::proto;
use anyhow::{Result, bail};
use tokio::sync::oneshot;

impl PluginContext<'_> {
    /// Prepare a complete asset bundle through the host's shared cache.
    ///
    /// The host authorizes the source against this plugin's configured models.
    /// The owned context may be used from HTTP handlers after initialization.
    pub async fn prepare_model_assets(&self, plan: &AssetPlan) -> Result<PreparedModelAssets> {
        plan.validate()?;
        if !self.host_supports(ASSET_CAPABILITY) {
            bail!("host does not support model_assets.v1");
        }
        let request_id = next_host_request_id();
        let (tx, rx) = oneshot::channel();
        insert_pending_host_response(&self.pending_host_responses, request_id, tx);
        let mut guard =
            PendingHostResponseGuard::new(request_id, self.pending_host_responses.clone());
        self.send_payload(
            proto::envelope::Payload::PrepareModelAssetsRequest(proto::PrepareModelAssetsRequest {
                plan_json: serde_json::to_string(plan)?,
            }),
            request_id,
        )
        .await?;
        let response = tokio::time::timeout(std::time::Duration::from_secs(1800), rx)
            .await
            .map_err(|_| anyhow::anyhow!("model asset preparation timed out"))???;
        guard.disarm();
        match response.payload {
            Some(proto::envelope::Payload::PrepareModelAssetsResponse(response)) => {
                Ok(serde_json::from_str(&response.assets_json)?)
            }
            Some(proto::envelope::Payload::ErrorResponse(error)) => bail!(error.message),
            _ => bail!("unexpected model asset response"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{PendingHostResponses, remove_pending_host_response};
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use tokio::sync::mpsc;

    fn context(capabilities: &[&str]) -> (PluginContext<'static>, mpsc::Receiver<proto::Envelope>) {
        let (tx, rx) = mpsc::channel(8);
        let context = PluginContext::new("onnx".into(), tx, PendingHostResponses::default())
            .with_host_capabilities(capabilities.iter().map(|c| c.to_string()).collect());
        (context, rx)
    }

    fn plan() -> AssetPlan {
        AssetPlan {
            model: "owner/model@pin".into(),
            repo: "owner/model".into(),
            revision: "pin".into(),
            files: vec!["onnx/model.onnx".into()],
        }
    }

    #[tokio::test]
    async fn missing_capability_or_invalid_plan_sends_nothing() {
        let (old_host, mut outbound) = context(&[]);
        let error = old_host.prepare_model_assets(&plan()).await.unwrap_err();
        assert!(error.to_string().contains(ASSET_CAPABILITY));
        assert!(outbound.try_recv().is_err());
        assert!(old_host.pending_host_responses.lock().unwrap().is_empty());

        let (context, mut outbound) = context(&[ASSET_CAPABILITY]);
        let mut invalid = plan();
        invalid.files = vec!["../secret".into()];
        assert!(context.prepare_model_assets(&invalid).await.is_err());
        assert!(outbound.try_recv().is_err());
        assert!(context.pending_host_responses.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn host_answer_is_correlated_and_deserialized() {
        let (context, mut outbound) = context(&[ASSET_CAPABILITY]);
        let pending = context.pending_host_responses.clone();
        let assets = PreparedModelAssets {
            directory: PathBuf::from("bundle"),
            files: BTreeMap::from([(
                "onnx/model.onnx".into(),
                PathBuf::from("bundle/onnx/model.onnx"),
            )]),
            revision: "resolved-commit".into(),
        };
        let expected = assets.clone();
        let host = tokio::spawn(async move {
            let request = outbound.recv().await.unwrap();
            assert_eq!(request.plugin_id, "onnx");
            let Some(proto::envelope::Payload::PrepareModelAssetsRequest(body)) = request.payload
            else {
                panic!("expected an asset preparation request");
            };
            assert_eq!(
                serde_json::from_str::<AssetPlan>(&body.plan_json).unwrap(),
                plan()
            );
            let sender = remove_pending_host_response(&pending, request.request_id).unwrap();
            sender
                .send(Ok(proto::Envelope {
                    request_id: request.request_id,
                    payload: Some(proto::envelope::Payload::PrepareModelAssetsResponse(
                        proto::PrepareModelAssetsResponse {
                            assets_json: serde_json::to_string(&assets).unwrap(),
                        },
                    )),
                    ..Default::default()
                }))
                .unwrap();
        });
        assert_eq!(
            context.prepare_model_assets(&plan()).await.unwrap(),
            expected
        );
        host.await.unwrap();
        assert!(context.pending_host_responses.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn cancellation_removes_only_the_cancelled_asset_request() {
        let (context, mut outbound) = context(&[ASSET_CAPABILITY]);
        let pending = context.pending_host_responses.clone();
        let (sibling, _answer) = oneshot::channel();
        pending.lock().unwrap().insert(0, sibling);
        let task = tokio::spawn(async move { context.prepare_model_assets(&plan()).await });
        let sent = outbound.recv().await.unwrap();
        assert!(pending.lock().unwrap().contains_key(&sent.request_id));
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let remaining = pending.lock().unwrap();
        assert_eq!(remaining.len(), 1);
        assert!(remaining.contains_key(&0));
    }
}

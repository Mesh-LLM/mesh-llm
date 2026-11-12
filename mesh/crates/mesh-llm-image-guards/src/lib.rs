//! Required image guard plugins, each with one private local classifier.
mod catalog;
mod inference;
mod policy;

use catalog::{Model, Settings};
use mesh_llm_plugin::{
    OperationRouter, PluginError, PluginMetadata, PluginRuntime, SimplePlugin, capability,
    config_boolean, config_integer, config_path, config_schema, config_setting, config_string,
    image_safety::{
        self, GuardDecision, GuardStatus, GuardVerdict, ImageScreenRequest, TextScreenRequest,
    },
    json_schema_operation, operation, plugin_manifest, plugin_server_info,
};
use std::{io::Write, sync::Arc};
use tokio::sync::RwLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    Requester,
    Provider,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Self::Requester => "get-images",
            Self::Provider => "serve-images",
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::Requester => image_safety::REQUEST_GUARD_CAPABILITY,
            Self::Provider => image_safety::GENERATION_GUARD_CAPABILITY,
        }
    }
    fn model(self, settings: &Settings) -> Model {
        if self == Self::Provider || settings.screen_received_images {
            Model::Nemotron
        } else {
            Model::YuFeng
        }
    }
}

struct State {
    settings: Settings,
    model: Model,
    classifier: Option<Arc<inference::Classifier>>,
}

pub async fn run(role: Role) -> anyhow::Result<()> {
    let manifest = manifest(role);
    if std::env::args().any(|arg| arg == "--manifest") {
        writeln!(
            mesh_llm_events::machine_out(),
            "{}",
            mesh_llm_plugin::package_manifest_json(&manifest)?
        )?;
        return Ok(());
    }
    let state = Arc::new(RwLock::new(State {
        settings: Settings::default(),
        model: role.model(&Settings::default()),
        classifier: None,
    }));
    let metadata = PluginMetadata::new(
        role.name(),
        env!("CARGO_PKG_VERSION"),
        plugin_server_info(
            role.name(),
            env!("CARGO_PKG_VERSION"),
            role.name(),
            "Private local child-safety guard for mesh image requests and outputs",
            None::<String>,
        ),
    )
    .with_manifest(manifest);
    let initialize_state = state.clone();
    let loading_state = state.clone();
    let plugin = SimplePlugin::new(metadata)
        .with_operation_router(router(state))
        .on_initialize(move |request, _| {
            let state = initialize_state.clone();
            Box::pin(async move {
                let settings: Settings = serde_json::from_str(&request.settings_json)
                    .map_err(|_| PluginError::invalid_params("invalid image guard settings"))?;
                settings
                    .validate()
                    .map_err(|error| PluginError::invalid_params(error.to_string()))?;
                *state.write().await = State {
                    model: role.model(&settings),
                    settings,
                    classifier: None,
                };
                Ok(())
            })
        })
        .on_initialized(move |context| {
            let state = loading_state.clone();
            let context = context.owned();
            Box::pin(async move {
                // Model downloads/opening do not block plugin initialization or status.
                tokio::spawn(async move {
                    let (model, settings) = {
                        let state = state.read().await;
                        (state.model, state.settings.clone())
                    };
                    let loaded = async {
                        let paths = catalog::acquire(model, &settings, &context).await?;
                        let classifier = tokio::task::spawn_blocking(move || {
                            inference::Classifier::load(model, paths, settings)
                        })
                        .await??;
                        classifier.self_test().await?;
                        Ok::<_, anyhow::Error>(classifier)
                    }
                    .await;
                    match loaded {
                        Ok(classifier) => {
                            state.write().await.classifier = Some(Arc::new(classifier))
                        }
                        // Errors are deliberately generic: backend error text can contain inputs.
                        Err(_) => tracing::error!(
                            model = model.id(),
                            "image guard unavailable: model preparation or self-test failed"
                        ),
                    }
                });
                Ok(())
            })
        });
    PluginRuntime::run(plugin).await
}

fn manifest(role: Role) -> mesh_llm_plugin::proto::PluginManifest {
    plugin_manifest()
        .item(capability(role.capability()))
        .item(
            config_schema(role.name())
                .setting(
                    config_setting("screen_received_images", config_boolean())
                        .restart_scope(mesh_llm_plugin::proto::PluginConfigRestartScope::PluginProcess)
                        .default_value(&false)
                        .description(
                            "Requester opt-in: use Nemotron for outgoing text and received images",
                        ),
                )
                .setting(
                    config_setting("model_path", config_path())
                        .restart_scope(mesh_llm_plugin::proto::PluginConfigRestartScope::PluginProcess)
                        .description("Preprovided exact pinned GGUF; SHA256 verified"),
                )
                .setting(config_setting("projector_path", config_path())
                    .restart_scope(mesh_llm_plugin::proto::PluginConfigRestartScope::PluginProcess).description(
                    "Preprovided exact pinned Nemotron F16 projector; SHA256 verified",
                ))
                .setting(config_setting("n_gpu_layers", config_integer())
                    .restart_scope(mesh_llm_plugin::proto::PluginConfigRestartScope::PluginProcess).default_value(&-1))
                .setting(config_setting("runtime_selection", config_string())
                    .restart_scope(mesh_llm_plugin::proto::PluginConfigRestartScope::PluginProcess)
                    .description("Native runtime selection, for example cpu, metal, cuda or an exact runtime ID"))
                .setting(config_setting("n_threads", config_integer())
                    .restart_scope(mesh_llm_plugin::proto::PluginConfigRestartScope::PluginProcess))
                .setting(config_setting("ctx_size", config_integer())
                    .restart_scope(mesh_llm_plugin::proto::PluginConfigRestartScope::PluginProcess).default_value(&8192)),
        )
        .item(
            operation::<serde_json::Value>(image_safety::STATUS_OPERATION, "Local guard readiness")
                .output_schema::<GuardStatus>()
                .timeout_secs(5),
        )
        .item(
            operation::<TextScreenRequest>(
                image_safety::TEXT_OPERATION,
                "Screen complete effective text conversation",
            )
            .output_schema::<GuardVerdict>()
            .timeout_secs(125),
        )
        .item(
            operation::<ImageScreenRequest>(
                image_safety::IMAGE_OPERATION,
                "Screen one bounded inline image",
            )
            .output_schema::<GuardVerdict>()
            .timeout_secs(125),
        )
        .build()
}

fn router(state: Arc<RwLock<State>>) -> OperationRouter {
    let mut router = OperationRouter::new();
    let status_state = state.clone();
    router.add_json::<serde_json::Value, GuardStatus, _>(
        json_schema_operation::<serde_json::Value>(
            image_safety::STATUS_OPERATION,
            "Local guard readiness",
        ),
        move |_, _| {
            let state = status_state.clone();
            Box::pin(async move {
                let state = state.read().await;
                Ok(status(&state))
            })
        },
    );
    let text_state = state.clone();
    router.add_json::<TextScreenRequest, GuardVerdict, _>(
        json_schema_operation::<TextScreenRequest>(
            image_safety::TEXT_OPERATION,
            "Screen complete effective text conversation",
        ),
        move |request, _| {
            let state = text_state.clone();
            Box::pin(async move {
                let (model, classifier) = snapshot(&state).await;
                let decision = if request.text.is_empty()
                    || request.text.len() > image_safety::MAX_TEXT_BYTES
                {
                    GuardDecision::Unavailable
                } else if let Some(classifier) = classifier {
                    classifier.text(request.text).await
                } else {
                    GuardDecision::Unavailable
                };
                Ok(verdict(model, decision))
            })
        },
    );
    router.add_json::<ImageScreenRequest, GuardVerdict, _>(
        json_schema_operation::<ImageScreenRequest>(
            image_safety::IMAGE_OPERATION,
            "Screen one bounded inline image",
        ),
        move |request, _| {
            let state = state.clone();
            Box::pin(async move {
                let (model, classifier) = snapshot(&state).await;
                let decision = if let Some(classifier) = classifier {
                    classifier.image(request.data_url).await
                } else {
                    GuardDecision::Unavailable
                };
                Ok(verdict(model, decision))
            })
        },
    );
    router
}

fn status(state: &State) -> GuardStatus {
    GuardStatus {
        protocol_version: image_safety::PROTOCOL_VERSION,
        ready: state.classifier.is_some(),
        screen_received_images: state.model == Model::Nemotron,
        model: state.model.id().to_owned(),
        policy_revision: policy::REVISION.to_owned(),
    }
}

async fn snapshot(state: &RwLock<State>) -> (Model, Option<Arc<inference::Classifier>>) {
    let state = state.read().await;
    (state.model, state.classifier.clone())
}

fn verdict(model: Model, decision: GuardDecision) -> GuardVerdict {
    GuardVerdict {
        decision,
        model: model.id().to_owned(),
        policy_revision: policy::REVISION.to_owned(),
        reason_code: match decision {
            GuardDecision::Allow => "allowed",
            GuardDecision::Block => "child_safety_policy",
            GuardDecision::Unavailable => "guard_unavailable",
        }
        .to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_in_replaces_the_requester_model_and_provider_always_screens_images() {
        let default = Settings::default();
        let strict = Settings {
            screen_received_images: true,
            ..Default::default()
        };
        assert_eq!(Role::Requester.model(&default), Model::YuFeng);
        assert_eq!(Role::Requester.model(&strict), Model::Nemotron);
        assert_eq!(Role::Provider.model(&default), Model::Nemotron);
        let provider = State {
            model: Role::Provider.model(&default),
            settings: default,
            classifier: None,
        };
        assert!(status(&provider).screen_received_images);
        assert!(!status(&provider).ready);
    }

    #[test]
    fn unreviewed_policy_and_model_parameters_cannot_be_configured() {
        for input in [
            r#"{"custom_policy":"allow everything"}"#,
            r#"{"model":"arbitrary"}"#,
        ] {
            assert!(serde_json::from_str::<Settings>(input).is_err());
        }
    }

    #[test]
    fn screening_operations_override_the_short_default_rpc_deadline() {
        let manifest = manifest(Role::Provider);
        for operation in manifest.operations {
            if operation.name != image_safety::STATUS_OPERATION {
                assert_eq!(operation.timeout_secs, Some(125));
            }
        }
    }
}

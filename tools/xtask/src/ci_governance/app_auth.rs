//! One verified App, separate repository-restricted read and check tokens.
mod jwt;

use super::{
    activation::Anchors,
    collector::{self, ApiResponse, Clock, GithubClient, GithubRead, Request, SecretToken},
};
use crate::{command::DynResult, process::Cancellation};
use collector::request::{OBSERVER_PERMISSIONS, PUBLISHER_PERMISSIONS};
use serde::{Deserialize, de::DeserializeOwned};
use std::{collections::BTreeMap, path::Path};

pub(crate) struct PublisherClients {
    pub observer: GithubClient,
    pub publisher: GithubClient,
}

pub(crate) fn observer_client(
    root: &Path,
    anchors: &Anchors,
    cancellation: &Cancellation,
    clock: &impl Clock,
) -> DynResult<GithubClient> {
    let mut app = verified_app(root, anchors, cancellation, clock)?;
    let token = mint(&mut app, anchors, false, clock)?;
    GithubClient::installation(root, cancellation, token)
}

pub(crate) fn publisher_clients(
    root: &Path,
    anchors: &Anchors,
    cancellation: &Cancellation,
    clock: &impl Clock,
) -> DynResult<PublisherClients> {
    let mut app = verified_app(root, anchors, cancellation, clock)?;
    let observer = mint(&mut app, anchors, false, clock)?;
    let publisher = mint(&mut app, anchors, true, clock)?;
    Ok(PublisherClients {
        observer: GithubClient::installation(root, cancellation, observer)?,
        publisher: GithubClient::installation(root, cancellation, publisher)?,
    })
}

fn verified_app(
    root: &Path,
    anchors: &Anchors,
    cancellation: &Cancellation,
    clock: &impl Clock,
) -> DynResult<GithubClient> {
    if anchors.observer_app_id != anchors.publisher_app_id
        || anchors.observer_installation_id != anchors.publisher_installation_id
    {
        return Err("governance requires one anchored App with separate downscoped tokens".into());
    }
    let credential = jwt::credential(root, anchors.publisher_app_id, cancellation, clock.now()?)?;
    let mut app = GithubClient::app_jwt(root, cancellation, credential)?;
    verify_app(&mut app, anchors)?;
    Ok(app)
}

#[derive(Deserialize)]
struct AppReply {
    id: u64,
    permissions: BTreeMap<String, String>,
}
#[derive(Deserialize)]
struct Account {
    login: String,
}
#[derive(Deserialize)]
struct InstallationReply {
    id: u64,
    app_id: u64,
    target_type: String,
    account: Account,
    permissions: BTreeMap<String, String>,
    repository_selection: String,
    suspended_at: serde_json::Value,
}

fn verify_app(client: &mut impl GithubRead, anchors: &Anchors) -> DynResult<()> {
    let app: AppReply = response(client.send(&Request::App)?, 200)?;
    if app.id != anchors.publisher_app_id || app.permissions != app_permissions() {
        return Err(
            "GitHub App identity or exact least-privilege permissions differ from reviewed anchors"
                .into(),
        );
    }
    let installation: InstallationReply = response(
        client.send(&Request::AppInstallation {
            id: anchors.publisher_installation_id,
        })?,
        200,
    )?;
    if installation.id != anchors.publisher_installation_id
        || installation.app_id != anchors.publisher_app_id
        || installation.target_type != "Organization"
        || installation.account.login != "Mesh-LLM"
        || installation.repository_selection != "selected"
        || !installation.suspended_at.is_null()
        || installation.permissions != app_permissions()
    {
        return Err("GitHub App installation, organization, selected repository scope or permissions differ from reviewed anchors".into());
    }
    Ok(())
}

fn app_permissions() -> BTreeMap<String, String> {
    let mut permissions: BTreeMap<_, _> = OBSERVER_PERMISSIONS
        .iter()
        .map(|(name, level)| ((*name).into(), (*level).into()))
        .collect();
    permissions.insert("checks".into(), "write".into());
    permissions
}

#[derive(Deserialize)]
struct TokenRepository {
    id: u64,
    full_name: String,
}
#[derive(Deserialize)]
struct MintReply {
    token: SecretToken,
    expires_at: String,
    permissions: BTreeMap<String, String>,
    repositories: Vec<TokenRepository>,
    repository_selection: String,
}

fn mint(
    client: &mut impl GithubRead,
    anchors: &Anchors,
    publisher: bool,
    clock: &impl Clock,
) -> DynResult<SecretToken> {
    let request = if publisher {
        Request::MintPublisherToken {
            installation_id: anchors.publisher_installation_id,
            repository_id: anchors.repository_id,
        }
    } else {
        Request::MintObserverToken {
            installation_id: anchors.observer_installation_id,
            repository_id: anchors.repository_id,
        }
    };
    let reply: MintReply = response(client.send(&request)?, 201)?;
    let expected: BTreeMap<String, String> = if publisher {
        PUBLISHER_PERMISSIONS.as_slice()
    } else {
        OBSERVER_PERMISSIONS.as_slice()
    }
    .iter()
    .map(|(name, level)| ((*name).into(), (*level).into()))
    .collect();
    let expires = collector::parse_timestamp(&reply.expires_at)
        .map_err(|_| "invalid minted governance token expiry")?;
    let now = clock.now()?;
    if reply.permissions != expected
        || reply.repository_selection != "selected"
        || reply.repositories.len() != 1
        || reply.repositories[0].id != anchors.repository_id
        || reply.repositories[0].full_name != "Mesh-LLM/mesh-llm"
        || expires <= now
        || expires - now > 3600
    {
        return Err("minted governance token must have exact downscoped permissions, one reviewed repository and bounded expiry".into());
    }
    Ok(reply.token)
}

pub(crate) fn response<T: DeserializeOwned>(
    response: ApiResponse,
    expected_status: u16,
) -> DynResult<T> {
    if response.status != expected_status || response.body.len() > 2 * 1024 * 1024 {
        return Err("governance API response status or size is not supported".into());
    }
    serde_json::from_slice(&response.body)
        .map_err(|_| "invalid governance API response schema".into())
}

#[cfg(test)]
mod tests;

//! Space handles, response types, and runtime/hardware/secrets APIs.
//!
//! Space-specific operations (runtime, hardware, secrets, variables, pause/restart, duplicate)
//! live as methods on [`HFRepository<RepoTypeSpace>`](HFRepository). Get a handle with
//! [`HFClient::space`](crate::HFClient::space) and call the methods directly — there is no separate
//! `HFSpace` wrapper.
//!
//! Response structs such as [`SpaceRuntime`] and [`SpaceVariable`] live in this module; bon-generated
//! `*Builder` types for each Space API appear here for rustdoc.

use bon::bon;
use serde::{Deserialize, Serialize};

use crate::error::HFResult;
use crate::repository::{HFRepository, RepoTypeSpace, RepoUrl};
use crate::retry;

/// Runtime state of a Space: stage, hardware, storage, and mounted volumes.
///
/// Returned by Space lifecycle methods such as
/// [`runtime`](HFRepository::runtime), [`pause`](HFRepository::pause), and
/// [`restart`](HFRepository::restart) on [`HFRepository<RepoTypeSpace>`](HFRepository).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpaceRuntime {
    /// Lifecycle stage of the Space (e.g., `"RUNNING"`, `"BUILDING"`, `"PAUSED"`, `"SLEEPING"`).
    pub stage: String,
    /// Current and requested hardware for the Space. `None` while a Space is `BUILDING` for the
    /// first time.
    #[serde(default)]
    pub hardware: Option<SpaceHardware>,
    /// Idle seconds before the Space is put to sleep. `None` means the default policy applies (Spaces
    /// on free `cpu-basic` hardware sleep after 48 hours; upgraded hardware never sleeps by default).
    #[serde(rename = "gcTimeout", default)]
    pub sleep_time: Option<u64>,
    /// Persistent storage attached to the Space (`"small"`, `"medium"`, or `"large"`), if any.
    #[serde(default)]
    pub storage: Option<String>,
    /// Hot-reloading state for the Space if a hot-reload commit is in progress.
    #[serde(default)]
    pub hot_reloading: Option<SpaceHotReloading>,
    /// Volumes mounted in the Space. `None` if none are attached.
    #[serde(default)]
    pub volumes: Option<Vec<Volume>>,
}

/// Hardware running a Space, with the most recently requested hardware.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpaceHardware {
    /// Hardware currently running the Space (e.g., `"cpu-basic"`, `"t4-medium"`). `None` if no
    /// hardware is assigned yet.
    #[serde(default)]
    pub current: Option<String>,
    /// Hardware most recently requested for the Space. May differ from `current` if a request is
    /// in flight. `None` if no hardware has been requested yet.
    #[serde(default)]
    pub requested: Option<String>,
}

/// Hot-reloading state for a Space.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpaceHotReloading {
    /// Status of the hot-reload commit, e.g., `"created"` or `"canceled"`.
    pub status: String,
    /// Per-replica statuses, each a `[replica_hash, status]` pair.
    pub replica_statuses: Vec<serde_json::Value>,
}

/// A volume mounted inside a Space (or Job) container.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Volume {
    /// Volume kind: `"bucket"`, `"model"`, `"dataset"`, or `"space"`.
    #[serde(rename = "type")]
    pub r#type: String,
    /// Source identifier, e.g., `"username/my-bucket"` or `"username/my-model"`.
    pub source: String,
    /// Mount path inside the container (must start with `/`).
    pub mount_path: String,
    /// Git revision for repo-backed volumes; defaults to `"main"` server-side when omitted.
    #[serde(default)]
    pub revision: Option<String>,
    /// Whether the mount is read-only. Forced to `true` for repo-backed volumes; defaults to `false`
    /// for buckets.
    #[serde(default)]
    pub read_only: Option<bool>,
    /// Subfolder inside the source to mount (e.g., `"path/to/dir"`).
    #[serde(default)]
    pub path: Option<String>,
}

/// A public environment variable set on a Space (non-secret).
///
/// Secrets are not returned — only variables declared via the Space's variables API.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpaceVariable {
    /// Variable name (e.g., `"MODEL_REPO_ID"`).
    pub key: String,
    /// Variable value. `None` if the Hub returns the variable without a value.
    pub value: Option<String>,
    /// Human-readable description of what the variable is for.
    pub description: Option<String>,
    /// ISO-8601 timestamp of the last update to this variable, if it has been updated since creation.
    pub updated_at: Option<String>,
}

#[bon]
impl HFRepository<RepoTypeSpace> {
    /// Fetch the current runtime state of the Space (hardware, stage, URL, etc.).
    ///
    /// Endpoint: `GET /api/spaces/{repo_id}/runtime`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use hf_hub::HFClient;
    /// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
    /// let client = HFClient::builder().build()?;
    /// let runtime = client.space("owner", "name").runtime().send().await?;
    /// # let _ = runtime;
    /// # Ok(()) }
    /// ```
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn runtime(&self) -> HFResult<SpaceRuntime> {
        let url = format!(
            "{}/api/spaces/{}/runtime",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(response.json().await?)
    }

    /// Request an upgrade or downgrade of the Space's hardware tier.
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/hardware`.
    ///
    /// # Parameters
    ///
    /// - `hardware` (required): hardware flavor to request (e.g., `"cpu-basic"`, `"t4-small"`, `"a10g-small"`).
    /// - `sleep_time`: seconds of inactivity before the Space is put to sleep. `0` means never sleep.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn request_hardware(
        &self,
        /// Hardware flavor to request (e.g., `"cpu-basic"`, `"t4-small"`, `"a10g-small"`).
        hardware: &str,
        /// Seconds of inactivity before the Space is put to sleep. `0` means never sleep.
        sleep_time: Option<u64>,
    ) -> HFResult<SpaceRuntime> {
        let url = format!(
            "{}/api/spaces/{}/hardware",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let mut body = serde_json::json!({ "flavor": hardware });
        if let Some(sleep_time) = sleep_time {
            body["sleepTime"] = serde_json::json!(sleep_time);
        }
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(response.json().await?)
    }

    /// Configure the number of seconds of inactivity before the Space is put to sleep.
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/sleeptime`.
    ///
    /// # Parameters
    ///
    /// - `sleep_time` (required): seconds of inactivity before the Space is put to sleep. `0` means never sleep.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn set_sleep_time(
        &self,
        /// Seconds of inactivity before the Space is put to sleep. `0` means never sleep.
        sleep_time: u64,
    ) -> HFResult<()> {
        let url = format!(
            "{}/api/spaces/{}/sleeptime",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let body = serde_json::json!({ "seconds": sleep_time });
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }

    /// Pause the Space, stopping it from consuming compute resources.
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/pause`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn pause(&self) -> HFResult<SpaceRuntime> {
        let url = format!(
            "{}/api/spaces/{}/pause",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(response.json().await?)
    }

    /// Restart a paused or errored Space.
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/restart`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn restart(&self) -> HFResult<SpaceRuntime> {
        let url = format!(
            "{}/api/spaces/{}/restart",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(response.json().await?)
    }

    /// Add or update a secret (encrypted environment variable) on the Space.
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/secrets`.
    ///
    /// # Parameters
    ///
    /// - `key` (required): secret key name.
    /// - `value` (required): secret value.
    /// - `description`: human-readable description of the secret.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn add_secret(
        &self,
        /// Secret key name.
        key: &str,
        /// Secret value.
        value: &str,
        /// Human-readable description of the secret.
        description: Option<&str>,
    ) -> HFResult<()> {
        let url = format!(
            "{}/api/spaces/{}/secrets",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let mut body = serde_json::json!({ "key": key, "value": value });
        if let Some(ref desc) = description {
            body["description"] = serde_json::json!(desc);
        }
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }

    /// Delete a secret from the Space by key.
    ///
    /// Endpoint: `DELETE /api/spaces/{repo_id}/secrets`.
    ///
    /// # Parameters
    ///
    /// - `key` (required): secret key name to delete.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_secret(
        &self,
        /// Secret key name to delete.
        key: &str,
    ) -> HFResult<()> {
        let url = format!(
            "{}/api/spaces/{}/secrets",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let body = serde_json::json!({ "key": key });
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .delete(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }

    /// Add or update a public environment variable on the Space.
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/variables`.
    ///
    /// # Parameters
    ///
    /// - `key` (required): variable key name.
    /// - `value` (required): variable value.
    /// - `description`: human-readable description of the variable.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn add_variable(
        &self,
        /// Variable key name.
        key: &str,
        /// Variable value.
        value: &str,
        /// Human-readable description of the variable.
        description: Option<&str>,
    ) -> HFResult<()> {
        let url = format!(
            "{}/api/spaces/{}/variables",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let mut body = serde_json::json!({ "key": key, "value": value });
        if let Some(ref desc) = description {
            body["description"] = serde_json::json!(desc);
        }
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }

    /// Delete a public environment variable from the Space by key.
    ///
    /// Endpoint: `DELETE /api/spaces/{repo_id}/variables`.
    ///
    /// # Parameters
    ///
    /// - `key` (required): variable key name to delete.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_variable(
        &self,
        /// Variable key name to delete.
        key: &str,
    ) -> HFResult<()> {
        let url = format!(
            "{}/api/spaces/{}/variables",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let body = serde_json::json!({ "key": key });
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .delete(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }

    /// Duplicate this Space to a new repository.
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/duplicate`.
    ///
    /// # Parameters
    ///
    /// - `to_id`: destination repository ID in `"owner/name"` format. Defaults to the authenticated user's namespace
    ///   with the same name.
    /// - `private`: whether the duplicated Space should be private.
    /// - `hardware`: hardware flavor identifier the duplicated Space should run on (see `Hardware values` below). When
    ///   omitted, the Hub picks a default (typically `"cpu-basic"`) — pass a value explicitly to mirror the source
    ///   Space's hardware.
    /// - `storage`: persistent storage tier identifier. One of `"small"`, `"medium"`, or `"large"`. Omit to duplicate
    ///   without persistent storage. See the [Spaces storage docs](https://huggingface.co/docs/hub/spaces-storage) for
    ///   current sizes.
    /// - `sleep_time`: seconds of inactivity before the Space is put to sleep. `0` means never sleep.
    /// - `secrets`: encrypted environment variables to set on the duplicated Space (see `Secret/variable shape` below).
    /// - `variables`: public (non-secret) environment variables to set on the duplicated Space (see `Secret/variable
    ///   shape` below).
    ///
    /// # Hardware values
    ///
    /// The Hub uses lowercase, hyphenated identifiers. Common values include `"cpu-basic"`, `"cpu-upgrade"`,
    /// `"t4-small"`, `"t4-medium"`, `"l4x1"`, `"l4x4"`, `"a10g-small"`, `"a10g-large"`, `"a10g-largex2"`,
    /// `"a10g-largex4"`, `"a100-large"`, `"h100"`, `"h100x8"`, and `"zero-a10g"`. The authoritative, current list lives
    /// in the [Spaces GPU hardware docs](https://huggingface.co/docs/hub/spaces-gpus).
    ///
    /// # Secret/variable shape
    ///
    /// Each entry in `secrets` or `variables` is a JSON object with the following keys:
    ///
    /// - `key` (string, required): variable/secret name.
    /// - `value` (string, required): variable/secret value.
    /// - `description` (string, optional): human-readable description.
    ///
    /// ```rust,no_run
    /// use serde_json::json;
    ///
    /// let secrets = vec![
    ///     json!({ "key": "HF_TOKEN", "value": "hf_...", "description": "API token" }),
    ///     json!({ "key": "DB_PASSWORD", "value": "s3cret" }),
    /// ];
    /// let variables = vec![json!({ "key": "MODEL_NAME", "value": "bert-base-uncased" })];
    /// ```
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn duplicate(
        &self,
        /// Destination repository ID in `"owner/name"` format. Defaults to the authenticated user's namespace
        /// with the same name.
        to_id: Option<&str>,
        /// Whether the duplicated Space should be private.
        private: Option<bool>,
        /// Hardware flavor identifier the duplicated Space should run on. When omitted, the Hub picks a default
        /// (typically `"cpu-basic"`) — pass a value explicitly to mirror the source Space's hardware.
        hardware: Option<&str>,
        /// Persistent storage tier identifier. One of `"small"`, `"medium"`, or `"large"`. Omit to duplicate
        /// without persistent storage.
        storage: Option<&str>,
        /// Seconds of inactivity before the Space is put to sleep. `0` means never sleep.
        sleep_time: Option<u64>,
        /// Encrypted environment variables to set on the duplicated Space.
        secrets: Option<Vec<serde_json::Value>>,
        /// Public (non-secret) environment variables to set on the duplicated Space.
        variables: Option<Vec<serde_json::Value>>,
    ) -> HFResult<RepoUrl> {
        let url = format!(
            "{}/api/spaces/{}/duplicate",
            self.hf_client.endpoint(),
            self.repo_path()
        );
        let mut body = serde_json::Map::new();
        if let Some(ref to_id) = to_id {
            body.insert("repository".into(), serde_json::json!(to_id));
        }
        if let Some(private) = private {
            body.insert("private".into(), serde_json::json!(private));
        }
        if let Some(ref hw) = hardware {
            body.insert("hardware".into(), serde_json::json!(hw));
        }
        if let Some(ref storage) = storage {
            body.insert("storage".into(), serde_json::json!(storage));
        }
        if let Some(sleep_time) = sleep_time {
            body.insert("sleepTime".into(), serde_json::json!(sleep_time));
        }
        if let Some(ref secrets) = secrets {
            body.insert("secrets".into(), serde_json::json!(secrets));
        }
        if let Some(ref variables) = variables {
            body.insert("variables".into(), serde_json::json!(variables));
        }
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(response.json().await?)
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFRepositorySync<RepoTypeSpace> {
    /// Blocking counterpart of [`HFRepository::runtime`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn runtime(&self) -> HFResult<SpaceRuntime> {
        self.runtime.block_on(self.inner.runtime().send())
    }

    /// Blocking counterpart of [`HFRepository::request_hardware`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn request_hardware(
        &self,
        hardware: &str,
        sleep_time: Option<u64>,
    ) -> HFResult<SpaceRuntime> {
        self.runtime.block_on(
            self.inner
                .request_hardware()
                .hardware(hardware)
                .maybe_sleep_time(sleep_time)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::set_sleep_time`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn set_sleep_time(&self, sleep_time: u64) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.set_sleep_time().sleep_time(sleep_time).send())
    }

    /// Blocking counterpart of [`HFRepository::pause`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn pause(&self) -> HFResult<SpaceRuntime> {
        self.runtime.block_on(self.inner.pause().send())
    }

    /// Blocking counterpart of [`HFRepository::restart`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn restart(&self) -> HFResult<SpaceRuntime> {
        self.runtime.block_on(self.inner.restart().send())
    }

    /// Blocking counterpart of [`HFRepository::add_secret`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn add_secret(&self, key: &str, value: &str, description: Option<&str>) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .add_secret()
                .key(key)
                .value(value)
                .maybe_description(description)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::delete_secret`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_secret(&self, key: &str) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.delete_secret().key(key).send())
    }

    /// Blocking counterpart of [`HFRepository::add_variable`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn add_variable(&self, key: &str, value: &str, description: Option<&str>) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .add_variable()
                .key(key)
                .value(value)
                .maybe_description(description)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::delete_variable`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_variable(&self, key: &str) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.delete_variable().key(key).send())
    }

    /// Blocking counterpart of [`HFRepository::duplicate`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn duplicate(
        &self,
        to_id: Option<&str>,
        private: Option<bool>,
        hardware: Option<&str>,
        storage: Option<&str>,
        sleep_time: Option<u64>,
        secrets: Option<Vec<serde_json::Value>>,
        variables: Option<Vec<serde_json::Value>>,
    ) -> HFResult<RepoUrl> {
        self.runtime.block_on(
            self.inner
                .duplicate()
                .maybe_to_id(to_id)
                .maybe_private(private)
                .maybe_hardware(hardware)
                .maybe_storage(storage)
                .maybe_sleep_time(sleep_time)
                .maybe_secrets(secrets)
                .maybe_variables(variables)
                .send(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{SpaceRuntime, SpaceVariable};
    use crate::repository::RepoType;

    #[test]
    fn test_space_handle_constructor() {
        let client = crate::HFClient::builder().build().unwrap();
        let space = client.space("huggingface-projects", "diffusers-gallery");

        assert_eq!(space.repo_type().singular(), "space");
        assert_eq!(space.repo_path(), "huggingface-projects/diffusers-gallery");
    }

    #[test]
    fn test_space_runtime_deserialize_hardware() {
        let json = r#"{"stage":"RUNNING","hardware":{"current":"cpu-basic","requested":"t4-medium"},"storage":null}"#;
        let runtime: SpaceRuntime = serde_json::from_str(json).unwrap();
        assert_eq!(runtime.stage, "RUNNING");
        let hardware = runtime.hardware.as_ref().unwrap();
        assert_eq!(hardware.current.as_deref(), Some("cpu-basic"));
        assert_eq!(hardware.requested.as_deref(), Some("t4-medium"));
        assert_eq!(runtime.storage, None);
    }

    #[test]
    fn test_space_runtime_deserialize_minimal() {
        let json = r#"{"stage":"BUILDING"}"#;
        let runtime: SpaceRuntime = serde_json::from_str(json).unwrap();
        assert_eq!(runtime.stage, "BUILDING");
        assert!(runtime.hardware.is_none());
    }

    #[test]
    fn test_space_runtime_sleep_time_from_gc_timeout() {
        let json = r#"{"stage":"RUNNING","gcTimeout":172800}"#;
        let runtime: SpaceRuntime = serde_json::from_str(json).unwrap();
        assert_eq!(runtime.sleep_time, Some(172800));
    }

    #[test]
    fn test_space_runtime_volumes_and_hot_reloading() {
        let json = r#"{
            "stage":"RUNNING",
            "volumes":[{"type":"model","source":"u/m","mountPath":"/data","readOnly":true}],
            "hotReloading":{"status":"created","replicaStatuses":[]}
        }"#;
        let runtime: SpaceRuntime = serde_json::from_str(json).unwrap();
        let volumes = runtime.volumes.as_ref().unwrap();
        assert_eq!(volumes.len(), 1);
        assert_eq!(volumes[0].r#type, "model");
        assert_eq!(volumes[0].mount_path, "/data");
        assert_eq!(volumes[0].read_only, Some(true));
        assert_eq!(runtime.hot_reloading.as_ref().unwrap().status, "created");
    }

    #[test]
    fn test_space_runtime_ignores_unknown_fields() {
        let json = r#"{"stage":"RUNNING","replicas":{"current":1},"someNewField":42}"#;
        let runtime: SpaceRuntime = serde_json::from_str(json).unwrap();
        assert_eq!(runtime.stage, "RUNNING");
    }

    #[test]
    fn test_space_variable_deserialize() {
        let json = r#"{"key":"MODEL_ID","value":"gpt2","description":"The model"}"#;
        let var: SpaceVariable = serde_json::from_str(json).unwrap();
        assert_eq!(var.key, "MODEL_ID");
        assert_eq!(var.value.as_deref(), Some("gpt2"));
    }
}

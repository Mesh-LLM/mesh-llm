//! Type-level markers for the four Hugging Face Hub repo kinds.
//!
//! The [`RepoType`] sealed trait and its four marker structs ([`RepoTypeModel`],
//! [`RepoTypeDataset`], [`RepoTypeSpace`], [`RepoTypeKernel`]) parameterize
//! [`HFRepository<T>`](super::HFRepository) and
//! [`HFRepositorySync<T>`](crate::HFRepositorySync) so the repo kind is encoded in the
//! type system. Methods that differ per repo kind (such as
//! [`info`](super::HFRepository::info)) dispatch at compile time.
//!
//! See [`RepoType`] for the full overview, including the table of strings each marker
//! returns and how to pass markers to client builders.
//!
//! All items here are re-exported flat at `hf_hub::repository::…` and at the crate root,
//! so the canonical paths are `hf_hub::RepoType` and `hf_hub::RepoTypeModel` etc.

mod sealed {
    pub trait Sealed {}
}

/// Type-level marker for a Hugging Face Hub repo kind (model, dataset, Space, or kernel).
///
/// `RepoType` is sealed — only the four marker structs in this module implement it:
/// [`RepoTypeModel`], [`RepoTypeDataset`], [`RepoTypeSpace`], [`RepoTypeKernel`]. Each is
/// zero-sized and serves as the type parameter on [`HFRepository<T>`](super::HFRepository)
/// and [`HFRepositorySync<T>`](crate::HFRepositorySync), so the repo kind is encoded in the
/// type system and methods that differ per kind (such as
/// [`info`](super::HFRepository::info)) dispatch at compile time.
///
/// # Picking a marker
///
/// Two equivalent ways:
///
/// ```rust,no_run
/// use hf_hub::{HFClient, RepoTypeDataset};
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::new()?;
///
/// // Typed shortcut — preferred when the kind is known at the call site:
/// let repo = client.dataset("rajpurkar", "squad");
///
/// // Explicit marker — useful when the kind comes from a generic context:
/// let repo = client.repository(RepoTypeDataset, "rajpurkar", "squad");
/// # let _ = repo; Ok(()) }
/// ```
///
/// Methods like [`HFClient::create_repository`](crate::HFClient::create_repository) /
/// [`delete_repository`](crate::HFClient::delete_repository) /
/// [`move_repository`](crate::HFClient::move_repository) take the marker as a builder
/// value:
///
/// ```rust,no_run
/// use hf_hub::{HFClient, RepoTypeSpace};
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// # let client = HFClient::new()?;
/// client
///     .create_repository()
///     .repo_id("acme/my-space")
///     .repo_type(RepoTypeSpace)
///     .space_sdk("static")
///     .send()
///     .await?;
/// # Ok(()) }
/// ```
///
/// # Trait methods
///
/// Each method returns the same value regardless of the receiver — they're properties
/// of the type, not the instance. The receiver is `&self` so callers with a value
/// in hand can write `marker.singular()`. Inside generic code where only the type
/// is in scope (`impl<T: RepoType>`), materialize the marker on the fly with
/// `T::default().singular()` — markers are zero-sized, so it's a no-op:
///
/// | trait method     | `RepoTypeModel` | `RepoTypeDataset` | `RepoTypeSpace` | `RepoTypeKernel` |
/// | ---------------- | --------------- | ----------------- | --------------- | ---------------- |
/// | [`singular`]     | `"model"`       | `"dataset"`       | `"space"`       | `"kernel"`       |
/// | [`plural`]       | `"models"`      | `"datasets"`      | `"spaces"`      | `"kernels"`      |
/// | [`url_prefix`]   | `""`            | `"datasets/"`     | `"spaces/"`     | `"kernels/"`     |
///
/// The markers also implement [`Display`](std::fmt::Display) (writing the singular
/// form), so `tracing` fields and `format!("{}", RepoTypeModel)` work directly.
///
/// [`singular`]: Self::singular
/// [`plural`]: Self::plural
/// [`url_prefix`]: Self::url_prefix
pub trait RepoType:
    sealed::Sealed + Default + Copy + Clone + std::fmt::Debug + Send + Sync + 'static
{
    /// Lowercase singular form of the repo kind: `"model"`, `"dataset"`, `"space"`, or `"kernel"`.
    /// Used in logs, error messages, and the `"type"` field of repo create/delete/move bodies.
    fn singular(&self) -> &'static str;

    /// Lowercase plural form, also the API segment in `/api/{plural}/{repo_id}` URLs:
    /// `"models"`, `"datasets"`, `"spaces"`, or `"kernels"`.
    fn plural(&self) -> &'static str;

    /// Path prefix used in resolve URLs (`<endpoint>/{prefix}{repo_id}/resolve/...`):
    /// `""` for [`RepoTypeModel`] (the model namespace is unprefixed) and `"<plural>/"`
    /// for the other three kinds.
    fn url_prefix(&self) -> &'static str;
}

/// Model-repository marker. See [`RepoType`] for the trait, the per-kind string
/// table, and usage examples; use [`HFClient::model`](crate::HFClient::model) for the
/// typed shortcut.
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeModel;

/// Dataset-repository marker. See [`RepoType`] for the trait, the per-kind string
/// table, and usage examples; use [`HFClient::dataset`](crate::HFClient::dataset) for
/// the typed shortcut.
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeDataset;

/// Space-repository marker. See [`RepoType`] for the trait, the per-kind string
/// table, and usage examples; use [`HFClient::space`](crate::HFClient::space) for the
/// typed shortcut. This is the marker used to access Space-only methods on
/// [`HFRepository`](super::HFRepository) such as
/// [`runtime`](super::HFRepository::runtime) and
/// [`pause`](super::HFRepository::pause).
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeSpace;

/// Kernel-repository marker. See [`RepoType`] for the trait, the per-kind string
/// table, and usage examples; use [`HFClient::kernel`](crate::HFClient::kernel) for the
/// typed shortcut. Note that
/// [`HFRepository::<RepoTypeKernel>::info`](super::HFRepository::info) returns a slim
/// shape — to get full model-style metadata for a kernel repo, build a model handle
/// for the same id and call `info()` on that.
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeKernel;

/// Runtime-tagged repository kind. Holds an enum value rather than encoding the kind in
/// the type system — useful when the kind is decided at runtime (CLI flag, config string,
/// upstream enum) and you don't want to thread a type parameter through downstream code.
///
/// Implements [`RepoType`] so it can be used as the type parameter on
/// [`HFRepository<RepoTypeAny>`](super::HFRepository), and implements
/// [`FromStr`](std::str::FromStr) for both singular and plural string forms
/// (e.g. `"model"`, `"models"`, `"dataset"`, `"datasets"`).
/// The trait methods ([`singular`](RepoType::singular), [`plural`](RepoType::plural),
/// [`url_prefix`](RepoType::url_prefix)) dispatch on the held variant at runtime.
///
/// All the standard file, commit, and repo-management APIs are available on
/// `HFRepository<RepoTypeAny>` — uploads/downloads, tree listings, commit history and diffs,
/// branches, tags, settings, and existence checks all work identically to a typed handle and
/// hit the right Hub endpoint for the runtime variant. The kind-specific
/// [`HFRepository::<RepoTypeModel>::info`](super::HFRepository::info) (and its per-kind
/// siblings) and the Space-only runtime helpers (`runtime`, `pause`, `restart`, secrets,
/// variables, hardware, duplicate) are *not* available — they live on the per-kind typed
/// handles.
#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RepoTypeAny {
    #[default]
    Model,
    Dataset,
    Space,
    Kernel,
}

impl sealed::Sealed for RepoTypeModel {}
impl sealed::Sealed for RepoTypeDataset {}
impl sealed::Sealed for RepoTypeSpace {}
impl sealed::Sealed for RepoTypeKernel {}
impl sealed::Sealed for RepoTypeAny {}

impl RepoType for RepoTypeModel {
    fn singular(&self) -> &'static str {
        "model"
    }
    fn plural(&self) -> &'static str {
        "models"
    }
    fn url_prefix(&self) -> &'static str {
        ""
    }
}

impl RepoType for RepoTypeDataset {
    fn singular(&self) -> &'static str {
        "dataset"
    }
    fn plural(&self) -> &'static str {
        "datasets"
    }
    fn url_prefix(&self) -> &'static str {
        "datasets/"
    }
}

impl RepoType for RepoTypeSpace {
    fn singular(&self) -> &'static str {
        "space"
    }
    fn plural(&self) -> &'static str {
        "spaces"
    }
    fn url_prefix(&self) -> &'static str {
        "spaces/"
    }
}

impl RepoType for RepoTypeKernel {
    fn singular(&self) -> &'static str {
        "kernel"
    }
    fn plural(&self) -> &'static str {
        "kernels"
    }
    fn url_prefix(&self) -> &'static str {
        "kernels/"
    }
}

impl std::fmt::Display for RepoTypeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.singular())
    }
}

impl std::fmt::Display for RepoTypeDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.singular())
    }
}

impl std::fmt::Display for RepoTypeSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.singular())
    }
}

impl std::fmt::Display for RepoTypeKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.singular())
    }
}

impl RepoType for RepoTypeAny {
    fn singular(&self) -> &'static str {
        match self {
            Self::Model => RepoTypeModel.singular(),
            Self::Dataset => RepoTypeDataset.singular(),
            Self::Space => RepoTypeSpace.singular(),
            Self::Kernel => RepoTypeKernel.singular(),
        }
    }
    fn plural(&self) -> &'static str {
        match self {
            Self::Model => RepoTypeModel.plural(),
            Self::Dataset => RepoTypeDataset.plural(),
            Self::Space => RepoTypeSpace.plural(),
            Self::Kernel => RepoTypeKernel.plural(),
        }
    }
    fn url_prefix(&self) -> &'static str {
        match self {
            Self::Model => RepoTypeModel.url_prefix(),
            Self::Dataset => RepoTypeDataset.url_prefix(),
            Self::Space => RepoTypeSpace.url_prefix(),
            Self::Kernel => RepoTypeKernel.url_prefix(),
        }
    }
}

impl std::fmt::Display for RepoTypeAny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.singular())
    }
}

impl std::str::FromStr for RepoTypeAny {
    type Err = crate::error::HFError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "model" | "models" => Ok(Self::Model),
            "dataset" | "datasets" => Ok(Self::Dataset),
            "space" | "spaces" => Ok(Self::Space),
            "kernel" | "kernels" => Ok(Self::Kernel),
            _ => Err(crate::error::HFError::InvalidParameter(format!(
                "unknown repo type: {s:?}. Expected one of: model(s), dataset(s), space(s), kernel(s)"
            ))),
        }
    }
}

#[cfg(test)]
mod repo_type_any_tests {
    use std::str::FromStr;

    use super::*;
    use crate::HFRepository;
    use crate::client::HFClient;

    #[test]
    fn from_str_accepts_singular_and_plural() {
        assert_eq!(RepoTypeAny::from_str("model").unwrap(), RepoTypeAny::Model);
        assert_eq!(RepoTypeAny::from_str("models").unwrap(), RepoTypeAny::Model);
        assert_eq!(
            RepoTypeAny::from_str("dataset").unwrap(),
            RepoTypeAny::Dataset
        );
        assert_eq!(
            RepoTypeAny::from_str("datasets").unwrap(),
            RepoTypeAny::Dataset
        );
        assert_eq!(RepoTypeAny::from_str("space").unwrap(), RepoTypeAny::Space);
        assert_eq!(RepoTypeAny::from_str("spaces").unwrap(), RepoTypeAny::Space);
        assert_eq!(
            RepoTypeAny::from_str("kernel").unwrap(),
            RepoTypeAny::Kernel
        );
        assert_eq!(
            RepoTypeAny::from_str("kernels").unwrap(),
            RepoTypeAny::Kernel
        );
    }

    #[test]
    fn from_str_rejects_unknown() {
        let err = RepoTypeAny::from_str("Model").unwrap_err();
        assert!(matches!(err, crate::error::HFError::InvalidParameter(_)));
        assert!(RepoTypeAny::from_str("Datasets").is_err());
        assert!(RepoTypeAny::from_str("").is_err());
        assert!(RepoTypeAny::from_str("collection").is_err());
    }

    #[test]
    fn trait_methods_dispatch_at_runtime() {
        let cases = [
            (RepoTypeAny::Model, "model", "models", ""),
            (RepoTypeAny::Dataset, "dataset", "datasets", "datasets/"),
            (RepoTypeAny::Space, "space", "spaces", "spaces/"),
            (RepoTypeAny::Kernel, "kernel", "kernels", "kernels/"),
        ];
        for (kind, sing, plur, prefix) in cases {
            assert_eq!(kind.singular(), sing);
            assert_eq!(kind.plural(), plur);
            assert_eq!(kind.url_prefix(), prefix);
            assert_eq!(format!("{kind}"), sing);
        }
    }

    #[test]
    fn handle_from_str_via_repo_type_any() {
        let client = HFClient::builder().build().unwrap();

        let kind: RepoTypeAny = "datasets".parse().unwrap();
        let repo: HFRepository<RepoTypeAny> = client.repository(kind, "rajpurkar", "squad");

        assert_eq!(repo.owner(), "rajpurkar");
        assert_eq!(repo.name(), "squad");
        assert_eq!(repo.repo_path(), "rajpurkar/squad");
        assert_eq!(repo.repo_type().singular(), "dataset");
        assert_eq!(repo.repo_type().plural(), "datasets");
        assert_eq!(repo.repo_type().url_prefix(), "datasets/");
    }

    #[test]
    fn default_is_model() {
        assert_eq!(RepoTypeAny::default(), RepoTypeAny::Model);
        assert_eq!(RepoTypeAny::default().singular(), "model");
    }

    #[test]
    fn handle_repo_type_reflects_runtime_kind() {
        let client = HFClient::builder().build().unwrap();
        for (kind_str, expected_singular, expected_plural, expected_prefix) in [
            ("model", "model", "models", ""),
            ("models", "model", "models", ""),
            ("dataset", "dataset", "datasets", "datasets/"),
            ("datasets", "dataset", "datasets", "datasets/"),
            ("space", "space", "spaces", "spaces/"),
            ("kernels", "kernel", "kernels", "kernels/"),
        ] {
            let kind: RepoTypeAny = kind_str.parse().unwrap();
            let repo = client.repository(kind, "owner", "name");
            assert_eq!(
                repo.repo_type().singular(),
                expected_singular,
                "input {kind_str:?}"
            );
            assert_eq!(
                repo.repo_type().plural(),
                expected_plural,
                "input {kind_str:?}"
            );
            assert_eq!(
                repo.repo_type().url_prefix(),
                expected_prefix,
                "input {kind_str:?}"
            );
        }
    }
}

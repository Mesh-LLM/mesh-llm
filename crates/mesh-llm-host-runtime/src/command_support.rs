//! Public support API for the `mesh-llm` binary command handlers.
//!
//! This is intentionally not a command implementation module. It exposes the
//! host-runtime operations the binary crate needs after CLI ownership moved out
//! of host-runtime.

pub mod discovery {
    pub mod nostr {
        pub use crate::network::nostr::{
            DiscoveredMesh, MeshFilter, MeshListing, discover, rotate_keys, score_mesh,
        };
    }

    pub use crate::discovery::{DiscoveryScope, MeshDiscoveryMode};
    pub use crate::mesh::load_last_mesh_id;
    pub use crate::network::discovery::{LAN_SERVICE_TYPE, LanDiscoveredMesh, discover_lan};
    pub use crate::runtime::instance::{
        RuntimeProcessTarget, collect_runtime_stop_targets, runtime_root,
    };
    pub use crate::runtime::nostr_relays;
}

pub mod models {
    pub mod election {
        pub use crate::inference::election::total_model_bytes;
    }

    pub mod skippy {
        pub use crate::inference::skippy::{
            CertificationGateStatus, SkippyCertificationRequest, certify_layer_package,
            identity_from_layer_package, is_layer_package_ref, materialized_stage_cache_dir,
            materialized_stages_for_sources, prune_unpinned_materialized_stages,
            remove_materialized_stages_for_sources, resolve_hf_package_to_local,
        };
    }

    pub use crate::models::remote_catalog;
    pub use crate::models::{
        DeleteResult, ModelCapabilities, ModelCleanupPlan, ModelCleanupResult, ModelDetails,
        ResolvedModel, SearchArtifactFilter, SearchHit, SearchProgress, SearchSort,
        ShowVariantsProgress, delete, download_model_ref_with_progress_details,
        execute_model_cleanup, find_model_path, find_remote_catalog_model_exact,
        huggingface_hub_cache_dir, huggingface_identity_for_path, installed_model_capabilities,
        installed_model_display_name, installed_model_huggingface_ref,
        layered_package_layer_count_for_path, layered_package_total_bytes_for_path,
        load_model_usage_record_for_path, model_usage_cache_dir, plan_model_cleanup,
        remote_catalog_model_draft_ref, remote_catalog_model_ref, run_update,
        scan_installed_models, search_catalog_json_payload, search_catalog_models,
        search_huggingface, search_huggingface_json_payload, show_exact_model,
        show_model_variants_with_progress,
    };
    pub use crate::models::{capabilities, catalog};
}

pub mod plugin {
    pub use crate::plugin::{
        ExternalPluginSpec, GpuAssignment, GpuConfig, MeshConfig, PluginHostMode, PluginManager,
        ResolvedPlugins, ToolCallResult, bundled_cli_plugin_spec, load_config,
    };
    pub use crate::runtime::load_resolved_plugins;
}

pub mod config {
    pub use crate::plugin::{config_path, validate_config_file};
    pub use mesh_llm_config::{
        ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource,
        ConfigDiagnosticSeverity, ConfigDiagnosticSource, ConfigPath,
    };
}

pub mod runtime_instances {
    pub use crate::runtime::instance::{LocalInstanceSnapshot, runtime_root, scan_local_instances};
}

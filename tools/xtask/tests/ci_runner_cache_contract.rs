//! Actual local action decisions with finite external tool boundaries.
#![cfg(unix)]
#[allow(dead_code, unused_imports)]
#[path = "../src/process/mod.rs"]
mod process;
#[path = "ci_runner_cache_contract/runtime.rs"]
mod runtime;
#[path = "ci_runner_cache_contract/support.rs"]
mod support;
#[allow(dead_code)]
#[path = "../src/ci_validation/lane_results/workflow_yaml.rs"]
mod workflow_yaml;

#[path = "../src/ci_validation/workflow_guards/cache_authority.rs"]
mod cache_authority;
use workflow_yaml::Node;
#[path = "ci_runner_cache_contract/authority.rs"]
mod authority;
#[path = "ci_runner_cache_contract/installer.rs"]
mod installer;

#[path = "../src/ci_validation/workflow_guards/cache_callers.rs"]
mod cache_callers;
#[path = "../src/ci_validation/workflow_guards/cache_consumers.rs"]
mod cache_consumers;
#[path = "../src/ci_validation/workflow_guards/cache_predicate.rs"]
mod cache_predicate;
#[path = "ci_runner_cache_contract/consumers.rs"]
mod consumers;

#[path = "../src/ci_validation/workflow_guards/cache_boundaries.rs"]
mod cache_boundaries;

#[path = "../src/ci_validation/workflow_guards/cache_evidence.rs"]
mod cache_evidence;
#[path = "../src/ci_validation/workflow_guards/cache_identity.rs"]
mod cache_identity;
#[path = "ci_runner_cache_contract/evidence_contracts.rs"]
mod evidence_contracts;
#[path = "ci_runner_cache_contract/evidence_runtime.rs"]
mod evidence_runtime;

#[path = "ci_runner_cache_contract/publication.rs"]
mod publication;

#[path = "ci_runner_cache_contract/controller_checks.rs"]
mod controller_checks;

#[path = "ci_runner_cache_contract/native_sdk_workflow.rs"]
mod native_sdk_workflow;
#[path = "ci_runner_cache_contract/runtime_action.rs"]
mod runtime_action;
#[path = "ci_runner_cache_contract/sdk_prepare.rs"]
mod sdk_prepare;
#[path = "ci_runner_cache_contract/sdk_resolver.rs"]
mod sdk_resolver;
#[path = "ci_runner_cache_contract/static_abi_action.rs"]
mod static_abi_action;
#[path = "ci_runner_cache_contract/static_abi_workflow.rs"]
mod static_abi_workflow;
#[path = "ci_runner_cache_contract/swift_workflow.rs"]
mod swift_workflow;

#[path = "ci_runner_cache_contract/verified_pr_authority.rs"]
mod verified_pr_authority;

#[path = "ci_runner_cache_contract/protected_native_producer.rs"]
mod protected_native_producer;

#[path = "ci_runner_cache_contract/authority_callers.rs"]
mod authority_callers;

#[path = "ci_runner_cache_contract/safetensors_smoke/mod.rs"]
mod safetensors_smoke;

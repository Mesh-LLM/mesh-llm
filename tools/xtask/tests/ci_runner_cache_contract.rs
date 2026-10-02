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

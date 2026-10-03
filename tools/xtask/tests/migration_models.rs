//! `models {generate,resolve,restore-inputs}` parity with
//! `scripts/generate-test-model-manifests.py`,
//! `scripts/resolve-test-model-manifest.py` and the resolve step of
//! `.github/actions/restore-test-model/action.yml`. Every expectation under
//! `fixtures/models` is legacy output captured with Python 3.13 and bash 5
//! (capture scripts: `.omo/evidence/task-14-capture/`).

#[path = "migration_models/support.rs"]
mod support;

#[path = "migration_models/generator.rs"]
mod generator;
#[path = "migration_models/live.rs"]
mod live;
#[path = "migration_models/resolver.rs"]
mod resolver;
#[path = "migration_models/resolver_matrix.rs"]
mod resolver_matrix;
#[path = "migration_models/restore_inputs.rs"]
mod restore_inputs;

#[path = "migration_models/current_contracts.rs"]
mod current_contracts;

#[path = "migration_models/repository_contracts.rs"]
mod repository_contracts;

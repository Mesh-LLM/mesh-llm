use anyhow::{Result, ensure};
use mesh_llm_cli::shared_serving::SharedServingArgs;
use mesh_llm_host_runtime::sdk::shared_serving::{
    ServingMembershipConfig, SharedServingConfig, run_shared_serving,
};

pub(super) async fn run(args: &SharedServingArgs) -> Result<()> {
    let mut trust = std::collections::BTreeMap::new();
    for named in &args.membership_trust {
        ensure!(
            trust
                .insert(named.name.clone(), named.path.clone())
                .is_none(),
            "duplicate membership trust store"
        );
        ensure!(
            args.membership
                .iter()
                .any(|member| member.name == named.name),
            "trust store names an unknown membership"
        );
    }
    let memberships = args
        .membership
        .iter()
        .map(|member| ServingMembershipConfig {
            name: member.name.clone(),
            token_file: member.path.clone(),
            trust_store: trust.remove(&member.name),
            owner_key: args.attestation_key.clone(),
        })
        .collect();
    run_shared_serving(
        SharedServingConfig {
            model_path: args.model_path.clone(),
            model_id: args.model_id.clone(),
            state_dir: args.state_dir.clone(),
            context_size: args.context_size,
            gpu_layers: args.gpu_layers,
            concurrency: args.concurrency.into(),
            queue_capacity: args.queue_capacity.into(),
            membership_queue_capacity: args.membership_queue_capacity.into(),
            lan_only: args.lan_only,
            bind_ip: args.bind_ip,
            memberships,
        },
        args.management_port,
    )
    .await
}

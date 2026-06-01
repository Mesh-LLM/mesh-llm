use anyhow::Result;
use mesh_llm_cli::{BinaryFlavor, Cli, Command};
use mesh_llm_system::{autoupdate, backend};

pub async fn run_update(cli: &Cli) -> Result<()> {
    let (requested_version, flavor, detect_flavor) = match &cli.command {
        Some(Command::Update {
            version,
            flavor,
            detect_flavor,
        }) => (
            version.as_deref(),
            binary_flavor_to_backend(*flavor),
            *detect_flavor,
        ),
        _ => (None, None, false),
    };
    autoupdate::run_update_command(autoupdate::UpdateCommandOptions {
        flavor,
        detect_flavor,
        requested_version,
        current_version: env!("CARGO_PKG_VERSION"),
    })
    .await
}

fn binary_flavor_to_backend(flavor: Option<BinaryFlavor>) -> Option<backend::BinaryFlavor> {
    flavor.map(|flavor| match flavor {
        BinaryFlavor::Cpu => backend::BinaryFlavor::Cpu,
        BinaryFlavor::Cuda => backend::BinaryFlavor::Cuda,
        BinaryFlavor::Rocm => backend::BinaryFlavor::Rocm,
        BinaryFlavor::Vulkan => backend::BinaryFlavor::Vulkan,
        BinaryFlavor::Metal => backend::BinaryFlavor::Metal,
    })
}

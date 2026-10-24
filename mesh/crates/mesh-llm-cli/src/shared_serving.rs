//! Explicit opt-in shared-model serving; ordinary `serve` is unchanged.

use std::path::PathBuf;

use clap::Args;

#[derive(Clone, Debug, Args)]
pub struct SharedServingArgs {
    /// One complete local GGUF; never a split or automatically elected model.
    #[arg(long)]
    pub model_path: PathBuf,
    /// Model ID offered independently to every attached mesh.
    #[arg(long)]
    pub model_id: String,
    /// Isolated host state directory (must not be used by another process).
    #[arg(long)]
    pub state_dir: PathBuf,
    /// Independent membership NAME=TOKEN_FILE; repeat for each mesh.
    #[arg(long, required = true, value_parser = parse_named_path)]
    pub membership: Vec<NamedPath>,
    /// Per-membership trust store NAME=FILE, using the existing trust-store format.
    #[arg(long, value_parser = parse_named_path)]
    pub membership_trust: Vec<NamedPath>,
    /// Owner keystore used to sign each distinct endpoint's attestation.
    #[arg(long)]
    pub attestation_key: Option<PathBuf>,
    /// Total simultaneous native generations, shared rather than per mesh.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..=64))]
    pub concurrency: u16,
    /// Total waiting requests, excluding admitted execution.
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u16).range(0..=4096))]
    pub queue_capacity: u16,
    /// Maximum waiting requests in any one membership.
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u16).range(0..=4096))]
    pub membership_queue_capacity: u16,
    /// Native context window per generation.
    #[arg(long, default_value_t = 4096, value_parser = clap::value_parser!(u32).range(512..))]
    pub context_size: u32,
    /// GPU layers; zero selects CPU, -1 offloads all supported layers.
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub gpu_layers: i32,
    /// Loopback-only authenticated management API port; zero chooses a free port.
    #[arg(long, default_value_t = 0)]
    pub management_port: u16,
    /// Disable relay and public STUN for explicitly local/LAN memberships.
    #[arg(long)]
    pub lan_only: bool,
    /// Bind every membership's QUIC endpoint to this local interface address.
    #[arg(long)]
    pub bind_ip: Option<std::net::IpAddr>,
}

#[derive(Clone, Debug)]
pub struct NamedPath {
    pub name: String,
    pub path: PathBuf,
}

fn parse_named_path(value: &str) -> Result<NamedPath, String> {
    let (name, path) = value
        .split_once('=')
        .ok_or_else(|| "expected NAME=FILE".to_owned())?;
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
    {
        return Err("membership name must be 1–64 ASCII letters, digits, '-' or '_'".to_owned());
    }
    if path.is_empty() {
        return Err("membership file path must not be empty".to_owned());
    }
    Ok(NamedPath {
        name: name.to_owned(),
        path: path.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::Parser;

    #[test]
    fn shared_serving_requires_explicit_model_memberships_and_state() {
        assert!(Cli::try_parse_from(["mesh-llm", "serve-shared"]).is_err());
        let cli = Cli::try_parse_from([
            "mesh-llm",
            "serve-shared",
            "--model-path",
            "model.gguf",
            "--model-id",
            "shared",
            "--state-dir",
            "state",
            "--membership",
            "work=work.token",
            "--membership",
            "friends=friends.token",
            "--lan-only",
        ])
        .unwrap();
        let Some(Command::ServeShared(args)) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(args.membership.len(), 2);
        assert_eq!(args.concurrency, 1);
        assert!(args.lan_only);
    }

    #[test]
    fn membership_names_cannot_escape_their_state_directory() {
        for value in ["../work=x", "work/other=x", "=x", "work=", "work", "..=x"] {
            assert!(parse_named_path(value).is_err(), "accepted {value}");
        }
        assert_eq!(
            parse_named_path("work=/tmp/a=b").unwrap().path,
            PathBuf::from("/tmp/a=b")
        );
    }
}

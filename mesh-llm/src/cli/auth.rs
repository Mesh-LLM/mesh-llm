use anyhow::{anyhow, Result};
use clap::Subcommand;
use std::path::PathBuf;

#[derive(Debug, Clone, Subcommand)]
pub enum AuthCommand {
    /// Derive and store an owner key from a passphrase
    Init {
        /// Provide passphrase non-interactively (WARNING: visible in process list and shell history)
        #[arg(long)]
        passphrase: Option<String>,

        /// Overwrite existing owner key without prompting
        #[arg(long)]
        force: bool,

        /// Path to owner key file (default: ~/.mesh-llm/owner-key)
        #[arg(long)]
        owner_key: Option<PathBuf>,
    },
    /// Display the current owner key fingerprint
    Status {
        /// Path to owner key file (default: ~/.mesh-llm/owner-key)
        #[arg(long)]
        owner_key: Option<PathBuf>,
    },
}

pub async fn run_auth_init(
    passphrase: Option<String>,
    force: bool,
    owner_key: Option<PathBuf>,
) -> Result<()> {
    let key_path = match owner_key {
        Some(p) => p,
        None => {
            let home = dirs::home_dir()
                .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
            home.join(".mesh-llm").join("owner-key")
        }
    };

    if key_path.exists() {
        if !force {
            return Err(anyhow!(
                "Owner key already exists at {}. Use --force to overwrite.",
                key_path.display()
            ));
        }
        eprintln!(
            "Warning: Overwriting existing owner key. Running nodes must be restarted to use the new key."
        );
    }

    let raw_passphrase = {
        use zeroize::Zeroizing;
        Zeroizing::new(match passphrase {
            Some(p) => p,
            None => {
                return Err(anyhow!(
                    "No passphrase provided. Use --passphrase flag for non-interactive use."
                ));
            }
        })
    };

    let key_bytes = {
        use zeroize::Zeroizing;
        Zeroizing::new(crate::auth::derive_owner_key(&raw_passphrase)?)
    };

    let hex_content = {
        use zeroize::Zeroizing;
        Zeroizing::new(crate::auth::format_key_hex(&*key_bytes))
    };

    if let Some(parent) = key_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    tokio::fs::write(&key_path, hex_content.as_bytes()).await?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&key_path, perms)?;
    }

    let fingerprint = crate::identity::owner_fingerprint_from_key_material(*key_bytes);
    println!("Owner key initialized.");
    println!("Fingerprint: {}", fingerprint);
    println!("Key stored at: {}", key_path.display());
    println!();
    println!("Run this same command with the same passphrase on all mesh nodes.");

    Ok(())
}

pub async fn dispatch_auth_command(command: AuthCommand) -> Result<()> {
    match command {
        AuthCommand::Init {
            passphrase,
            force,
            owner_key,
        } => {
            let resolved_passphrase = match passphrase {
                Some(p) => Some(p),
                None => {
                    use std::io::IsTerminal;
                    if std::io::stdin().is_terminal() {
                        use zeroize::Zeroizing;
                        println!("Enter passphrase (2+ words, e.g. \"river cloud brick\"):");
                        let input = Zeroizing::new(rpassword::read_password()?);
                        if input.trim().is_empty() {
                            return Err(anyhow!(
                                "No passphrase entered. Use --passphrase flag for non-interactive use."
                            ));
                        }
                        Some(input.as_str().to_owned())
                    } else {
                        return Err(anyhow!(
                            "No passphrase provided. Use --passphrase flag for non-interactive use."
                        ));
                    }
                }
            };

            run_auth_init(resolved_passphrase, force, owner_key).await?;
            Ok(())
        }
        AuthCommand::Status { owner_key } => {
            run_auth_status(owner_key).await?;
            Ok(())
        }
    }
}

pub async fn run_auth_status(owner_key: Option<PathBuf>) -> Result<()> {
    let key_path = match owner_key {
        Some(p) => p,
        None => {
            let home = dirs::home_dir()
                .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
            home.join(".mesh-llm").join("owner-key")
        }
    };

    if !key_path.exists() {
        return Err(anyhow!(
            "No owner key configured at {}. Run `mesh-llm auth init` to create one.",
            key_path.display()
        ));
    }

    let contents = tokio::fs::read_to_string(&key_path).await?;
    let raw = contents.trim_end_matches(&['\r', '\n'][..]);
    if raw.is_empty() {
        return Err(anyhow!(
            "Owner key file is empty at {}. Run `mesh-llm auth init` to create one.",
            key_path.display()
        ));
    }
    let bytes = hex::decode(raw)
        .map_err(|_| anyhow!("Owner key at {} is not valid hex", key_path.display()))?;
    if bytes.len() != 32 {
        return Err(anyhow!(
            "Owner key at {} has invalid length: expected 32 bytes, got {}",
            key_path.display(),
            bytes.len()
        ));
    }
    let mut key_bytes: [u8; 32] = bytes.try_into().unwrap();

    let fingerprint = crate::identity::owner_fingerprint_from_key_material(key_bytes);
    println!("Owner key: {}", key_path.display());
    println!("Fingerprint: {}", fingerprint);

    // Query mesh peer status (optional — graceful failure if mesh not running)
    let client_result = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build();

    let mesh_resp = match client_result {
        Ok(client) => client
            .get("http://localhost:3131/api/status")
            .send()
            .await
            .ok(),
        Err(_) => None,
    };

    match mesh_resp {
        Some(resp) => match resp.json::<serde_json::Value>().await {
            Ok(json) => {
                if let Some(peers) = json["peers"].as_array() {
                    let matched = peers
                        .iter()
                        .filter(|p| {
                            p["owner_fingerprint"].as_str() == Some(fingerprint.as_str())
                                && p["owner_fingerprint_verified"].as_bool().unwrap_or(false)
                        })
                        .count();
                    let mismatched = peers.len() - matched;
                    println!(
                        "Mesh peers: {} matched, {} with different fingerprint",
                        matched, mismatched
                    );
                } else {
                    println!("Mesh not running — peer status unavailable");
                }
            }
            Err(_) => {
                println!("Mesh not running — peer status unavailable");
            }
        },
        None => {
            println!("Mesh not running — peer status unavailable");
        }
    }

    // Zeroize key material before returning
    {
        use zeroize::Zeroize;
        key_bytes.zeroize();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::AuthCommand;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).expect("parse failed")
    }

    #[test]
    fn test_auth_init_parses() {
        let cli = parse(&["mesh-llm", "auth", "init"]);
        assert!(matches!(
            cli.command,
            Some(Command::Auth {
                command: AuthCommand::Init { .. }
            })
        ));
    }

    #[test]
    fn test_auth_init_force_flag() {
        let cli = parse(&["mesh-llm", "auth", "init", "--force"]);
        if let Some(Command::Auth {
            command: AuthCommand::Init { force, .. },
        }) = cli.command
        {
            assert!(force);
        } else {
            panic!("wrong command");
        }
    }

    #[test]
    fn test_auth_init_passphrase_flag() {
        let cli = parse(&[
            "mesh-llm",
            "auth",
            "init",
            "--passphrase",
            "fire lake mountain",
        ]);
        if let Some(Command::Auth {
            command: AuthCommand::Init { passphrase, .. },
        }) = cli.command
        {
            assert_eq!(passphrase.unwrap(), "fire lake mountain");
        } else {
            panic!("wrong command");
        }
    }

    #[test]
    fn test_auth_status_parses() {
        let cli = parse(&["mesh-llm", "auth", "status"]);
        assert!(matches!(
            cli.command,
            Some(Command::Auth {
                command: AuthCommand::Status { .. }
            })
        ));
    }

    #[test]
    fn test_auth_init_owner_key_flag() {
        let cli = parse(&["mesh-llm", "auth", "init", "--owner-key", "/tmp/test-key"]);
        if let Some(Command::Auth {
            command: AuthCommand::Init { owner_key, .. },
        }) = cli.command
        {
            assert_eq!(owner_key.unwrap().to_str().unwrap(), "/tmp/test-key");
        } else {
            panic!("wrong command");
        }
    }
}

#[cfg(test)]
mod init_tests {
    use std::fs;
    use tempfile::TempDir;

    fn temp_key_path() -> (TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test-owner-key");
        (dir, path)
    }

    #[tokio::test]
    async fn test_init_creates_key_file() {
        let (_dir, path) = temp_key_path();
        let result = super::run_auth_init(
            Some("fire lake mountain river cloud forest stone".to_string()),
            false,
            Some(path.clone()),
        )
        .await;
        assert!(result.is_ok(), "auth init should succeed: {:?}", result);
        assert!(path.exists(), "key file should be created");
        let content = fs::read_to_string(&path).expect("read file");
        assert_eq!(content.len(), 64, "key file should be 64 hex chars");
        assert!(
            content.chars().all(|c| c.is_ascii_hexdigit()),
            "content should be valid hex"
        );
    }

    #[tokio::test]
    async fn test_init_refuses_overwrite_without_force() {
        let (_dir, path) = temp_key_path();
        fs::write(&path, "a".repeat(64)).expect("write initial");
        let result = super::run_auth_init(
            Some("fire lake mountain river cloud forest stone".to_string()),
            false,
            Some(path.clone()),
        )
        .await;
        assert!(result.is_err(), "should refuse overwrite without force");
        let content = fs::read_to_string(&path).expect("read file");
        assert_eq!(content, "a".repeat(64), "file should be unchanged");
    }

    #[tokio::test]
    async fn test_init_force_overwrites() {
        let (_dir, path) = temp_key_path();
        fs::write(&path, "a".repeat(64)).expect("write initial");
        let result = super::run_auth_init(
            Some("different words for a new key here".to_string()),
            true,
            Some(path.clone()),
        )
        .await;
        assert!(result.is_ok(), "force overwrite should succeed");
        let content = fs::read_to_string(&path).expect("read file");
        assert_ne!(content, "a".repeat(64), "file should be overwritten");
        assert_eq!(content.len(), 64, "new content should be 64 hex chars");
    }

    #[tokio::test]
    async fn test_init_rejects_empty_passphrase() {
        let (_dir, path) = temp_key_path();
        let result = super::run_auth_init(Some("".to_string()), false, Some(path.clone())).await;
        assert!(result.is_err(), "empty passphrase should be rejected");
        assert!(!path.exists(), "no file should be created");
    }

    #[tokio::test]
    async fn test_init_rejects_single_word() {
        let (_dir, path) = temp_key_path();
        let result =
            super::run_auth_init(Some("singleword".to_string()), false, Some(path.clone())).await;
        assert!(result.is_err(), "single word passphrase should be rejected");
        assert!(!path.exists(), "no file should be created");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_init_sets_permissions_600() {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, path) = temp_key_path();
        let result = super::run_auth_init(
            Some("fire lake mountain river cloud forest stone".to_string()),
            false,
            Some(path.clone()),
        )
        .await;
        assert!(result.is_ok());
        let metadata = fs::metadata(&path).expect("metadata");
        let mode = metadata.permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "permissions should be 0o600");
    }
}

#[cfg(test)]
mod status_tests {
    use tempfile::TempDir;

    fn temp_key_path() -> (TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test-owner-key");
        (dir, path)
    }

    #[tokio::test]
    async fn test_status_valid_key_file() {
        let (_dir, path) = temp_key_path();
        super::run_auth_init(
            Some("fire lake mountain river cloud forest stone".to_string()),
            false,
            Some(path.clone()),
        )
        .await
        .expect("init should succeed");

        let result = super::run_auth_status(Some(path.clone())).await;
        assert!(result.is_ok(), "auth status should succeed: {:?}", result);
    }

    #[tokio::test]
    async fn test_status_missing_key_file() {
        let (_dir, path) = temp_key_path();
        let result = super::run_auth_status(Some(path.clone())).await;
        assert!(result.is_err(), "should return error for missing key file");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("No owner key configured"),
            "error should mention missing key: {}",
            msg
        );
        assert!(
            msg.contains("mesh-llm auth init"),
            "error should suggest `auth init`: {}",
            msg
        );
    }

    #[test]
    fn test_status_fingerprint_format() {
        let key_bytes = [42u8; 32];
        let fingerprint = crate::identity::owner_fingerprint_from_key_material(key_bytes);
        assert_eq!(fingerprint.len(), 64, "fingerprint must be 64 chars");
        assert!(
            fingerprint
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
            "fingerprint must be lowercase hex: {}",
            fingerprint
        );
    }

    #[tokio::test]
    async fn test_status_offline_mesh_exits_ok() {
        let (_dir, path) = temp_key_path();
        super::run_auth_init(
            Some("fire lake mountain river cloud forest stone".to_string()),
            false,
            Some(path.clone()),
        )
        .await
        .expect("init should succeed");

        let result = super::run_auth_status(Some(path.clone())).await;
        assert!(
            result.is_ok(),
            "status must succeed even when mesh is not running: {:?}",
            result
        );
    }
}

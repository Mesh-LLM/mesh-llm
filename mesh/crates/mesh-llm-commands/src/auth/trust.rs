//! `mesh-llm auth trust` subcommands, the trust store enforcement notes
//! shared with `auth status`, and the locked update every auth command that
//! writes the trust store goes through.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mesh_llm_cli::{TrustCommand, shell};
use mesh_llm_events::audit::{audit_events, emit_audit};
use mesh_llm_identity::{TrustPolicy, TrustStore, default_trust_store_path, save_trust_store};

use super::{cli_trust_policy_to_identity, load_effective_trust_store, resolve_trust_store_path};

pub(super) fn run_trust_command(command: &TrustCommand) -> Result<()> {
    match command {
        TrustCommand::Add {
            owner_id,
            label,
            trust_store,
        } => run_add(owner_id, label.clone(), trust_store.clone()),
        TrustCommand::Remove {
            owner_id,
            trust_store,
        } => run_remove(owner_id, trust_store.clone()),
        TrustCommand::List { trust_store } => run_list(trust_store.clone()),
        TrustCommand::Policy {
            policy,
            trust_store,
        } => run_set_policy(cli_trust_policy_to_identity(*policy), trust_store.clone()),
    }
}

fn run_add(owner_id: &str, label: Option<String>, trust_store: Option<PathBuf>) -> Result<()> {
    let trust_store_path = resolve_trust_store_path(trust_store)?;
    let store = update_trust_store(&trust_store_path, |store| {
        store.add_trusted_owner(owner_id.to_owned(), label);
        true
    })?;
    let mut err = mesh_llm_events::console_err();
    writeln!(
        err,
        "Trusted owner {owner_id} in {}",
        trust_store_path.display()
    )?;
    write_enforcement_notes(&mut err, &store, &trust_store_path)?;
    Ok(())
}

fn run_remove(owner_id: &str, trust_store: Option<PathBuf>) -> Result<()> {
    let trust_store_path = resolve_trust_store_path(trust_store)?;
    let mut removed = false;
    update_trust_store(&trust_store_path, |store| {
        removed = store.remove_trusted_owner(owner_id);
        removed
    })?;
    let mut err = mesh_llm_events::console_err();
    if removed {
        writeln!(
            err,
            "Removed trusted owner {owner_id} from {}",
            trust_store_path.display()
        )?;
    } else {
        writeln!(err, "Trusted owner {owner_id} was not present.")?;
    }
    Ok(())
}

fn run_list(trust_store: Option<PathBuf>) -> Result<()> {
    let trust_store_path = resolve_trust_store_path(trust_store)?;
    let store = load_effective_trust_store(&trust_store_path)?;
    let mut err = mesh_llm_events::console_err();
    writeln!(err, "Trust store:     {}", trust_store_path.display())?;
    writeln!(err, "Policy:          {:?}", store.policy)?;
    write_enforcement_notes(&mut err, &store, &trust_store_path)?;
    write_trust_store_entries(&mut err, &store)?;
    Ok(())
}

fn write_trust_store_entries(err: &mut impl Write, store: &TrustStore) -> Result<()> {
    writeln!(err)?;
    writeln!(err, "Trusted owners:")?;
    if store.trusted_owners.is_empty() {
        writeln!(err, "- none")?;
    } else {
        for owner in &store.trusted_owners {
            match owner.label.as_deref() {
                Some(label) => writeln!(err, "- {} ({label})", owner.owner_id)?,
                None => writeln!(err, "- {}", owner.owner_id)?,
            }
        }
    }
    writeln!(err)?;
    writeln!(err, "Revoked owners:")?;
    if store.revoked_owners.is_empty() {
        writeln!(err, "- none")?;
    } else {
        for owner in &store.revoked_owners {
            match owner.reason.as_deref() {
                Some(reason) => writeln!(err, "- {} ({reason})", owner.owner_id)?,
                None => writeln!(err, "- {}", owner.owner_id)?,
            }
        }
    }
    writeln!(err)?;
    writeln!(err, "Revoked node certs:")?;
    if store.revoked_node_certs.is_empty() {
        writeln!(err, "- none")?;
    } else {
        for cert in &store.revoked_node_certs {
            match cert.reason.as_deref() {
                Some(reason) => writeln!(err, "- {} ({reason})", cert.cert_id)?,
                None => writeln!(err, "- {}", cert.cert_id)?,
            }
        }
    }
    writeln!(err)?;
    writeln!(err, "Revoked node IDs:")?;
    if store.revoked_node_ids.is_empty() {
        writeln!(err, "- none")?;
    } else {
        for node in &store.revoked_node_ids {
            match node.reason.as_deref() {
                Some(reason) => writeln!(err, "- {} ({reason})", node.node_endpoint_id)?,
                None => writeln!(err, "- {}", node.node_endpoint_id)?,
            }
        }
    }
    Ok(())
}

/// Run `mesh-llm auth trust policy <policy>`.
pub(super) fn run_set_policy(policy: TrustPolicy, trust_store: Option<PathBuf>) -> Result<()> {
    let trust_store_path = resolve_trust_store_path(trust_store)?;
    let mut previous = policy;
    let store = update_trust_store(&trust_store_path, |store| {
        previous = std::mem::replace(&mut store.policy, policy);
        true
    })?;

    emit_audit(audit_events::admin_action(
        Some("system".to_string()),
        "auth_trust_policy",
        Some(policy_name(policy)),
        true,
        None,
    ))
    .context("Trust policy was saved, but its audit record could not be written")?;

    let mut err = mesh_llm_events::console_err();
    writeln!(
        err,
        "Trust policy set to {} (was {}) in {}",
        policy_name(policy),
        policy_name(previous),
        trust_store_path.display()
    )?;
    writeln!(err, "{}", restart_note(&trust_store_path))?;
    write_enforcement_notes(&mut err, &store, &trust_store_path)?;
    Ok(())
}

/// Load the trust store at `path`, let `update` change it, and save it when
/// `update` returns true, all while holding an exclusive lock on a file
/// beside the store. Every auth command that writes the store goes through
/// here, so two of them running at once take turns instead of one saving
/// over the other's change.
pub(super) fn update_trust_store(
    path: &Path,
    update: impl FnOnce(&mut TrustStore) -> bool,
) -> Result<TrustStore> {
    let _lock = lock_trust_store(path)?;
    let mut store = load_effective_trust_store(path)?;
    if update(&mut store) {
        save_trust_store(path, &store)?;
    }
    Ok(store)
}

/// Lock `.<file name>.lock` beside the trust store, waiting while another
/// command holds it. Dropping the returned file releases the lock, and so
/// does the process exiting.
fn lock_trust_store(path: &Path) -> Result<File> {
    let file_name = path
        .file_name()
        .with_context(|| format!("Trust store path {} has no file name", path.display()))?;
    let lock_path = path.with_file_name(format!(".{}.lock", file_name.to_string_lossy()));
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("Failed to open trust store lock {}", lock_path.display()))?;
    lock.lock()
        .with_context(|| format!("Failed to lock {}", lock_path.display()))?;
    Ok(lock)
}

/// When nodes pick up a policy just stored at `trust_store_path`.
fn restart_note(trust_store_path: &Path) -> String {
    if is_default_trust_store(trust_store_path) {
        return "Nodes apply it at their next start, unless they run with --trust-policy."
            .to_owned();
    }
    let default = default_trust_store_path().map_or_else(
        |_| "the default trust store".to_owned(),
        |path| path.display().to_string(),
    );
    format!(
        "Default node startup reads {default}, not this file. This custom policy applies \
         at the next start only to a `serve --memberships-config` membership whose \
         `trust_store` names this file."
    )
}

/// Whether `path` is the trust store nodes load at startup.
fn is_default_trust_store(path: &Path) -> bool {
    let Ok(default) = default_trust_store_path() else {
        return false;
    };
    std::path::absolute(path).is_ok_and(|path| path == default)
}

/// Point out trust store entries that the stored policy does not enforce.
///
/// Trusted owners only gate admission under `allowlist`, and revocations
/// only under `require-owned` or `allowlist`; under any other policy they are
/// kept but have no effect.
pub(super) fn write_enforcement_notes(
    err: &mut impl Write,
    store: &TrustStore,
    trust_store_path: &Path,
) -> Result<()> {
    for note in enforcement_notes(store, trust_store_path) {
        writeln!(err, "{note}")?;
    }
    Ok(())
}

fn enforcement_notes(store: &TrustStore, trust_store_path: &Path) -> Vec<String> {
    let current = policy_name(store.policy);
    let has_trusted_owners = !store.trusted_owners.is_empty();
    let mut notes = Vec::new();
    if has_trusted_owners && store.policy != TrustPolicy::Allowlist {
        notes.push(format!(
            "Note: trusted owners are only enforced under the allowlist policy, and this trust \
             store's policy is {current}. To enforce them, run `{}`.",
            set_policy_command(TrustPolicy::Allowlist, trust_store_path)
        ));
    }
    let has_revocations = !store.revoked_owners.is_empty()
        || !store.revoked_node_certs.is_empty()
        || !store.revoked_node_ids.is_empty();
    if has_revocations && matches!(store.policy, TrustPolicy::Off | TrustPolicy::PreferOwned) {
        // allowlist enforces revocations too, and require-owned would leave
        // the trusted owners unenforced again.
        let enforcing = if has_trusted_owners {
            TrustPolicy::Allowlist
        } else {
            TrustPolicy::RequireOwned
        };
        notes.push(format!(
            "Note: revocations are only enforced under the require-owned or allowlist policy, \
             and this trust store's policy is {current}. To enforce them, run `{}`.",
            set_policy_command(enforcing, trust_store_path)
        ));
    }
    notes
}

fn set_policy_command(policy: TrustPolicy, trust_store_path: &Path) -> String {
    let mut command = format!("mesh-llm auth trust policy {}", policy_name(policy));
    if !is_default_trust_store(trust_store_path) {
        command.push_str(&format!(
            " --trust-store {}",
            shell::single_quote(&trust_store_path.to_string_lossy())
        ));
    }
    command
}

/// The policy as `--trust-policy` and `auth trust policy` spell it.
fn policy_name(policy: TrustPolicy) -> &'static str {
    match policy {
        TrustPolicy::Off => "off",
        TrustPolicy::PreferOwned => "prefer-owned",
        TrustPolicy::RequireOwned => "require-owned",
        TrustPolicy::Allowlist => "allowlist",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_trust_store_path(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mesh-llm-auth-trust-{label}-{}",
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("trusted-owners.json")
    }

    fn store_with_owner(policy: TrustPolicy) -> TrustStore {
        let mut store = TrustStore {
            policy,
            ..TrustStore::default()
        };
        store.add_trusted_owner("owner-a".to_owned(), None);
        store
    }

    #[test]
    fn set_policy_persists_and_keeps_entries() {
        let path = temporary_trust_store_path("set-policy");
        let mut store = store_with_owner(TrustPolicy::Off);
        store.revoke_owner("owner-b".to_owned(), Some("lost key".to_owned()));
        store.revoke_node_cert("cert-a".to_owned(), None);
        store.revoke_node_id("node-a".to_owned(), Some("retired".to_owned()));
        save_trust_store(&path, &store).unwrap();

        run_set_policy(TrustPolicy::Allowlist, Some(path.clone())).unwrap();

        let saved = load_effective_trust_store(&path).unwrap();
        assert_eq!(saved.policy, TrustPolicy::Allowlist);
        assert_eq!(saved.trusted_owners, store.trusted_owners);
        assert_eq!(saved.revoked_owners, store.revoked_owners);
        assert_eq!(saved.revoked_node_certs, store.revoked_node_certs);
        assert_eq!(saved.revoked_node_ids, store.revoked_node_ids);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn concurrent_updates_keep_every_change() {
        let path = temporary_trust_store_path("concurrent");
        let start = std::sync::Arc::new(std::sync::Barrier::new(8));
        let writers: Vec<_> = (0..8)
            .map(|index| {
                let path = path.clone();
                let start = start.clone();
                std::thread::spawn(move || {
                    start.wait();
                    update_trust_store(&path, |store| {
                        store.add_trusted_owner(format!("owner-{index}"), None);
                        true
                    })
                    .unwrap();
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }

        let saved = load_effective_trust_store(&path).unwrap();
        assert_eq!(saved.trusted_owners.len(), 8, "{:?}", saved.trusted_owners);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn trust_store_lock_is_shared_between_processes() {
        const CHILD_LOCK_PATH: &str = "MESH_LLM_TEST_TRUST_STORE_LOCK_PATH";
        if let Some(path) = std::env::var_os(CHILD_LOCK_PATH) {
            let lock = OpenOptions::new().write(true).open(path).unwrap();
            assert!(matches!(
                lock.try_lock(),
                Err(std::fs::TryLockError::WouldBlock)
            ));
            return;
        }

        let path = temporary_trust_store_path("process-lock");
        let lock = lock_trust_store(&path).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "auth::trust::tests::trust_store_lock_is_shared_between_processes",
            ])
            .env(
                CHILD_LOCK_PATH,
                path.with_file_name(".trusted-owners.json.lock"),
            )
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        drop(lock);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn set_policy_creates_a_missing_trust_store() {
        let path = temporary_trust_store_path("set-policy-new");

        run_set_policy(TrustPolicy::RequireOwned, Some(path.clone())).unwrap();

        let saved = load_effective_trust_store(&path).unwrap();
        assert_eq!(saved.policy, TrustPolicy::RequireOwned);
        assert!(saved.trusted_owners.is_empty());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn set_policy_save_failure_leaves_blocking_file_unchanged() {
        let path = temporary_trust_store_path("set-policy-failure");
        let blocking_file = path.with_file_name("blocker");
        std::fs::write(&blocking_file, b"not a directory").unwrap();

        let result = run_set_policy(
            TrustPolicy::Allowlist,
            Some(blocking_file.join("trusted-owners.json")),
        );

        assert!(result.is_err());
        assert_eq!(std::fs::read(&blocking_file).unwrap(), b"not a directory");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn trusted_owners_outside_allowlist_get_a_note() {
        let path = Path::new("custom/trusted-owners.json");
        for policy in [
            TrustPolicy::Off,
            TrustPolicy::PreferOwned,
            TrustPolicy::RequireOwned,
        ] {
            let notes = enforcement_notes(&store_with_owner(policy), path);
            assert_eq!(notes.len(), 1, "{policy:?}: {notes:?}");
            assert!(notes[0].contains(&format!("policy is {}.", policy_name(policy))));
            assert!(notes[0].contains(
                "`mesh-llm auth trust policy allowlist --trust-store 'custom/trusted-owners.json'`"
            ));
        }
        assert!(enforcement_notes(&store_with_owner(TrustPolicy::Allowlist), path).is_empty());
    }

    #[test]
    fn hint_quotes_a_custom_trust_store_path() {
        let path = Path::new("owner's trust/$HOME; store.json");
        let notes = enforcement_notes(&store_with_owner(TrustPolicy::Off), path);
        assert!(
            notes[0].contains(r#"--trust-store 'owner'"'"'s trust/$HOME; store.json'`"#),
            "{notes:?}"
        );
    }

    #[test]
    fn revocations_outside_owned_policies_get_a_note() {
        let path = Path::new("custom/trusted-owners.json");
        let mut store = TrustStore::default();
        store.revoke_node_cert("cert-a".to_owned(), None);

        for policy in [TrustPolicy::Off, TrustPolicy::PreferOwned] {
            store.policy = policy;
            let notes = enforcement_notes(&store, path);
            assert_eq!(notes.len(), 1, "{policy:?}: {notes:?}");
            assert!(notes[0].contains("revocations are only enforced"));
            assert!(notes[0].contains("mesh-llm auth trust policy require-owned"));
        }
        for policy in [TrustPolicy::RequireOwned, TrustPolicy::Allowlist] {
            store.policy = policy;
            assert!(enforcement_notes(&store, path).is_empty(), "{policy:?}");
        }
    }

    #[test]
    fn revocation_note_suggests_allowlist_when_owners_are_trusted() {
        let path = Path::new("custom/trusted-owners.json");
        let mut store = store_with_owner(TrustPolicy::Off);
        store.revoke_node_cert("cert-a".to_owned(), None);

        let notes = enforcement_notes(&store, path);

        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(notes[1].contains("revocations are only enforced"));
        assert!(notes[1].contains("mesh-llm auth trust policy allowlist"));
    }

    #[test]
    fn restart_note_says_nodes_do_not_read_a_custom_trust_store() {
        let note = restart_note(Path::new("custom/trusted-owners.json"));
        assert!(note.contains("not this file"), "{note}");
        assert!(note.contains("serve --memberships-config"), "{note}");
    }

    #[test]
    fn restart_note_for_the_default_trust_store_names_the_next_start() {
        let Ok(default_path) = default_trust_store_path() else {
            return;
        };
        assert_eq!(
            restart_note(&default_path),
            "Nodes apply it at their next start, unless they run with --trust-policy."
        );
    }

    #[test]
    fn an_empty_trust_store_gets_no_notes() {
        let path = Path::new("custom/trusted-owners.json");
        assert!(enforcement_notes(&TrustStore::default(), path).is_empty());
    }

    #[test]
    fn hint_omits_the_default_trust_store_path() {
        let Ok(default_path) = default_trust_store_path() else {
            return;
        };
        let notes = enforcement_notes(&store_with_owner(TrustPolicy::Off), &default_path);
        assert!(notes[0].contains("`mesh-llm auth trust policy allowlist`"));
    }

    #[test]
    fn policy_names_match_the_cli_spelling() {
        use clap::ValueEnum;

        for cli_policy in mesh_llm_cli::TrustPolicy::value_variants() {
            assert_eq!(
                policy_name(cli_trust_policy_to_identity(*cli_policy)),
                cli_policy.as_str()
            );
        }
    }
}

//! The release check that finishes before startup continues.

use std::future::Future;

use mesh_llm_events::{OutputEvent, emit_event};

use super::RuntimeOptions;
use crate::plugin::MeshConfig;
use crate::system::autoupdate::{self, UpdateNotice};

/// Shows a notice when a newer release is out.
pub(super) async fn announce_newer_release(
    options: &RuntimeOptions,
    checked_updates: bool,
    config: &MeshConfig,
) {
    let skip = skip_release_check(options, checked_updates, config);
    let notice =
        newer_release_notice(skip, || autoupdate::check_for_update(crate::BUILD_VERSION)).await;
    if let Some(notice) = notice {
        let _ = emit_event(OutputEvent::Info {
            message: notice.message,
            context: None,
        });
    }
}

/// Skips when `--auto-update` already checked, the command is `update` or
/// prints machine output, or `updates.check` turns the check off.
fn skip_release_check(
    options: &RuntimeOptions,
    checked_updates: bool,
    config: &MeshConfig,
) -> bool {
    checked_updates
        || options.command_is_update
        || options.command_uses_machine_output
        || !config.updates.check_enabled()
}

async fn newer_release_notice<F, Fut>(skip: bool, check: F) -> Option<UpdateNotice>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Option<UpdateNotice>>,
{
    if skip { None } else { check().await }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_check_false_skips_the_release_check() {
        let options = RuntimeOptions::default();
        let mut config = MeshConfig::default();
        assert!(!skip_release_check(&options, false, &config));

        config.updates.check = Some(false);
        assert!(skip_release_check(&options, false, &config));

        config.updates.check_override = Some(true);
        assert!(!skip_release_check(&options, false, &config));
    }

    #[test]
    fn update_commands_and_machine_output_skip_the_release_check() {
        let config = MeshConfig::default();
        assert!(skip_release_check(
            &RuntimeOptions::default(),
            true,
            &config
        ));
        let options = RuntimeOptions {
            command_is_update: true,
            ..RuntimeOptions::default()
        };
        assert!(skip_release_check(&options, false, &config));
        let options = RuntimeOptions {
            command_uses_machine_output: true,
            ..RuntimeOptions::default()
        };
        assert!(skip_release_check(&options, false, &config));
    }

    #[tokio::test]
    async fn a_skipped_check_sends_no_request() {
        let notice = newer_release_notice(true, || async {
            panic!("the release check must not run");
        })
        .await;
        assert!(notice.is_none());
    }

    #[tokio::test]
    async fn a_check_that_runs_returns_its_notice() {
        let notice = newer_release_notice(false, || async {
            Some(UpdateNotice {
                message: "new".to_string(),
            })
        })
        .await;
        assert_eq!(notice.map(|notice| notice.message).as_deref(), Some("new"));
    }
}

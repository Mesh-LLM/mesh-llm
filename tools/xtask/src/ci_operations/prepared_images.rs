//! Prepared-automation consumers of `check`: on Linux, `prepare-automation`
//! verifies the runner as one fixed image, so every catalogued job that runs it
//! must be bound to that environment and backend. The binding census alone
//! accepts any matching workflow and catalog pair, which let the nightly
//! stability job stay on `public web` and stop at that verification each run.

use crate::ci_operations::json_access::{Outcome, is_str, item, item_by, require};
use crate::ci_operations::workflow_census::{Workflows, job, read_text};
use crate::ci_plan::document::Json;
use crate::prepared_input::value_format::display;
use std::path::Path;

const ACTION: &str = ".github/actions/prepare-automation/action.yml";
const USES: &str = "./.github/actions/prepare-automation";

/// The `<environment> <backend>` pair the action's image profile verifies.
fn verified_image(action: &str) -> Outcome<(String, String)> {
    let pairs: Vec<Vec<&str>> = action
        .lines()
        .filter_map(|line| line.trim().strip_prefix("run: verify-runner-image "))
        .map(|rest| rest.split_whitespace().collect())
        .collect();
    require(pairs.len() == 1 && pairs[0].len() >= 2, || {
        format!("{ACTION}: expected exactly one verify-runner-image step")
    })?;
    Ok((pairs[0][0].to_owned(), pairs[0][1].to_owned()))
}

/// Whether a job runs the action in its image profile, the one that verifies
/// the runner image; another `runner-profile` skips the verification.
fn prepares_in_image_profile(body: &str) -> bool {
    body.contains(USES)
        && !body.lines().any(|line| {
            line.trim()
                .strip_prefix("runner-profile:")
                .is_some_and(|profile| profile.trim() != "image")
        })
}

pub(crate) fn check_prepared_images(
    catalog: &Json,
    workflows: &Workflows,
    root: &Path,
) -> Outcome<()> {
    let (environment, backend) = verified_image(&read_text(&root.join(ACTION))?)?;
    let (images, roles) = (item(catalog, "images")?, item(catalog, "consumer_roles")?);
    for (role_id, role) in roles.as_object().unwrap_or_default() {
        let image_id = item(role, "image_id")?;
        let image = item_by(images, image_id)?;
        for binding in item(role, "bindings")?.as_array().unwrap_or_default() {
            let workflow = display(Some(item(binding, "workflow")?));
            let job_id = display(Some(item(binding, "job")?));
            let body = workflows.get(&workflow).and_then(|jobs| job(jobs, &job_id));
            if !body.is_some_and(prepares_in_image_profile) {
                continue;
            }
            require(
                is_str(item(image, "environment")?, &environment)
                    && is_str(item(image, "backend")?, &backend),
                || {
                    format!(
                        "{workflow}:{job_id} ({role_id}): prepare-automation verifies `{environment} {backend}`, but the job is bound to {}",
                        display(Some(image_id))
                    )
                },
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_operations::evidence_input::decode;
    use std::path::PathBuf;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn catalog(text: &str) -> Json {
        decode(text.as_bytes()).unwrap()
    }

    fn nightly(body: &str) -> Workflows {
        Workflows::from([(
            "nightly-stability-run.yml".to_owned(),
            vec![("stability".to_owned(), body.to_owned())],
        )])
    }

    const PREPARED: &str =
        "    steps:\n      - id: automation\n        uses: ./.github/actions/prepare-automation\n";

    #[test]
    fn the_action_verifies_one_public_cpu_image() {
        let action = read_text(&root().join(ACTION)).unwrap();
        assert_eq!(
            verified_image(&action).unwrap(),
            ("public".to_owned(), "cpu".to_owned())
        );
        assert!(verified_image("steps: []\n").is_err());
    }

    #[test]
    fn a_prepared_job_bound_to_the_web_image_is_rejected() {
        let text = std::fs::read_to_string(root().join("ci/runner-images.json")).unwrap();
        check_prepared_images(&catalog(&text), &nightly(PREPARED), &root()).unwrap();
        // The pair the nightly job shipped with before #2757.
        let start = text.find("\"nightly-stability\"").unwrap();
        let web = format!(
            "{}{}",
            &text[..start],
            text[start..].replacen("\"public-cpu\"", "\"public-web\"", 1)
        );
        let error = check_prepared_images(&catalog(&web), &nightly(PREPARED), &root()).unwrap_err();
        assert_eq!(
            error,
            "nightly-stability-run.yml:stability (nightly-stability): prepare-automation verifies `public cpu`, but the job is bound to public-web"
        );
    }

    #[test]
    fn only_the_image_profile_is_tied_to_the_verified_image() {
        let text = std::fs::read_to_string(root().join("ci/runner-images.json")).unwrap();
        let start = text.find("\"nightly-stability\"").unwrap();
        let web = catalog(&format!(
            "{}{}",
            &text[..start],
            text[start..].replacen("\"public-cpu\"", "\"public-web\"", 1)
        ));
        let hosted = format!("{PREPARED}        with:\n          runner-profile: hosted-bare\n");
        check_prepared_images(&web, &nightly(&hosted), &root()).unwrap();
        check_prepared_images(&web, &nightly("    steps: []\n"), &root()).unwrap();
        let image = format!("{PREPARED}        with:\n          runner-profile: image\n");
        assert!(check_prepared_images(&web, &nightly(&image), &root()).is_err());
    }
}

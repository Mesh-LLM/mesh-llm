//! Unique preview identities, without editing the workspace version.
use crate::repository::check_report::CheckReport;

pub(super) fn run(args: &[String]) -> CheckReport {
    match version(env!("CARGO_PKG_VERSION"), args) {
        Ok(version) => CheckReport {
            stdout: format!("version={version}\n"),
            ..Default::default()
        },
        Err(error) => CheckReport {
            stderr: format!("{error}\n"),
            code: 1,
            ..Default::default()
        },
    }
}

fn version(base: &str, args: &[String]) -> Result<String, &'static str> {
    let [run, attempt] = args else {
        return Err("usage: release preview-version RUN_ID ATTEMPT");
    };
    for value in [run, attempt] {
        if value.is_empty()
            || value.starts_with('0')
            || value.len() > 20
            || !value.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err("preview identity must be a positive decimal of at most 20 digits");
        }
    }
    let core = base.split('-').next().ok_or("missing workspace version")?;
    Ok(format!("v{core}-preview.{run}.{attempt}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_identity_is_always_a_preview() {
        assert_eq!(
            version("0.78.0-rc.1", &["123".into(), "2".into()]).unwrap(),
            "v0.78.0-preview.123.2"
        );
    }

    #[test]
    fn reject_shell_and_semver_ambiguities() {
        for input in ["", "0", "01", "-1", "1\n", "$(id)", "1.2"] {
            assert!(version("0.78.0", &[input.into(), "1".into()]).is_err());
        }
        assert!(version("0.78.0", &[]).is_err());
    }
}

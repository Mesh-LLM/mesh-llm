use crate::{command::DynResult, repository::check_args::Grammar};
use std::time::Duration;
pub(super) const USAGE: &str = "automation decisions-smoke [--base-url HTTP_OR_HTTPS_ROOT] [--model ID] [--timeout SECONDS]; default http://127.0.0.1:9337, one overall 120-second deadline; HTTPS/non-IPv4 uses the existing bounded curl prerequisite";
pub(super) struct Options {
    pub base: String,
    pub model: Option<String>,
    pub timeout: Duration,
}
pub(super) fn valid_id(id: &str) -> bool {
    !id.trim().is_empty() && id.len() <= 4096 && !id.chars().any(char::is_control)
}
impl Options {
    pub(super) fn parse(args: &[String]) -> DynResult<Self> {
        const G: Grammar = Grammar {
            usage: USAGE,
            values: &["--base-url", "--model", "--timeout"],
            flags: &[],
        };
        let parsed = G
            .parse(args)
            .map_err(|_| "invalid Decisions smoke arguments")?;
        if !parsed.positionals.is_empty() || G.values.iter().any(|key| parsed.all(key).len() > 1) {
            return Err("Decisions smoke requires unique named options".into());
        }
        let model = parsed.last("--model").map(str::to_owned);
        if model.as_deref().is_some_and(|id| !valid_id(id)) {
            return Err("Decisions model must be bounded single-line text".into());
        }
        let url = url::Url::parse(parsed.last("--base-url").unwrap_or("http://127.0.0.1:9337"))
            .map_err(|_| "invalid Decisions base URL")?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !url.path().trim_end_matches('/').is_empty()
        {
            return Err("Decisions base URL requires a credential-free HTTP(S) root without query or fragment".into());
        }
        let seconds: u64 = parsed
            .last("--timeout")
            .unwrap_or("120")
            .parse()
            .map_err(|_| "invalid Decisions timeout")?;
        if !(1..=3600).contains(&seconds) {
            return Err("Decisions timeout must be 1..3600 seconds".into());
        }
        Ok(Self {
            base: url.as_str().trim_end_matches('/').to_owned(),
            model,
            timeout: Duration::from_secs(seconds),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn options_refuse_ambiguous_or_sensitive_inputs() {
        for args in [
            vec!["--base-url", "http://u:p@host"],
            vec!["--base-url", "http://host?token=s"],
            vec!["--base-url", "file:///tmp"],
            vec!["--timeout", "0"],
            vec!["--timeout", "3601"],
            vec!["--model", "bad\nname"],
            vec!["--model", "one", "--model", "two"],
        ] {
            assert!(
                Options::parse(&args.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err()
            );
        }
        assert_eq!(
            Options::parse(&[]).unwrap().timeout,
            Duration::from_secs(120)
        );
    }
}

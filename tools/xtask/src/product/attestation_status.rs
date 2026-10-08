use crate::repository::check_report::CheckReport;
use serde::Deserialize;
use std::io::Read;

#[derive(Deserialize)]
struct Inspection {
    status: Status,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Status {
    Valid,
    Invalid,
    Missing,
}

fn project(bytes: &[u8]) -> Result<&'static str, serde_json::Error> {
    let inspection: Inspection = serde_json::from_slice(bytes)?;
    Ok(match inspection.status {
        Status::Valid => "valid\n",
        Status::Invalid => "invalid\n",
        Status::Missing => "missing\n",
    })
}

pub(super) fn run(args: &[String]) -> CheckReport {
    let result = (|| {
        if !args.is_empty() {
            return Err("usage: product attestation-status < inspection.json".to_owned());
        }
        let mut bytes = Vec::new();
        std::io::stdin()
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        project(&bytes).map_err(|error| error.to_string())
    })();
    match result {
        Ok(status) => CheckReport::success(status.to_owned()),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn projects_inspection_verdict_without_changing_it() {
        for status in ["valid", "invalid", "missing"] {
            let input = format!(r#"{{"status":"{status}","error":null}}"#);
            assert_eq!(
                super::project(input.as_bytes()).unwrap(),
                format!("{status}\n")
            );
        }
    }

    #[test]
    fn rejects_unknown_or_missing_verdicts() {
        for input in [r#"{"status":"success"}"#, "{}", "null"] {
            assert!(super::project(input.as_bytes()).is_err());
        }
    }
}

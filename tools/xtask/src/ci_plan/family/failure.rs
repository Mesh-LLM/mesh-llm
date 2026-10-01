use crate::repository::check_report::CheckReport;

pub(super) enum Failure {
    Policy(String),
    Runtime(String),
    MissingOutput { stdout: String },
}

impl Failure {
    pub(super) fn io(error: std::io::Error) -> Self {
        Self::Runtime(error.to_string())
    }

    pub(super) fn report(self) -> CheckReport {
        match self {
            Self::Policy(reason) => CheckReport {
                stdout: String::new(),
                stderr: format!("family battery plan failed: {reason}\n"),
                code: 2,
            },
            Self::Runtime(reason) => CheckReport::failure(String::new(), format!("{reason}\n")),
            Self::MissingOutput { stdout } => CheckReport {
                stdout,
                stderr: "family battery plan failed: --github-output requires --output\n".into(),
                code: 2,
            },
        }
    }
}

impl From<String> for Failure {
    fn from(reason: String) -> Self {
        Self::Policy(reason)
    }
}

use crate::repository::check_report::CheckReport;

pub(super) struct Program {
    pub(super) name: &'static str,
    pub(super) positionals: &'static [&'static str],
}

impl Program {
    pub(super) fn parse<'a>(&self, args: &'a [String]) -> Result<Vec<&'a str>, CheckReport> {
        let mut values = Vec::with_capacity(self.positionals.len());
        let mut positional_only = false;
        for arg in args {
            match arg.as_str() {
                "--" if !positional_only => positional_only = true,
                "-h" | "--help" if !positional_only => {
                    return Err(CheckReport::success(format!(
                        "{}\nArguments: {}\nOptions: -h, --help\n",
                        self.usage(),
                        self.positionals.join(", ")
                    )));
                }
                value if !positional_only && value.starts_with('-') && value != "-" => {
                    return Err(self.error(&format!("unrecognized argument: {value}")));
                }
                value => values.push(value),
            }
        }
        if values.len() > self.positionals.len() {
            return Err(self.error("too many positional arguments"));
        }
        if let Some(missing) = self.positionals.get(values.len()..)
            && !missing.is_empty()
        {
            return Err(self.error(&format!("required arguments: {}", missing.join(", "))));
        }
        Ok(values)
    }

    fn usage(&self) -> String {
        format!("usage: {} [-h] {}\n", self.name, self.positionals.join(" "))
    }

    fn error(&self, message: &str) -> CheckReport {
        CheckReport {
            stdout: String::new(),
            stderr: format!("{}{}: error: {message}\n", self.usage(), self.name),
            code: 2,
        }
    }
}

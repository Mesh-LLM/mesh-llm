use std::path::PathBuf;

pub(super) const USAGE: &str = "cargo xtool ci-ops sccache-summary [--format text|json] [--minimum-hit-rate <0..1>] <evidence-path>...";
pub(super) struct Options {
    pub paths: Vec<PathBuf>,
    pub minimum: Option<f64>,
    pub json: bool,
}

pub(super) fn parse(arguments: &[String]) -> Result<Option<Options>, String> {
    if matches!(arguments, [flag] if flag == "--help" || flag == "-h") {
        return Ok(None);
    }
    let mut options = Options {
        paths: Vec::new(),
        minimum: None,
        json: false,
    };
    let mut format_seen = false;
    let mut positionals = false;
    let mut arguments = arguments.iter();
    while let Some(argument) = arguments.next() {
        if !positionals && argument == "--" {
            positionals = true;
        } else if !positionals && argument == "--format" {
            if format_seen {
                return Err("format may be supplied only once".into());
            }
            format_seen = true;
            options.json = match arguments.next().map(String::as_str) {
                Some("json") => true,
                Some("text") => false,
                _ => return Err("format requires text or json".into()),
            };
        } else if !positionals && argument == "--minimum-hit-rate" {
            if options.minimum.is_some() {
                return Err("minimum hit rate may be supplied only once".into());
            }
            let minimum: f64 = arguments
                .next()
                .ok_or("minimum hit rate requires a value")?
                .parse()
                .map_err(|_| "minimum hit rate must be a finite number between 0 and 1")?;
            if !minimum.is_finite() || !(0.0..=1.0).contains(&minimum) {
                return Err("minimum hit rate must be a finite number between 0 and 1".into());
            }
            options.minimum = Some(minimum);
        } else if !positionals && argument.starts_with('-') {
            return Err(format!("unknown option: {argument}"));
        } else if argument.is_empty() {
            return Err("evidence path must not be empty".into());
        } else {
            options.paths.push(argument.into());
        }
    }
    if options.paths.is_empty() {
        return Err("at least one evidence path is required".into());
    }
    Ok(Some(options))
}

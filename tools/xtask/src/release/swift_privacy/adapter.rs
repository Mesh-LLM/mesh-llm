use super::{
    adapter_args,
    adapter_error::{self, Error},
    embedding::{ValidatedTemplate, discover_embedded},
    native_lint::NativeLint,
};
use crate::{automation::command_interrupt::Interrupt, command::DynResult};
use std::io::Write;

pub(crate) fn run(arguments: &[String]) -> DynResult<()> {
    let mut stdout = crate::cli_output::stdout();
    if arguments == ["--help"] {
        writeln!(stdout, "{}", adapter_args::USAGE)?;
        stdout.flush()?;
        return Ok(());
    }
    let options = adapter_args::parse(arguments)?;
    let template = ValidatedTemplate::read(options.template)?;
    let cwd = std::env::current_dir()?;
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let native = NativeLint {
        options: &options,
        cwd: &cwd,
        cancellation: &cancellation,
    };
    let primary = verify(&template, &native, &mut stdout);
    let verified = adapter_error::finish(primary, interrupt.finish())?;
    if let Some((root, count)) = verified {
        writeln!(
            stdout,
            "verified {count} embedded privacy manifest file(s) in {}",
            root.display()
        )?;
        stdout.flush()?;
    }
    Ok(())
}

fn verify<'a>(
    template: &ValidatedTemplate<'_>,
    native: &NativeLint<'a>,
    stdout: &mut impl Write,
) -> Result<Option<(&'a std::path::Path, usize)>, Error> {
    let mut stderr = crate::cli_output::stderr();
    native.lint(template.path(), &mut stderr)?;
    writeln!(
        stdout,
        "verified Swift privacy manifest: {}",
        template.path().display()
    )?;
    stdout.flush()?;
    match native.options.xcframework {
        None => Ok(None),
        Some(root) => {
            let embedded = discover_embedded(root)?;
            for path in &embedded {
                let equal = template.compare(path)?;
                native.lint(equal, &mut stderr)?;
            }
            Ok(Some((root, embedded.len())))
        }
    }
}

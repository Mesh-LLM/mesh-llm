use std::io::{self, Write};

pub(crate) fn stdout() -> impl Write {
    io::stdout()
}

pub(crate) fn stderr() -> impl Write {
    io::stderr()
}

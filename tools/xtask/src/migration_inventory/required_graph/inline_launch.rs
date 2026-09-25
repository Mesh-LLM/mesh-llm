//! Names the unresolved boundary of an inline interpreter launch that has no
//! joined source contract. Only launches whose unresolved part is a specific,
//! source-visible selection get a named reason; every other launch keeps the
//! generic reason so a new fixed inline interpreter is never hidden.

const GENERIC: &str = "inline interpreter/installer or Python dynamic process candidate requires source-backed caller contract";
const SELECTED: &str = "inline Python program launched through a runtime-selected interpreter binding; selector output and launch are not source-joined";
const VARIABLE: &str = "Python script target is a shell variable; its bound path and branch are not joined as a child edge";
const DISCOVERY: &str = "unittest discovery selects Python test modules by pattern; the module set is not joined as child edges";
const ROOTED: &str =
    "Python script path is built from a shell variable; no repository child edge is bound";

pub(super) fn unresolved_reason(block: &str) -> &'static str {
    if selected_interpreter_program(block) {
        SELECTED
    } else if variable_script_target(block) {
        VARIABLE
    } else if block.contains(" -m unittest discover") {
        DISCOVERY
    } else if block.contains(".py") {
        ROOTED
    } else {
        GENERIC
    }
}

fn variable_name(quoted: &str) -> Option<&str> {
    let (name, _) = quoted.split_once('"')?;
    (!name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_'))
    .then_some(name)
}

/// `"$python" -c ...`, `"$py" - ...`, `"$python_bin" - ...` or
/// `"$(python_bin)" - ...`: the executable word is a quoted selector whose
/// name starts with `py`, and its next argument is an inline program flag.
fn selected_interpreter_program(block: &str) -> bool {
    block.match_indices("\"$").any(|(index, _)| {
        let rest = &block[index + 2..];
        let (name, tail) = match rest.strip_prefix('(') {
            Some(call) => match call.split_once(")\"") {
                Some((name, tail)) => (name, tail),
                None => return false,
            },
            None => match variable_name(rest) {
                Some(name) => (name, &rest[name.len() + 1..]),
                None => return false,
            },
        };
        name.starts_with("py")
            && name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            && (tail.starts_with(" -c ") || tail.starts_with(" - ") || tail == " -")
    })
}

/// `python3 "$HELPER" subcommand ...`: a fixed interpreter runs a script whose
/// path is a plain shell variable rather than a source-visible file.
pub(super) fn variable_script_target(block: &str) -> bool {
    ["python3 \"$", "python \"$"].iter().any(|launch| {
        block.match_indices(launch).any(|(index, _)| {
            let rest = &block[index + launch.len()..];
            variable_name(rest).is_some()
        })
    })
}

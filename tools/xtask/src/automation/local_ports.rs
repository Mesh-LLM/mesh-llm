//! Transient OS-assigned loopback ports for an imminent local launch.
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::net::{Ipv4Addr, TcpListener};

const MAX_PORTS: usize = 16;
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation local-ports COUNT",
    values: &[],
    flags: &["--help"],
};

fn reserve(count: usize) -> DynResult<Vec<TcpListener>> {
    if !(1..=MAX_PORTS).contains(&count) {
        return Err(format!("port count must be between 1 and {MAX_PORTS}").into());
    }
    (0..count)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(Into::into))
        .collect()
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    let [count] = parsed.positionals.as_slice() else {
        return GRAMMAR.error("exactly one port count is required").emit();
    };
    let listeners = reserve(count.parse()?)?;
    let ports = listeners
        .iter()
        .map(|listener| listener.local_addr().map(|addr| addr.port().to_string()))
        .collect::<std::io::Result<Vec<_>>>()?;
    // All listeners remain held together during selection, then close before
    // printing. This command offers no reservation after returning.
    drop(listeners);
    CheckReport::success(format!("{}\n", ports.join(","))).emit()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simultaneous_os_selection_is_distinct_and_excludes_occupied_ports() {
        let occupied = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let occupied_port = occupied.local_addr().unwrap().port();
        let listeners = reserve(16).unwrap();
        let mut ports = std::collections::BTreeSet::new();
        for listener in &listeners {
            let address = listener.local_addr().unwrap();
            assert_eq!(address.ip(), Ipv4Addr::LOCALHOST);
            assert_ne!(address.port(), 0);
            assert_ne!(address.port(), occupied_port);
            assert!(ports.insert(address.port()));
            assert!(TcpListener::bind(address).is_err());
        }
    }
    #[test]
    fn counts_outside_the_finite_budget_reject() {
        assert!(reserve(0).is_err());
        assert!(reserve(MAX_PORTS + 1).is_err());
    }
}

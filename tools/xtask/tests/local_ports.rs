use std::{
    net::{Ipv4Addr, TcpListener},
    process::Command,
};

#[test]
fn local_ports_cli_emits_only_distinct_csv_os_ports_without_occupied_endpoint() {
    let occupied = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    for count in [1, 2, 3, 5, 16] {
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "local-ports", &count.to_string()])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.ends_with('\n'));
        assert_eq!(text.lines().count(), 1);
        let ports = text
            .trim_end()
            .split(',')
            .map(|port| port.parse::<u16>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ports.len(), count);
        assert_eq!(
            ports
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            count
        );
        assert!(
            ports
                .iter()
                .all(|port| *port > 0 && *port != occupied.local_addr().unwrap().port())
        );
        // A caller must be able to acquire every advertised endpoint after the
        // allocator exits; keep all listeners alive together to prove release.
        let listeners = ports
            .iter()
            .map(|port| TcpListener::bind((Ipv4Addr::LOCALHOST, *port)).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(listeners.len(), count);
    }
}

#[test]
fn invalid_cli_counts_fail_without_port_output() {
    for args in [
        vec![],
        vec!["0"],
        vec!["-1"],
        vec!["17"],
        vec!["true"],
        vec!["1.5"],
        vec!["1", "2"],
        vec!["--unknown"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "local-ports"])
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
}

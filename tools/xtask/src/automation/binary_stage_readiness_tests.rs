use super::*;
#[test]
fn closed_inputs_reject_nonliteral_hosts_zero_ports_nonpositive_pids_and_unbounded_timeouts() {
    let valid = [
        "--host",
        "127.0.0.1",
        "--port",
        "3456",
        "--server-pid",
        "42",
        "--timeout-secs",
        "120",
    ];
    assert!(Options::parse(&valid.map(str::to_owned)).is_ok());
    for (index, invalid) in [
        (1, "localhost"),
        (3, "0"),
        (3, "65536"),
        (5, "0"),
        (5, "1"),
        (5, "-1"),
        (5, "2147483648"),
        (7, "0"),
        (7, "3601"),
        (7, "NaN"),
    ] {
        let mut args = valid.map(str::to_owned);
        args[index] = invalid.into();
        assert!(Options::parse(&args).is_err(), "{invalid}");
    }
    let mut args = valid.map(str::to_owned).to_vec();
    args.extend(["--host".into(), "::1".into()]);
    assert!(Options::parse(&args).is_err());
    args = valid.map(str::to_owned).to_vec();
    args.push("--unknown".into());
    assert!(Options::parse(&args).is_err());
}
#[test]
#[cfg(unix)]
fn cancelling_real_closed_socket_wait_is_observed_within_one_poll_interval() {
    let socket = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = socket.local_addr().unwrap();
    drop(socket);
    let options = Options {
        address,
        pid: i32::try_from(std::process::id()).unwrap(),
        timeout: Duration::from_secs(3),
    };
    let cancellation = Cancellation::default();
    let trigger = cancellation.clone();
    let before = std::time::Instant::now();
    let result = std::thread::scope(|scope| {
        scope.spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            trigger.cancel();
        });
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(wait(&options, &cancellation))
    });
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    assert!(before.elapsed() < Duration::from_secs(1));
}

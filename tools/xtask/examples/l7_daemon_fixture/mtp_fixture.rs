//! Inert orchestration fixture; never executes native MTP or loads a model.
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    time::Duration,
};
pub(super) fn selected(args: &[String]) -> bool {
    matches!(args,
        [addr, _, requests, _, concurrency, _, width, _]
        if addr == "--addr" && requests == "--requests"
            && concurrency == "--concurrency" && width == "--activation-width")
        || args.windows(2).any(|p| {
            p[0] == "--config"
                && std::fs::read_to_string(&p[1])
                    .is_ok_and(|s| s.contains("mtp-scheduler-benchmark"))
        })
}
pub(super) fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let value = |key| {
        args.windows(2)
            .find(|p| p[0] == key)
            .map(|p| p[1].as_str())
            .ok_or("fixture flag")
    };
    let native =
        PathBuf::from(std::env::var_os("LLAMA_STAGE_BUILD_DIR").ok_or("fixture build env")?);
    if args.first().is_some_and(|s| s == "serve-binary") {
        if !native.join("default-signal").exists() {
            crate::signals::install()?;
        }
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(value("--config")?)?)?;
        assert_eq!(config["native_mtp_enabled"], true);
        assert_eq!(config["selected_device"]["backend_device"], "CPU");
        assert_eq!(
            std::env::var("SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH")?,
            "1"
        );
        if native.join("early").exists() {
            std::process::exit(23);
        }
        let listener = TcpListener::bind(value("--bind-addr")?)?;
        listener.set_nonblocking(true)?;
        while !crate::signals::stopped() {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_nonblocking(false).unwrap();
                    stream.set_write_timeout(Some(Duration::from_secs(1)))?;
                    std::fs::write(native.join("ready-probed"), std::process::id().to_string())?;
                    if native.join("held").exists() {
                        while !crate::signals::stopped() {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        break;
                    }
                    let magic = if native.join("bad-magic").exists() {
                        0_i32
                    } else {
                        0x5352_4459_i32
                    };
                    stream.write_all(&magic.to_le_bytes())?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(e) => return Err(e.into()),
            }
        }
        return Ok(());
    }
    let concurrency: usize = value("--concurrency")?.parse()?;
    let requests: usize = value("--requests")?.parse()?;
    let mut stream = TcpStream::connect(value("--addr")?)?;
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;
    let mut magic = [0; 4];
    stream.read_exact(&mut magic)?;
    assert_eq!(magic, 0x5352_4459_i32.to_le_bytes());
    if native.join("client-fail").exists() && concurrency == 2 {
        std::process::exit(29);
    }
    let mut rows = Vec::new();
    let error_row = native.join("error-row").exists();
    let mismatch =
        native.join("mismatch").exists() && std::path::Path::new("new-stage.json").exists();
    for id in 1..=requests {
        if error_row && id == requests {
            rows.push(serde_json::json!({"request_id":id,"error":"inert request failure"}));
            continue;
        }
        rows.push(serde_json::json!({"request_id":if native.join("duplicate").exists(){1}else{id},"elapsed_ms":if native.join("huge-latency").exists(){f64::MAX}else{(id*2) as f64},"predicted":if mismatch {8}else{7},"draft":9,"verified":9,"accepted":true}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"requests":requests,"concurrency":concurrency,"makespan_ms":1000,"throughput_rps":requests-usize::from(error_row),"successful":requests-usize::from(error_row),"failed":usize::from(error_row),"drafted":requests-usize::from(error_row),"accepted":requests-usize::from(error_row),"acceptance_rate":if requests==usize::from(error_row){0.0}else{1.0},"per_request":rows})
        )?
    );
    Ok(())
}

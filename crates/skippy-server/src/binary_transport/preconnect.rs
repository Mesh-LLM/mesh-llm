use std::{
    net::TcpStream,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use skippy_protocol::StageConfig;

use super::stage_execution::connect_binary_downstream;

const WARM_DOWNSTREAM_RETRY_SLEEP: Duration = Duration::from_millis(500);
const WARM_DOWNSTREAM_SLOT_POLL: Duration = Duration::from_millis(50);
const WARM_DOWNSTREAM_CONNECT_TIMEOUT_SECS: u64 = 2;

pub(super) fn spawn_downstream_preconnector(
    config: StageConfig,
    warm_downstream: Arc<Mutex<Option<TcpStream>>>,
    shutdown: Arc<AtomicBool>,
) {
    if config.downstream.is_none() {
        return;
    }
    let thread_name = format!("skippy-warm-downstream-{}", config.stage_index);
    let _ = thread::Builder::new().name(thread_name).spawn(move || {
        run_downstream_preconnector(config, warm_downstream, shutdown);
    });
}

fn run_downstream_preconnector(
    config: StageConfig,
    warm_downstream: Arc<Mutex<Option<TcpStream>>>,
    shutdown: Arc<AtomicBool>,
) {
    while !shutdown.load(Ordering::SeqCst) {
        if warm_slot_is_full(&warm_downstream) {
            thread::sleep(WARM_DOWNSTREAM_SLOT_POLL);
            continue;
        }
        match connect_binary_downstream(&config, WARM_DOWNSTREAM_CONNECT_TIMEOUT_SECS) {
            Ok(Some(stream)) => {
                eprintln!(
                    "downstream warm preconnect ready: stage_id={} local={:?} remote={:?}",
                    config.stage_id,
                    stream.local_addr().ok(),
                    stream.peer_addr().ok(),
                );
                store_warm_stream(&warm_downstream, stream);
            }
            Ok(None) => return,
            Err(error) => {
                eprintln!(
                    "downstream warm preconnect failed: stage_id={} error={error:#}",
                    config.stage_id,
                );
                thread::sleep(WARM_DOWNSTREAM_RETRY_SLEEP);
            }
        }
    }
}

fn warm_slot_is_full(warm_downstream: &Arc<Mutex<Option<TcpStream>>>) -> bool {
    warm_downstream
        .lock()
        .map(|guard| guard.is_some())
        .unwrap_or(true)
}

fn store_warm_stream(warm_downstream: &Arc<Mutex<Option<TcpStream>>>, stream: TcpStream) {
    let Ok(mut guard) = warm_downstream.lock() else {
        return;
    };
    if guard.is_none() {
        *guard = Some(stream);
    }
}

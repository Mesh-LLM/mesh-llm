use std::{
    io,
    net::TcpStream,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use skippy_protocol::StageConfig;

use super::stage_execution::connect_binary_downstream_cancellable;

const WARM_DOWNSTREAM_RETRY_SLEEP: Duration = Duration::from_millis(500);
const WARM_DOWNSTREAM_SLOT_POLL: Duration = Duration::from_millis(50);
const WARM_DOWNSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

pub(super) struct DownstreamPreconnector {
    shutdown: Arc<AtomicBool>,
    task: Option<thread::JoinHandle<()>>,
}

impl DownstreamPreconnector {
    pub(super) fn spawn(
        config: StageConfig,
        warm_downstream: Arc<Mutex<Option<TcpStream>>>,
        shutdown: Arc<AtomicBool>,
    ) -> io::Result<Self> {
        if config.downstream.is_none() {
            return Ok(Self {
                shutdown,
                task: None,
            });
        }
        let thread_name = format!("skippy-warm-downstream-{}", config.stage_index);
        let task_shutdown = shutdown.clone();
        let task = thread::Builder::new()
            .name(thread_name)
            .spawn(move || run_downstream_preconnector(config, warm_downstream, task_shutdown))?;
        Ok(Self {
            shutdown,
            task: Some(task),
        })
    }
}

impl Drop for DownstreamPreconnector {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(task) = self.task.take() {
            let _ = task.join();
        }
    }
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
        match connect_binary_downstream_cancellable(
            &config,
            WARM_DOWNSTREAM_CONNECT_TIMEOUT,
            &shutdown,
        ) {
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
                if shutdown.load(Ordering::SeqCst) {
                    return;
                }
                eprintln!(
                    "downstream warm preconnect failed: stage_id={} error={error:#}",
                    config.stage_id,
                );
                sleep_until_retry_or_shutdown(&shutdown);
            }
        }
    }
}

fn sleep_until_retry_or_shutdown(shutdown: &AtomicBool) {
    let deadline = std::time::Instant::now() + WARM_DOWNSTREAM_RETRY_SLEEP;
    while !shutdown.load(Ordering::SeqCst) {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        thread::sleep(remaining.min(WARM_DOWNSTREAM_SLOT_POLL));
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

#[cfg(test)]
mod tests {
    use super::DownstreamPreconnector;
    use crate::binary_transport::stage_execution::prefix_cache_test_config;
    use std::{
        net::TcpListener,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
    };

    #[test]
    fn dropping_preconnector_signals_and_joins_its_task() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let task_shutdown = shutdown.clone();
        let (finished_tx, finished_rx) = mpsc::sync_channel(1);
        let task = thread::spawn(move || {
            while !task_shutdown.load(Ordering::Acquire) {
                thread::yield_now();
            }
            finished_tx.send(()).unwrap();
        });
        let preconnector = DownstreamPreconnector {
            shutdown: shutdown.clone(),
            task: Some(task),
        };

        drop(preconnector);

        assert!(shutdown.load(Ordering::Acquire));
        finished_rx
            .try_recv()
            .expect("preconnector task must be joined before the guard is dropped");
    }

    #[test]
    fn spawned_preconnector_stops_and_joins_after_shutdown() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        drop(listener);
        let mut config = prefix_cache_test_config();
        config.downstream.as_mut().unwrap().endpoint = endpoint;
        let shutdown = Arc::new(AtomicBool::new(false));
        let preconnector =
            DownstreamPreconnector::spawn(config, Arc::new(Mutex::new(None)), shutdown.clone())
                .unwrap();
        thread::sleep(Duration::from_millis(20));

        shutdown.store(true, Ordering::Release);
        let (finished_tx, finished_rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            drop(preconnector);
            finished_tx.send(()).unwrap();
        });

        finished_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("spawned preconnector must stop and join within its connect bound");
    }
}

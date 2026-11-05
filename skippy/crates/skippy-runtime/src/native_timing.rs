//! Wall-clock timing of calls into the native Skippy ABI.
//!
//! Timings go to one optional process-wide observer; serving installs it for
//! debug telemetry. With no observer a timed call costs one atomic load. While
//! observed, each call also sets the id the observer gave it as the native
//! trace tag, so the spans and GPU work it causes name it as their parent
//! (see [`crate::native_tracing`]).

use std::{
    sync::OnceLock,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// One completed call across the native ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCall {
    /// The exported C symbol, for example `skippy_prefill_chunk`.
    pub symbol: &'static str,
    /// The id [`NativeCallObserver::begin`] returned for this call.
    pub id: u64,
    pub start_unix_nanos: u64,
    pub end_unix_nanos: u64,
}

pub trait NativeCallObserver: Send + Sync {
    /// Called on the calling thread just before the call. The id comes back
    /// in [`NativeCall::id`] and tags the native spans and GPU work the call
    /// causes, so it should be nonzero, unique among calls in flight, and
    /// leave the top bit (native span ids) clear.
    fn begin(&self, symbol: &'static str) -> u64;
    fn observe(&self, call: NativeCall);
}

static OBSERVER: OnceLock<Box<dyn NativeCallObserver>> = OnceLock::new();

/// Install the process-wide observer. The first installation wins; returns
/// false when one is already installed.
pub fn set_native_call_observer(observer: Box<dyn NativeCallObserver>) -> bool {
    OBSERVER.set(observer).is_ok()
}

/// Run `call`, reporting its wall time as `symbol` when observed.
pub(crate) fn timed<T>(symbol: &'static str, call: impl FnOnce() -> T) -> T {
    let Some(observer) = OBSERVER.get() else {
        return call();
    };
    // Installation is observed as a call of its own, before this one starts.
    crate::native_tracing::ensure_installed();
    observe(observer.as_ref(), symbol, |_| call())
}

/// Run `call` with the id its observation will carry, without installing
/// native tracing: for the installation itself.
pub(crate) fn timed_with_id<T>(symbol: &'static str, call: impl FnOnce(u64) -> T) -> T {
    match OBSERVER.get() {
        Some(observer) => observe(observer.as_ref(), symbol, call),
        None => call(0),
    }
}

fn observe<T>(
    observer: &dyn NativeCallObserver,
    symbol: &'static str,
    call: impl FnOnce(u64) -> T,
) -> T {
    let id = observer.begin(symbol);
    let start_unix_nanos = unix_nanos_now();
    let started = Instant::now();
    let caller = crate::native_tracing::swap_tag(id);
    let result = call(id);
    crate::native_tracing::swap_tag(caller);
    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    observer.observe(NativeCall {
        symbol,
        id,
        start_unix_nanos,
        end_unix_nanos: start_unix_nanos.saturating_add(elapsed),
    });
    result
}

fn unix_nanos_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
        })
}

/// Call `skippy_ffi::$symbol(..)` with its wall time reported under the
/// symbol's own name, so a span can never be mislabelled. Keep the
/// surrounding `unsafe` block at the call site.
macro_rules! native_call {
    ($symbol:ident($($arg:expr),* $(,)?)) => {
        $crate::native_timing::timed(stringify!($symbol), || {
            skippy_ffi::$symbol($($arg),*)
        })
    };
}
pub(crate) use native_call;

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{NativeCall, NativeCallObserver, set_native_call_observer, timed};

    struct Recorder(Arc<Mutex<Vec<NativeCall>>>);

    impl NativeCallObserver for Recorder {
        fn begin(&self, _symbol: &'static str) -> u64 {
            41
        }

        fn observe(&self, call: NativeCall) {
            self.0.lock().expect("recorder lock").push(call);
        }
    }

    #[test]
    fn observed_calls_report_their_symbol_and_wall_time() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        assert!(set_native_call_observer(Box::new(Recorder(calls.clone()))));
        assert!(!set_native_call_observer(Box::new(Recorder(calls.clone()))));

        let value = timed("skippy_test_symbol", || {
            std::thread::sleep(std::time::Duration::from_millis(1));
            7
        });

        assert_eq!(value, 7);
        let calls = calls.lock().expect("recorder lock");
        let call = calls
            .iter()
            .find(|call| call.symbol == "skippy_test_symbol")
            .expect("timed call observed");
        assert!(call.end_unix_nanos - call.start_unix_nanos >= 1_000_000);
        assert_eq!(call.id, 41, "the observer's id names the call");
    }
}

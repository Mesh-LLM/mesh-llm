//! Trace context for Skippy telemetry.
//!
//! One trace covers the whole process under a root span. A span's parent is
//! whatever is active where it starts: the innermost span entered on this
//! thread, else the span of the enclosing tokio task, else the root. Point
//! events attach to that same span. Work batched for several requests, such
//! as a scheduler iteration, links to the request spans bound to its sequence
//! ids.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, hash_map::RandomState},
    future::Future,
    hash::{BuildHasher, Hasher},
    marker::PhantomData,
    sync::{
        Arc, Mutex, OnceLock, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use serde_json::Value;

/// Events beyond this stay standalone spans rather than growing a buffer.
const SPAN_EVENT_LIMIT: usize = 1024;
const ROOT_EVENT_LIMIT: usize = 10_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SpanContext {
    pub(crate) trace_id: [u8; 16],
    pub(crate) span_id: [u8; 8],
}

#[derive(Clone, Debug)]
pub(crate) struct SpanEvent {
    pub(crate) name: String,
    pub(crate) time_unix_nanos: u64,
    pub(crate) attributes: BTreeMap<String, Value>,
}

struct EventBuffer {
    events: Mutex<Vec<SpanEvent>>,
    closed: AtomicBool,
    limit: usize,
}

impl EventBuffer {
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Vec::new()),
            closed: AtomicBool::new(false),
            limit,
        })
    }

    fn record(&self, event: SpanEvent) -> Result<(), SpanEvent> {
        let mut events = self.events.lock().unwrap_or_else(PoisonError::into_inner);
        if self.closed.load(Ordering::Acquire) || events.len() >= self.limit {
            return Err(event);
        }
        events.push(event);
        Ok(())
    }

    fn close(&self) -> Vec<SpanEvent> {
        let mut events = self.events.lock().unwrap_or_else(PoisonError::into_inner);
        self.closed.store(true, Ordering::Release);
        std::mem::take(&mut events)
    }
}

/// A span other telemetry can attach to: its identity, its parent, and the
/// events recorded while it is open.
#[derive(Clone)]
pub(crate) struct Frame {
    context: SpanContext,
    parent: Option<[u8; 8]>,
    events: Arc<EventBuffer>,
}

impl Frame {
    /// A new span under whatever is active here.
    pub(crate) fn child() -> Self {
        current_frame_or_root().new_child()
    }

    /// A new span under this one, wherever it is created.
    pub(crate) fn new_child(&self) -> Self {
        Self {
            context: SpanContext {
                trace_id: self.context.trace_id,
                span_id: next_span_id(),
            },
            parent: Some(self.context.span_id),
            events: EventBuffer::new(SPAN_EVENT_LIMIT),
        }
    }

    /// A new span under `parent`, which may be on another thread or ended.
    pub(crate) fn under(parent: SpanContext) -> Self {
        Self {
            context: SpanContext {
                trace_id: parent.trace_id,
                span_id: next_span_id(),
            },
            parent: Some(parent.span_id),
            events: EventBuffer::new(SPAN_EVENT_LIMIT),
        }
    }

    /// The span native code already identified by `tag`, under `parent`.
    pub(crate) fn tagged_under(parent: SpanContext, tag: u64) -> Self {
        Self {
            context: SpanContext {
                trace_id: parent.trace_id,
                span_id: tag.to_be_bytes(),
            },
            parent: Some(parent.span_id),
            events: EventBuffer::new(SPAN_EVENT_LIMIT),
        }
    }

    pub(crate) fn context(&self) -> SpanContext {
        self.context
    }

    /// The span id as a nonzero integer, for native code that can only carry
    /// a `u64`; [`context_for_tag`] turns it back into a parent.
    pub(crate) fn tag(&self) -> u64 {
        u64::from_be_bytes(self.context.span_id)
    }

    pub(crate) fn parent_span_id(&self) -> Option<[u8; 8]> {
        self.parent
    }

    /// Make this the active span on this thread until the guard drops. For
    /// synchronous code; async code uses [`Frame::enter_task`].
    pub(crate) fn enter(&self) -> Entered {
        // Pushing can only fail while the thread is tearing down; the guard
        // then removes nothing, which leaves the stack consistent.
        let depth = STACK
            .try_with(|stack| {
                let mut stack = stack.borrow_mut();
                stack.push(self.clone());
                stack.len() - 1
            })
            .ok();
        Entered {
            depth,
            _not_send: PhantomData,
        }
    }

    /// Make this the active span of the current task until the guard drops.
    /// The guard is `Send`, so it may be held across `.await`. Outside a task
    /// scope (see [`in_task`]) there is nothing to attach to and it is inert.
    pub(crate) fn enter_task(&self) -> TaskEntered {
        let stack = TASK_STACK.try_with(Arc::clone).ok();
        if let Some(stack) = stack.as_ref() {
            stack
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(self.clone());
        }
        TaskEntered {
            stack,
            span_id: self.context.span_id,
        }
    }

    /// Attach an event; hands it back once the span has ended or is full.
    pub(crate) fn record_event(&self, event: SpanEvent) -> Result<(), SpanEvent> {
        self.events.record(event)
    }

    /// End event collection and take what was recorded.
    pub(crate) fn take_events(&self) -> Vec<SpanEvent> {
        self.events.close()
    }
}

/// Keeps a span active on the current thread. Not `Send`, so it cannot be
/// held across an `.await` that may resume elsewhere. Dropping it also ends
/// any span entered above it that was left behind, so a missed guard cannot
/// leak into later work on the thread.
#[must_use = "the span is only active while the guard is alive"]
pub(crate) struct Entered {
    depth: Option<usize>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for Entered {
    fn drop(&mut self) {
        if let Some(depth) = self.depth {
            let _ = STACK.try_with(|stack| stack.borrow_mut().truncate(depth));
        }
    }
}

type TaskStack = Arc<Mutex<Vec<Frame>>>;

/// Keeps a span active for the task that entered it. Dropping it, on any
/// thread, removes that span and anything entered above it.
#[must_use = "the span is only active while the guard is alive"]
pub(crate) struct TaskEntered {
    stack: Option<TaskStack>,
    span_id: [u8; 8],
}

impl Drop for TaskEntered {
    fn drop(&mut self) {
        let Some(stack) = self.stack.as_ref() else {
            return;
        };
        let mut stack = stack.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(depth) = stack
            .iter()
            .rposition(|frame| frame.context.span_id == self.span_id)
        {
            stack.truncate(depth);
        }
    }
}

thread_local! {
    static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
}

tokio::task_local! {
    static TASK_STACK: TaskStack;
}

/// The innermost active span: this thread's, else the current task's.
pub(crate) fn current_frame() -> Option<Frame> {
    STACK
        .try_with(|stack| stack.borrow().last().cloned())
        .ok()
        .flatten()
        .or_else(|| {
            TASK_STACK
                .try_with(|stack| {
                    stack
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .last()
                        .cloned()
                })
                .ok()
                .flatten()
        })
}

/// The span a [`Frame::tag`] names, in this process's trace; the root for 0.
pub(crate) fn context_for_tag(tag: u64) -> SpanContext {
    let root = process().root().context();
    if tag == 0 {
        return root;
    }
    SpanContext {
        trace_id: root.trace_id,
        span_id: tag.to_be_bytes(),
    }
}

pub(crate) fn current_frame_or_root() -> Frame {
    current_frame().unwrap_or_else(|| process().root().clone())
}

/// Run `future` with `frame` as the span of its task.
pub(crate) async fn in_task<F: Future>(frame: Frame, future: F) -> F::Output {
    TASK_STACK
        .scope(Arc::new(Mutex::new(vec![frame])), future)
        .await
}

/// Carry the caller's active span onto the thread that runs `work`.
pub(crate) fn propagate<T, F>(work: F) -> impl FnOnce() -> T + Send + 'static
where
    F: FnOnce() -> T + Send + 'static,
{
    let frame = current_frame();
    move || {
        let _entered = frame.as_ref().map(Frame::enter);
        work()
    }
}

/// The process root span, created on first use.
pub(crate) struct ProcessTrace {
    root: Frame,
    start_unix_nanos: u64,
    finished: AtomicBool,
}

impl ProcessTrace {
    pub(crate) fn root(&self) -> &Frame {
        &self.root
    }

    pub(crate) fn start_unix_nanos(&self) -> u64 {
        self.start_unix_nanos
    }

    /// True for exactly one caller: the one that ends the process span.
    pub(crate) fn begin_finish(&self) -> bool {
        !self.finished.swap(true, Ordering::AcqRel)
    }

    /// Whether the process span has ended: the process is shutting down.
    pub(crate) fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
}

static PROCESS: OnceLock<ProcessTrace> = OnceLock::new();
static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn process() -> &'static ProcessTrace {
    PROCESS.get_or_init(|| ProcessTrace {
        root: Frame {
            context: SpanContext {
                trace_id: new_trace_id(),
                span_id: next_span_id(),
            },
            parent: None,
            events: EventBuffer::new(ROOT_EVENT_LIMIT),
        },
        start_unix_nanos: u64::try_from(super::now_unix_nanos()).unwrap_or_default(),
        finished: AtomicBool::new(false),
    })
}

fn next_span_id() -> [u8; 8] {
    NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed).to_be_bytes()
}

/// Per-process random keys make the id unique across runs without a new
/// dependency; it identifies a trace and is not a secret.
fn new_trace_id() -> [u8; 16] {
    let state = RandomState::new();
    let mut id = [0_u8; 16];
    for (half, bytes) in id.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut hasher = state.build_hasher();
        hasher.write_i64(super::now_unix_nanos());
        hasher.write_u32(std::process::id());
        hasher.write_usize(half);
        *bytes = hasher.finish().to_be_bytes();
    }
    id
}

static SEQUENCES: OnceLock<Mutex<HashMap<String, SpanContext>>> = OnceLock::new();

fn sequences() -> &'static Mutex<HashMap<String, SpanContext>> {
    SEQUENCES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Ties a scheduler sequence or session id to the span that submitted it.
#[must_use = "the binding is removed when the guard drops"]
pub(crate) struct BoundSequence {
    id: String,
    context: SpanContext,
}

impl Drop for BoundSequence {
    fn drop(&mut self) {
        let mut sequences = sequences().lock().unwrap_or_else(PoisonError::into_inner);
        if sequences.get(&self.id) == Some(&self.context) {
            sequences.remove(&self.id);
        }
    }
}

/// Bind `id` to the active span for as long as the guard lives.
pub(crate) fn bind_sequence(id: &str) -> BoundSequence {
    let context = current_frame_or_root().context();
    sequences()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(id.to_string(), context);
    BoundSequence {
        id: id.to_string(),
        context,
    }
}

pub(crate) fn sequence_context(id: &str) -> Option<SpanContext> {
    sequences()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(id)
        .copied()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        Frame, SpanEvent, bind_sequence, current_frame, current_frame_or_root, in_task, process,
        propagate, sequence_context,
    };

    fn current_context() -> super::SpanContext {
        current_frame_or_root().context()
    }

    fn event(name: &str) -> SpanEvent {
        SpanEvent {
            name: name.to_string(),
            time_unix_nanos: 1,
            attributes: BTreeMap::new(),
        }
    }

    #[test]
    fn spans_nest_under_the_entered_span_and_share_the_process_trace() {
        let outer = Frame::child();
        let _outer = outer.enter();
        let inner = Frame::child();
        assert_eq!(inner.parent_span_id(), Some(outer.context().span_id));
        assert_eq!(
            inner.context().trace_id,
            process().root().context().trace_id
        );
        {
            let _inner = inner.enter();
            assert_eq!(current_frame_or_root().context(), inner.context());
        }
        assert_eq!(current_frame_or_root().context(), outer.context());
    }

    #[test]
    fn without_an_active_span_work_attaches_to_the_root() {
        let handle = std::thread::spawn(|| current_frame_or_root().context());
        assert_eq!(handle.join().unwrap(), process().root().context());
    }

    #[test]
    fn propagated_work_runs_under_the_callers_span() {
        let frame = Frame::child();
        let _entered = frame.enter();
        let work = propagate(|| current_frame().map(|frame| frame.context()));
        let observed = std::thread::spawn(work).join().unwrap();
        assert_eq!(observed, Some(frame.context()));
    }

    #[test]
    fn task_scope_supplies_the_span_to_async_work() {
        let frame = Frame::child();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let observed = runtime.block_on(in_task(frame.clone(), async {
            current_frame().unwrap().context()
        }));
        assert_eq!(observed, frame.context());
    }

    #[test]
    fn events_stop_attaching_once_the_span_ends() {
        let frame = Frame::child();
        assert!(frame.record_event(event("first")).is_ok());
        let events = frame.take_events();
        assert_eq!(events.len(), 1);
        assert_eq!(frame.record_event(event("late")).unwrap_err().name, "late");
    }

    #[test]
    fn sequence_bindings_last_as_long_as_their_guard() {
        let frame = Frame::child();
        let _entered = frame.enter();
        {
            let _bound = bind_sequence("trace-context-test-seq");
            assert_eq!(
                sequence_context("trace-context-test-seq"),
                Some(frame.context())
            );
        }
        assert_eq!(sequence_context("trace-context-test-seq"), None);
    }

    #[test]
    fn leaving_a_thread_scope_ends_spans_left_inside_it() {
        let outer = Frame::child();
        let leaked = Frame::child();
        {
            let _outer = outer.enter();
            std::mem::forget(leaked.enter());
            assert_eq!(current_context(), leaked.context());
        }
        assert_ne!(current_context(), outer.context());
        assert_ne!(current_context(), leaked.context());
    }

    #[test]
    fn task_spans_nest_across_awaits_and_end_from_any_thread() {
        let request = Frame::child();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .unwrap();
        let (phase, during, after) = runtime.block_on(in_task(request.clone(), async {
            let phase = current_frame_or_root().new_child();
            let entered = phase.enter_task();
            tokio::task::yield_now().await;
            let during = current_context();
            // A guard moved off the task still ends its span.
            std::thread::spawn(move || drop(entered)).join().unwrap();
            (phase, during, current_context())
        }));
        assert_eq!(phase.parent_span_id(), Some(request.context().span_id));
        assert_eq!(during, phase.context());
        assert_eq!(after, request.context());
    }

    #[test]
    fn a_tag_names_its_span_for_children_made_elsewhere() {
        let call = Frame::child();
        assert_ne!(call.tag(), 0);
        let parent = super::context_for_tag(call.tag());
        assert_eq!(parent, call.context());
        let gpu = std::thread::spawn(move || Frame::under(parent))
            .join()
            .unwrap();
        assert_eq!(gpu.parent_span_id(), Some(call.context().span_id));
        assert_eq!(super::context_for_tag(0), process().root().context());
    }

    #[test]
    fn a_tagged_span_keeps_the_id_native_code_gave_it() {
        let call = Frame::child();
        let native = Frame::tagged_under(call.context(), (1 << 63) | 9);
        assert_eq!(native.tag(), (1 << 63) | 9);
        assert_eq!(native.parent_span_id(), Some(call.context().span_id));
        assert_eq!(native.context().trace_id, call.context().trace_id);
        assert_eq!(super::context_for_tag(native.tag()), native.context());
    }

    #[test]
    fn outside_a_task_scope_a_task_guard_is_inert() {
        let frame = Frame::child();
        let _entered = frame.enter_task();
        assert_ne!(current_context(), frame.context());
    }
}

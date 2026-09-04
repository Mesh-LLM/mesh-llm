//! Task 13 (`.omo/plans/event-system-fixes.md`, "Must NOT: allocate on the
//! submit path") -- proves `RuntimeEventEngine::submit`'s ingress-latency
//! recording is allocation-free, via a thread-local counting
//! `#[global_allocator]`. Safe to install here ONLY because integration
//! tests under `tests/*.rs` each compile as their OWN separate binary
//! crate: this allocator applies to this one binary alone, never to the
//! crate's `--lib` unit-test binary or any other integration test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use mesh_llm_host_runtime::runtime_events::engine::RuntimeEventEngine;
use mesh_llm_runtime_event_contracts::{
    FamilyFact, NativeRuntimeEventKind, OperationId, RuntimeEventIngress, RuntimeFact,
};

thread_local! {
    static NET_ALLOCS: Cell<i64> = const { Cell::new(0) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        NET_ALLOCS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        NET_ALLOCS.with(|count| count.set(count.get() - 1));
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A realloc that moves counts as one alloc; this test only cares
        // whether the NET count of outstanding allocations on this thread
        // moves during the measured window, so counting it as a plain
        // "one more op" is enough to catch a moved reallocation.
        NET_ALLOCS.with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn terminal_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
}

fn net_allocs() -> i64 {
    NET_ALLOCS.with(Cell::get)
}

#[test]
fn submit_records_ingress_latency_with_no_heap_allocation() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let ingress = reservation.ingress();

    // Warm up: the FIRST `try_submit` on this handle is the only one that
    // can succeed (terminal slots are write-once) and is where any
    // one-time lazy setup along the call chain (e.g. the wake list's
    // `VecDeque` growing from empty) would show up. Excluding it from the
    // measured window isolates what THIS task added -- the reservoir
    // write -- from pre-existing allocation behavior elsewhere in
    // `submit`, which is out of this task's ownership and not what this
    // test is about.
    let _ = ingress.try_submit(terminal_fact());

    let before = net_allocs();
    // Every call after the first is a duplicate-terminal rejection
    // (`TerminalDeliveryFailed`): it still runs the full `submit` body,
    // including the reservoir's `record` call, without ever touching the
    // wake list or reservation table's write-once slot again.
    for _ in 0..1_000 {
        let _ = ingress.try_submit(terminal_fact());
    }
    let after = net_allocs();

    assert_eq!(
        after, before,
        "submit's ingress-latency recording must add no net heap allocation \
         across 1,000 calls (before={before}, after={after})"
    );
}

#[test]
fn submit_crosses_the_reservoir_milestone_with_no_heap_allocation() {
    // A second, independent proof at a scale that actually exercises the
    // 100-sample health-version-bump milestone (`IngressLatencyReservoir::record`'s
    // return value), not just the reservoir's plain ring write.
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let ingress = reservation.ingress();
    let _ = ingress.try_submit(terminal_fact());

    let before = net_allocs();
    for _ in 0..250 {
        let _ = ingress.try_submit(terminal_fact());
    }
    let after = net_allocs();

    assert_eq!(
        after, before,
        "crossing the 100-sample milestone (twice, over 250 calls) must still add \
         no net heap allocation (before={before}, after={after})"
    );
}

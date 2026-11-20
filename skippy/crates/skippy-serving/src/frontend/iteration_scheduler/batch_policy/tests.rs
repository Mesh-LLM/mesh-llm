use super::*;

fn queued(
    session: &'static str,
    tokens: usize,
    phase: IterationBatchPhase,
) -> QueuedRequest<'static> {
    QueuedRequest {
        session,
        tokens,
        phase,
    }
}

fn view(
    queued: Vec<QueuedRequest<'static>>,
    lanes: usize,
    token_budget: usize,
) -> BatchView<'static> {
    BatchView {
        queued,
        active_sessions: 0,
        lanes,
        token_budget,
    }
}

fn default_tuning() -> BatchTuning {
    BatchTuning::resolve(BatchPlatform::Other, &BatchConfig::default(), None)
}

const DECODE: IterationBatchPhase = IterationBatchPhase::Decode;
const PREFILL: IterationBatchPhase = IterationBatchPhase::Prefill;

#[test]
fn device_names_map_to_platforms() {
    assert_eq!(
        BatchPlatform::from_device_name("MTL0"),
        BatchPlatform::Metal
    );
    assert_eq!(
        BatchPlatform::from_device_name("CUDA1"),
        BatchPlatform::Cuda
    );
    assert_eq!(
        BatchPlatform::from_device_name("Vulkan0"),
        BatchPlatform::Vulkan
    );
    assert_eq!(BatchPlatform::from_device_name("CPU"), BatchPlatform::Cpu);
    assert_eq!(
        BatchPlatform::from_device_name("SYCL0"),
        BatchPlatform::Other
    );
}

/// Every backend starts from the values the scheduler has always used.
#[test]
fn unset_tuning_is_the_scheduler_default_on_every_platform() {
    for platform in [
        BatchPlatform::Metal,
        BatchPlatform::Cuda,
        BatchPlatform::Vulkan,
        BatchPlatform::Cpu,
        BatchPlatform::Other,
    ] {
        let tuning = BatchTuning::resolve(platform, &BatchConfig::default(), None);
        assert_eq!(tuning.decode_window, Duration::from_millis(2));
        assert_eq!(tuning.prefill_window_per_token, Duration::from_micros(15));
        assert_eq!(tuning.prefill_window_max, Duration::from_millis(8));
    }
}

#[test]
fn configured_tuning_replaces_the_platform_values() {
    let config = BatchConfig {
        policy: None,
        decode_window_us: Some(500),
        prefill_window_per_token_us: Some(40),
    };
    let tuning = BatchTuning::resolve(BatchPlatform::Metal, &config, Some(20));
    assert_eq!(tuning.decode_window, Duration::from_micros(500));
    assert_eq!(tuning.prefill_window_per_token, Duration::from_micros(40));
    assert_eq!(tuning.prefill_window_max, Duration::from_millis(20));
    assert_eq!(tuning.prefill_window(100), Duration::from_millis(4));
    assert_eq!(tuning.prefill_window(4096), Duration::from_millis(20));
    assert_eq!(tuning.prefill_window(1), Duration::from_micros(500));
}

#[test]
fn a_prompt_gathers_a_full_wave_and_generation_the_active_sessions() {
    let policy = ArrivalOrder {
        tuning: default_tuning(),
    };
    let mut with_prompt = view(
        vec![queued("a", 1, DECODE), queued("b", 512, PREFILL)],
        4,
        2048,
    );
    with_prompt.active_sessions = 2;
    assert_eq!(
        policy.gather(&with_prompt),
        Gather {
            target: 4,
            window: Duration::from_micros(7_680),
        }
    );
    let mut generating = view(vec![queued("a", 1, DECODE)], 4, 2048);
    generating.active_sessions = 3;
    assert_eq!(
        policy.gather(&generating),
        Gather {
            target: 3,
            window: Duration::from_millis(2),
        }
    );
    assert_eq!(policy.gather(&view(Vec::new(), 4, 2048)).target, 0);
}

#[test]
fn arrival_order_keeps_every_rule() {
    let policy = ArrivalOrder {
        tuning: default_tuning(),
    };
    let view = view(
        vec![
            queued("a", 4, PREFILL),
            queued("b", 9, PREFILL),
            queued("a", 1, DECODE),
            queued("b", 1, DECODE),
            queued("c", 1, DECODE),
        ],
        3,
        8,
    );
    let selection = policy.select(&view);
    // b's prefill does not fit; its later decode must not overtake it
    assert_eq!(selection, Selection(vec![0, 4]));
    assert_eq!(check_selection(&view, &selection), Ok(()));
}

#[test]
fn a_selection_that_breaks_a_rule_is_named() {
    let view = view(
        vec![
            queued("a", 4, PREFILL),
            queued("b", 4, PREFILL),
            queued("a", 1, DECODE),
            queued("c", 6, PREFILL),
        ],
        2,
        8,
    );
    let rejected = |positions: &[usize]| check_selection(&view, &Selection(positions.to_vec()));
    assert_eq!(rejected(&[]), Err(BatchRejection::Empty));
    assert_eq!(rejected(&[1, 0]), Err(BatchRejection::NotInQueueOrder));
    assert_eq!(rejected(&[0, 0]), Err(BatchRejection::NotInQueueOrder));
    assert_eq!(rejected(&[0, 9]), Err(BatchRejection::OutOfRange));
    assert_eq!(rejected(&[1]), Err(BatchRejection::SkipsQueueHead));
    assert_eq!(rejected(&[0, 1, 3]), Err(BatchRejection::TooManyRequests));
    assert_eq!(rejected(&[0, 3]), Err(BatchRejection::OverTokenBudget));
    assert_eq!(rejected(&[0, 2]), Err(BatchRejection::SessionTwice));
    assert_eq!(rejected(&[0, 1]), Ok(()));
}

#[test]
fn a_later_request_may_not_overtake_its_own_session() {
    let view = view(
        vec![
            queued("a", 1, DECODE),
            queued("b", 4, PREFILL),
            queued("b", 1, DECODE),
        ],
        4,
        8,
    );
    assert_eq!(
        check_selection(&view, &Selection(vec![0, 2])),
        Err(BatchRejection::OvertakesSession)
    );
    assert_eq!(check_selection(&view, &Selection(vec![0, 1])), Ok(()));
}

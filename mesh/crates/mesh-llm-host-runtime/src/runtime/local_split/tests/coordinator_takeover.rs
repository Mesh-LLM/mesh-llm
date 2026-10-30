//! A peer that joins while a split loads can win the coordinator election and
//! place one of its own stages on the loading coordinator. On a 4-node GLM
//! split this left one Spark holding layers 0..24 of its own topology and
//! 24..45 of the newer one — the whole model — and the kernel killed it. The
//! loading coordinator has to read that takeover from its own stage control
//! and stand down before stage 0 loads.

use super::*;
use tokio::sync::oneshot;

const LOCAL: &str = "local-coordinator";
const NEWER: &str = "newer-coordinator";

fn claim(coordinator_id: &str, term: u64, topology: &str) -> skippy::StageCoordinatorClaim {
    skippy::StageCoordinatorClaim {
        model_id: "meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers@rev".to_string(),
        package_ref: "hf://meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers@rev".to_string(),
        manifest_sha256: "manifest".to_string(),
        topology_id: format!("topology-{topology}"),
        run_id: topology.to_string(),
        coordinator_id: coordinator_id.to_string(),
        coordinator_term: term,
        participant_set_hash: format!("participants-{topology}"),
        topology_hash: format!("stages-{topology}"),
        lease_until_unix_ms: u64::MAX,
    }
}

async fn send_claim(
    control: &skippy::StageControlHandle,
    claim: skippy::StageCoordinatorClaim,
) -> skippy::StageCoordinatorClaimAck {
    let (resp, rx) = oneshot::channel();
    control
        .sender()
        .send(skippy::StageControlCommand::Execute {
            request: skippy::StageControlRequest::Claim(claim),
            resp,
        })
        .expect("stage control accepts the claim");
    match rx
        .await
        .expect("stage control answers")
        .expect("claim runs")
    {
        skippy::StageControlResponse::ClaimAccepted(ack) => ack,
        other => panic!("claim answered with {other:?}"),
    }
}

#[tokio::test]
async fn a_newer_coordinator_claim_supersedes_the_loading_coordinator() {
    let control = skippy::spawn_stage_control_loop(skippy::SkippyTelemetryOptions::default());
    let ours = claim(LOCAL, 11, "a");
    assert!(send_claim(&control, ours.clone()).await.accepted);
    // Re-sending our own claim is how the coordinator checks its lease; with
    // nothing newer it is accepted unchanged.
    let ack = send_claim(&control, ours.clone()).await;
    assert_eq!(superseded_claim_reason(LOCAL, &ack), None);

    // The peer that joined late wins the election and claims this node for a
    // stage of its own topology.
    assert!(send_claim(&control, claim(NEWER, 12, "b")).await.accepted);

    let ack = send_claim(&control, ours).await;
    assert!(!ack.accepted);
    assert_eq!(ack.claim.coordinator_id, NEWER);
    let reason = superseded_claim_reason(LOCAL, &ack).expect("the takeover is seen");
    assert!(reason.contains(NEWER), "{reason}");
    assert!(
        reason.contains("stale coordinator term 11 < 12"),
        "{reason}"
    );
    control.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_failed_cutover_of_our_own_is_not_a_takeover() {
    let control = skippy::spawn_stage_control_loop(skippy::SkippyTelemetryOptions::default());
    let active = claim(LOCAL, 11, "a");
    assert!(send_claim(&control, active.clone()).await.accepted);
    // An auto-balance cutover claims the stages under a newer term, then fails
    // to load and leaves the active generation serving.
    assert!(send_claim(&control, claim(LOCAL, 12, "a2")).await.accepted);

    let ack = send_claim(&control, active).await;
    assert!(!ack.accepted);
    assert_eq!(superseded_claim_reason(LOCAL, &ack), None);
    control.shutdown().await.unwrap();
}

#[test]
fn a_claim_rejected_without_a_holder_is_not_a_takeover() {
    // An expired or malformed claim comes back with nothing holding the
    // model; the ack then echoes our own claim.
    let ack = skippy::StageCoordinatorClaimAck {
        accepted: false,
        claim: claim(LOCAL, 11, "a"),
        error: Some("coordinator claim lease is expired".to_string()),
    };
    assert_eq!(superseded_claim_reason(LOCAL, &ack), None);
}

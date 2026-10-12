//! Request-in admission, unchanged streaming, response-out settlement.
//! Deliberately independent of Skippy's legacy generation gate.
use super::request::PaidRequest;
use crate::{
    inference::election::{InferenceTarget, ModelTargets},
    mesh::Node,
};
use anyhow::{Context, Result, bail, ensure};
use mesh_llm_payments::{
    ledger::metered::{MeteredOutcome, MeteredRequest, MeteredStatus},
    service::PaymentService,
};
use mesh_llm_payments_types::{
    metered_wire::{self as wire, AdmissionTerms, Frame},
    metering::PromptEstimate,
    response_usage::{ResponseEvidence, ResponseUsage},
};
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};

/// Fixed seller safety policy for this slice, separate from buyer authorization.
const PEER_EXPOSURE_MSAT: u64 = 1_000_000;
const MAX_PROMPT_ESTIMATE: u64 = 131_072;

struct Session {
    service: Arc<PaymentService>,
    peer: String,
    terms: AdmissionTerms,
    request: PaidRequest,
    backend: super::metered_backend::Backend,
}

pub(crate) async fn serve(
    node: Node,
    peer: iroh::EndpointId,
    mut reader: impl AsyncRead + Unpin,
    mut writer: impl AsyncWrite + Unpin,
    targets: ModelTargets,
) -> Result<()> {
    let result = serve_inner(&node, peer, &mut reader, &mut writer, &targets).await;
    if result.is_err() {
        let _ = wire::write(
            &mut writer,
            &Frame::Error {
                message: "metered exchange interrupted; recover by request ID".into(),
            },
        )
        .await;
    }
    result
}

async fn serve_inner(
    node: &Node,
    peer: iroh::EndpointId,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
    targets: &ModelTargets,
) -> Result<()> {
    let initial = tokio::time::timeout(Duration::from_secs(10), wire::read(reader)).await??;
    let service = node.payment_service().await?;
    let peer = peer.to_string();
    if let Frame::Recover { id } = initial {
        return recover(&service, &id, &peer, writer).await;
    }
    if let Frame::Request { ref id, .. } = initial
        && service.ledger.metered_owner(id)?.is_some()
    {
        return recover(&service, id, &peer, writer).await;
    }
    let session = admit(node, service, peer, initial, targets).await?;
    let result = serve_admitted(node, &session, reader, writer).await;
    if result.is_err()
        && session
            .service
            .ledger
            .metered_settlement(&session.terms.id)?
            .status
            == MeteredStatus::Reserved
    {
        let dispatched = session
            .service
            .ledger
            .metered_owner(&session.terms.id)?
            .context("unknown metered request")?
            .dispatched;
        session.service.ledger.metered_finalize(
            &session.terms.id,
            if dispatched {
                MeteredOutcome::UsageMissing
            } else {
                MeteredOutcome::BackendError
            },
        )?;
    }
    result
}

async fn serve_admitted(
    node: &Node,
    session: &Session,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
) -> Result<()> {
    let deposit = session
        .service
        .metered_invoice(
            &session.terms.id,
            &session.peer,
            0,
            session.terms.deposit_msat,
        )
        .await?;
    wire::write(
        writer,
        &Frame::Admission {
            terms: session.terms.clone(),
            invoice: deposit.clone(),
        },
    )
    .await?;
    let accepted = tokio::time::timeout(Duration::from_secs(60), wire::read(reader)).await??;
    if !matches!(accepted, Frame::Accepted {}) {
        // No dispatch; keep any uncertain payment for authoritative refresh.
        session.service.metered_refresh(&session.terms.id).await?;
        let owner = session
            .service
            .ledger
            .metered_owner(&session.terms.id)?
            .context("unknown metered request")?;
        if !owner.dispatched {
            session
                .service
                .ledger
                .metered_finalize(&session.terms.id, MeteredOutcome::BackendError)?;
        }
        bail!("admission cancelled");
    }
    if let Some(invoice) = deposit {
        session.service.wait_received(&invoice).await?;
        session
            .service
            .ledger
            .mark_received(&invoice.payment_hash)?;
    }
    if !session
        .service
        .ledger
        .metered_claim_dispatch(&session.terms.id, &session.peer)?
    {
        return recover(&session.service, &session.terms.id, &session.peer, writer).await;
    }
    let _instance = if let super::metered_backend::Backend::Local(port) = session.backend {
        Some(node.begin_runtime_instance_request(port).await?)
    } else {
        None
    };
    let outcome = run_backend(session, reader, writer).await;
    drop(_instance);
    session
        .service
        .ledger
        .metered_finalize(&session.terms.id, outcome)?;
    settle(&session.service, &session.terms.id, &session.peer, writer).await
}

async fn admit(
    node: &Node,
    service: Arc<PaymentService>,
    peer: String,
    frame: Frame,
    targets: &ModelTargets,
) -> Result<Session> {
    let Frame::Request {
        id,
        model,
        pricing,
        max_total_msat,
        http,
    } = frame
    else {
        bail!("expected metered request")
    };
    uuid::Uuid::parse_str(&id)?;
    ensure!(
        service.ledger.pricing()?.get(&model) == Some(&pricing),
        "seller prices changed"
    );
    let mut request = PaidRequest::parse(&http)?;
    ensure!(request.model == model, "model mismatch");
    let estimate = PromptEstimate::from_request(&request.body)?;
    ensure!(
        !estimate.has_media && estimate.tokens <= MAX_PROMPT_ESTIMATE,
        "unsupported or oversized prompt"
    );
    // No tokenize RPC or silent output clamp: caller supplies the finite ceiling.
    let max_output_tokens = u64::from(
        request
            .max_tokens
            .context("metered inference requires explicit max_tokens")?,
    );
    ensure!(
        max_output_tokens <= 131_072,
        "output allowance exceeds seller limit"
    );
    let reservation = pricing
        .input_charge(estimate.tokens)?
        .checked_add(pricing.output_charge(max_output_tokens)?)
        .context("reservation overflow")?;
    ensure!(
        max_total_msat
            >= mesh_llm_payments::pricing::payment_cap_msat(reservation)?
                .checked_add(mesh_llm_payments::pricing::fee_allowance_msat(reservation)?)
                .context("cap overflow")?,
        "buyer cap below estimated exposure plus fees"
    );
    let internal =
        crate::network::openai::ingress::served_model_for_public_id(node, targets, &model).await;
    let backend = match targets.candidates(&internal).iter().find_map(|target| {
        if let InferenceTarget::Local(port) = target {
            Some(*port)
        } else {
            None
        }
    }) {
        Some(port) => super::metered_backend::Backend::Local(port),
        None => super::metered_backend::resolve_plugin(node, &model).await?,
    };
    request.use_backend_model(&internal);
    let admission = service.ledger.metered_admit(&MeteredRequest {
        id: &id,
        peer: &peer,
        pricing: &pricing,
        estimated_input_tokens: estimate.tokens,
        max_output_tokens,
        peer_cap_msat: PEER_EXPOSURE_MSAT,
        require_deposit: !service.metered_trusted(&peer).await?,
    })?;
    let terms = AdmissionTerms {
        id,
        model,
        pricing,
        prompt_estimate: estimate.tokens,
        max_output_tokens,
        max_total_msat,
        deposit_msat: admission.deposit_msat,
        credit_msat: service.ledger.peer_credit(&peer)?,
    };
    Ok(Session {
        service,
        peer,
        terms,
        request,
        backend,
    })
}

async fn run_backend(
    session: &Session,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
) -> MeteredOutcome {
    let backend = match session
        .backend
        .connect(&session.request, &session.terms.id)
        .await
    {
        Ok(backend) => backend,
        Err(failure) => return failure.outcome(),
    };
    forward_backend(backend, reader, writer).await
}

/// Observe the backend independently of buyer transport completion.
async fn forward_backend(
    mut backend: impl AsyncRead + Unpin,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
) -> MeteredOutcome {
    let mut usage = ResponseUsage::default();
    let mut buffer = [0_u8; 16 * 1024];
    let cancelled = wait_for_cancel_or_invalid_frame(reader);
    tokio::pin!(cancelled);
    loop {
        let read = tokio::select! {
            read = tokio::time::timeout(Duration::from_secs(300), backend.read(&mut buffer)) => read,
            _ = &mut cancelled => return evidence_outcome(usage.interrupted_evidence()),
        };
        match read {
            Ok(Ok(0)) => break,
            Ok(Ok(count)) => {
                usage.observe(&buffer[..count]);
                if wire::write(
                    writer,
                    &Frame::Output {
                        bytes: buffer[..count].to_vec(),
                    },
                )
                .await
                .is_err()
                {
                    return evidence_outcome(usage.interrupted_evidence());
                }
            }
            _ => return evidence_outcome(usage.interrupted_evidence()),
        }
    }
    usage.finish();
    evidence_outcome(usage.evidence())
}

fn evidence_outcome(evidence: ResponseEvidence) -> MeteredOutcome {
    match evidence {
        ResponseEvidence::Reported(usage) => MeteredOutcome::Usage(usage),
        ResponseEvidence::BackendError => MeteredOutcome::BackendError,
        ResponseEvidence::Missing => MeteredOutcome::UsageMissing,
    }
}

/// A clean send-half EOF is not cancellation: the buyer may still consume
/// the response. Invalid/truncated frames terminate conservatively, rather
/// than pretending the buyer sent Cancel. The response writer detects loss
/// of the receive half independently.
async fn wait_for_cancel_or_invalid_frame(reader: &mut (impl AsyncRead + Unpin)) -> Result<()> {
    let mut prefix = [0_u8; 1];
    match reader.read(&mut prefix).await {
        Ok(0) => std::future::pending::<Result<()>>().await,
        Ok(_) => {
            let mut framed = std::io::Cursor::new(prefix).chain(reader);
            ensure!(
                matches!(wire::read(&mut framed).await?, Frame::Cancel {}),
                "unexpected buyer frame during inference"
            );
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

async fn settle(
    service: &PaymentService,
    id: &str,
    peer: &str,
    writer: &mut (impl AsyncWrite + Unpin),
) -> Result<()> {
    let invoice = service.metered_final_invoice(id, peer).await?;
    let settlement = service.ledger.metered_settlement(id)?;
    wire::write(
        writer,
        &Frame::Settlement {
            request_id: id.into(),
            usage: settlement.usage,
            bill_msat: settlement.bill_msat,
            deposit_applied_msat: if settlement.usage.is_some() {
                settlement
                    .deposit_paid_msat
                    .saturating_sub(settlement.credit_created_msat)
            } else {
                0
            },
            credit_applied_msat: settlement.credit_applied_msat,
            credit_remaining_msat: service.ledger.peer_credit(peer)?,
            invoice: invoice.clone(),
        },
    )
    .await?;
    if settlement.status == MeteredStatus::UsageMissing {
        return wire::write(writer, &Frame::Pending {}).await;
    }
    if let Some(invoice) = invoice {
        service.wait_received(&invoice).await?;
        service.ledger.mark_received(&invoice.payment_hash)?;
    }
    service.metered_note_settled(id, peer).await?;
    wire::write(writer, &Frame::Complete {}).await
}

async fn recover(
    service: &PaymentService,
    id: &str,
    peer: &str,
    writer: &mut (impl AsyncWrite + Unpin),
) -> Result<()> {
    let owner = service
        .ledger
        .metered_owner(id)?
        .context("unknown metered request")?;
    ensure!(owner.peer == peer, "recovery peer mismatch");
    service.metered_refresh(id).await?;
    if service.ledger.metered_settlement(id)?.status == MeteredStatus::Reserved {
        return wire::write(writer, &Frame::Pending {}).await;
    }
    settle(service, id, peer, writer).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::Poll;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn send_half_eof_is_not_cancel_but_explicit_cancel_is() -> Result<()> {
        let mut eof = tokio::io::empty();
        let watcher = wait_for_cancel_or_invalid_frame(&mut eof);
        tokio::pin!(watcher);
        std::future::poll_fn(|cx| {
            assert!(matches!(
                std::future::Future::poll(watcher.as_mut(), cx),
                Poll::Pending
            ));
            Poll::Ready(())
        })
        .await;
        let mut bytes = Vec::new();
        wire::write(&mut bytes, &Frame::Cancel {}).await?;
        wait_for_cancel_or_invalid_frame(&mut bytes.as_slice()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn truncated_and_unexpected_buyer_frames_are_errors_not_cancel() -> Result<()> {
        let mut bytes = Vec::new();
        wire::write(&mut bytes, &Frame::Accepted {}).await?;
        assert!(
            wait_for_cancel_or_invalid_frame(&mut bytes.as_slice())
                .await
                .is_err()
        );
        let (mut send, mut recv) = tokio::io::duplex(8);
        send.write_all(&[0]).await?;
        send.shutdown().await?;
        assert!(wait_for_cancel_or_invalid_frame(&mut recv).await.is_err());
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/metered_abort.rs"]
mod abort_tests;

//! Metered (v2) provider ledger: exposure, deposit credit, finalize outcomes.
use super::invoice;
use crate::ledger::Ledger;
use crate::ledger::metered::{MeteredAdmission, MeteredOutcome, MeteredRequest, MeteredStatus};
use crate::ledger::receivables::Receivable;
use crate::pricing::Pricing;
use mesh_llm_payments_types::metering::MeteredUsage;

// 1 msat per token both ways keeps arithmetic readable.
fn rates() -> Pricing {
    Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        minimum_invoice_msat: 1,
    }
}

fn usage(prompt: u64, completion: u64) -> MeteredOutcome {
    MeteredOutcome::Usage(MeteredUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
    })
}

fn record(ledger: &Ledger, id: &str, segment: u32, n: u8, amount: u64, paid: bool) {
    let inv = invoice(n, amount);
    ledger
        .record_receivable(&Receivable {
            request_id: id.into(),
            peer: "peer".into(),
            segment,
            invoice: inv.clone(),
            tokens: 0,
            paid: false,
        })
        .unwrap();
    if paid {
        ledger.mark_received(&inv.payment_hash).unwrap();
    }
}

fn req(id: &str, input: u64, output: u64, cap: u64, deposit: bool) -> MeteredRequest<'_> {
    static RATES: std::sync::OnceLock<Pricing> = std::sync::OnceLock::new();
    MeteredRequest {
        id,
        peer: "peer",
        pricing: RATES.get_or_init(rates),
        estimated_input_tokens: input,
        max_output_tokens: output,
        peer_cap_msat: cap,
        require_deposit: deposit,
    }
}

fn admit(
    l: &Ledger,
    id: &str,
    input: u64,
    output: u64,
    cap: u64,
    deposit: bool,
) -> anyhow::Result<MeteredAdmission> {
    l.metered_admit(&req(id, input, output, cap, deposit))
}

fn open() -> (tempfile::TempDir, Ledger) {
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(dir.path()).unwrap();
    (dir, ledger)
}

#[test]
fn deposit_is_three_tenths_of_estimated_input_and_reservation_is_capped() {
    let (_d, l) = open();
    let a = l.metered_admit(&req("a", 1000, 500, 2000, true)).unwrap();
    assert_eq!((a.reserved_msat, a.deposit_msat), (1500, 300));
    // Idempotent replay; differing terms rejected.
    assert_eq!(admit(&l, "a", 1000, 500, 2000, true).unwrap(), a);
    assert!(admit(&l, "a", 1000, 501, 2000, true).is_err());
    // Concurrent exposure counts: 1500 + 600 > 2000.
    assert!(admit(&l, "b", 100, 500, 2000, false).is_err());
    let b = l.metered_admit(&req("b", 100, 400, 2000, false)).unwrap();
    assert_eq!(b.deposit_msat, 0);
    assert_eq!(l.metered_exposure("peer").unwrap(), 2000);
    assert!(admit(&l, "c", 10, 0, u64::MAX, false).is_err());
}

#[test]
fn usage_replaces_reservation_with_unpaid_bill_until_final_invoice_paid() {
    let dir = tempfile::tempdir().unwrap();
    {
        let l = Ledger::open(dir.path()).unwrap();
        admit(&l, "r", 1000, 500, 10_000, true).unwrap();
        record(&l, "r", 0, 1, 300, true);
        let s = l.metered_finalize("r", usage(900, 200)).unwrap();
        assert_eq!(s.status, MeteredStatus::Billed);
        assert_eq!(
            (s.bill_msat, s.deposit_paid_msat, s.remaining_msat),
            (Some(1100), 300, 800)
        );
        assert_eq!(l.metered_exposure("peer").unwrap(), 800);
        // Replay returns the same settlement; a different outcome is refused.
        assert_eq!(l.metered_finalize("r", usage(900, 200)).unwrap(), s);
        assert!(
            l.metered_finalize("r", MeteredOutcome::BackendError)
                .is_err()
        );
    }
    // Restart: exposure is derived from durable rows.
    let l = Ledger::open(dir.path()).unwrap();
    assert_eq!(l.metered_exposure("peer").unwrap(), 800);
    record(&l, "r", 1, 2, 800, true);
    assert_eq!(
        l.metered_settlement("r").unwrap().status,
        MeteredStatus::Settled
    );
    assert_eq!(l.metered_exposure("peer").unwrap(), 0);
}

#[test]
fn overpaid_deposit_becomes_credit_and_is_applied_once_to_the_next_bill() {
    let (_d, l) = open();
    admit(&l, "a", 1000, 10, 10_000, true).unwrap();
    record(&l, "a", 0, 1, 300, true);
    let s = l.metered_finalize("a", usage(100, 50)).unwrap();
    assert_eq!(
        (s.status, s.remaining_msat, s.credit_created_msat),
        (MeteredStatus::Settled, 0, 150)
    );
    l.metered_finalize("a", usage(100, 50)).unwrap();
    assert_eq!(l.peer_credit("peer").unwrap(), 150);

    admit(&l, "b", 10, 10, 10_000, false).unwrap();
    let s = l.metered_finalize("b", usage(100, 100)).unwrap();
    assert_eq!((s.credit_applied_msat, s.remaining_msat), (150, 50));
    l.metered_finalize("b", usage(100, 100)).unwrap();
    assert_eq!(l.peer_credit("peer").unwrap(), 0);
}

#[test]
fn backend_error_refunds_paid_deposit_as_credit_and_releases_exposure() {
    let (_d, l) = open();
    admit(&l, "e", 1000, 500, 10_000, true).unwrap();
    record(&l, "e", 0, 1, 300, true);
    let s = l
        .metered_finalize("e", MeteredOutcome::BackendError)
        .unwrap();
    assert_eq!(
        (s.status, s.bill_msat, s.credit_created_msat),
        (MeteredStatus::BackendError, None, 300)
    );
    assert_eq!(l.metered_exposure("peer").unwrap(), 0);
    assert_eq!(l.peer_credit("peer").unwrap(), 300);
    // Released before the deposit was paid: nothing credited.
    admit(&l, "u", 1000, 500, 10_000, true).unwrap();
    let s = l
        .metered_finalize("u", MeteredOutcome::BackendError)
        .unwrap();
    assert_eq!(s.credit_created_msat, 0);
    assert_eq!(l.peer_credit("peer").unwrap(), 300);
}

#[test]
fn usage_missing_stays_unresolved_without_bill_forfeit_or_credit() {
    let (_d, l) = open();
    admit(&l, "m", 1000, 500, 10_000, true).unwrap();
    record(&l, "m", 0, 1, 300, true);
    let s = l
        .metered_finalize("m", MeteredOutcome::UsageMissing)
        .unwrap();
    assert_eq!(s.status, MeteredStatus::UsageMissing);
    assert_eq!(
        (
            s.bill_msat,
            s.usage,
            s.credit_created_msat,
            s.remaining_msat
        ),
        (None, None, 0, 0)
    );
    assert_eq!(s.deposit_paid_msat, 300);
    assert_eq!(l.metered_exposure("peer").unwrap(), 1500);
    assert_eq!(l.peer_credit("peer").unwrap(), 0);
}

#[test]
fn metered_rows_are_invisible_to_legacy_recovery_and_debt_blocking() {
    let (_d, l) = open();
    admit(&l, "x", 1000, 500, 10_000, true).unwrap();
    record(&l, "x", 0, 1, 300, false);
    assert!(l.unpaid_input_batch(0).unwrap().is_empty());
    assert!(l.unpaid_input_requests("peer").unwrap().is_empty());
    l.metered_finalize("x", usage(1000, 100)).unwrap();
    assert!(l.uninvoiced_output().unwrap().is_empty());
}

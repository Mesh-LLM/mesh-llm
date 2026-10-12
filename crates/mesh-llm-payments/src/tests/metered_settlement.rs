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

#[test]
fn usage_missing_resolves_once_from_later_authoritative_usage() {
    let (_d, l) = open();
    admit(&l, "m", 1000, 500, 10_000, true).unwrap();
    record(&l, "m", 0, 1, 300, true);
    l.metered_finalize("m", MeteredOutcome::UsageMissing)
        .unwrap();
    // Replaying missing is a no-op; authoritative usage resolves it.
    l.metered_finalize("m", MeteredOutcome::UsageMissing)
        .unwrap();
    let s = l.metered_finalize("m", usage(900, 200)).unwrap();
    assert_eq!((s.status, s.remaining_msat), (MeteredStatus::Billed, 800));
    assert_eq!(l.metered_finalize("m", usage(900, 200)).unwrap(), s);
    assert!(
        l.metered_finalize("m", MeteredOutcome::UsageMissing)
            .is_err()
    );
    assert!(
        l.metered_finalize("m", MeteredOutcome::BackendError)
            .is_err()
    );
    assert_eq!(l.metered_exposure("peer").unwrap(), 800);
    // Backend-error finding also resolves a missing row, crediting the deposit.
    admit(&l, "n", 1000, 500, 10_000, true).unwrap();
    record(&l, "n", 0, 2, 300, true);
    l.metered_finalize("n", MeteredOutcome::UsageMissing)
        .unwrap();
    let s = l
        .metered_finalize("n", MeteredOutcome::BackendError)
        .unwrap();
    assert_eq!(s.credit_created_msat, 300);
    assert_eq!(l.metered_exposure("peer").unwrap(), 800);
}

#[test]
fn unpaid_final_invoice_counts_once_and_survives_unblock() {
    let (_d, l) = open();
    admit(&l, "f", 1000, 500, 10_000, false).unwrap();
    l.metered_finalize("f", usage(900, 200)).unwrap();
    record(&l, "f", 1, 3, 1100, false);
    // Recorded-but-unpaid final invoice: remaining only, no reservation.
    assert_eq!(l.metered_exposure("peer").unwrap(), 1100);
    let blocked = l.blocked_peers().unwrap();
    assert_eq!(blocked[0].debts[0].kind, "metered_invoice");
    assert!(l.has_outstanding_payment("peer").unwrap());
    assert!(l.unblock_peer("peer").is_err());
    assert_eq!(l.metered_exposure("peer").unwrap(), 1100);
    assert!(l.has_outstanding_payment("peer").unwrap());
    assert!(!l.receivables(Some("f")).unwrap()[0].paid);
}

#[test]
fn admission_identity_is_exact_token_terms_not_rounded_amounts() {
    let (_d, l) = open();
    let low = Pricing {
        input_msat_per_million: 1,
        output_msat_per_million: 1,
        minimum_invoice_msat: 1,
    };
    let mk = |input, output| MeteredRequest {
        id: "low",
        peer: "peer",
        pricing: &low,
        estimated_input_tokens: input,
        max_output_tokens: output,
        peer_cap_msat: 10_000,
        require_deposit: true,
    };
    let a = l.metered_admit(&mk(10, 10)).unwrap();
    // Same rounded charge, different token terms.
    assert!(l.metered_admit(&mk(11, 12)).is_err());
    assert_eq!(l.metered_admit(&mk(10, 10)).unwrap(), a);
    assert!(admit(&l, "d", 10, 10, 10_000, true).is_ok());
    assert!(admit(&l, "d", 10, 10, 10_000, false).is_err());
}

#[test]
fn late_deposit_after_finalize_is_applied_exactly_once() {
    let (_d, l) = open();
    let paid = |n: u8, amount| {
        l.mark_received(&invoice(n, amount).payment_hash).unwrap();
    };
    // BackendError: late deposit becomes credit.
    admit(&l, "be", 1000, 500, 100_000, true).unwrap();
    record(&l, "be", 0, 10, 300, false);
    l.metered_finalize("be", MeteredOutcome::BackendError)
        .unwrap();
    assert_eq!(l.peer_credit("peer").unwrap(), 0);
    paid(10, 300);
    paid(10, 300);
    assert_eq!(l.peer_credit("peer").unwrap(), 300);
    let s = l.metered_settlement("be").unwrap();
    assert_eq!((s.deposit_paid_msat, s.credit_created_msat), (300, 300));
    // Usage, no final invoice yet: late deposit reduces remaining.
    admit(&l, "u", 1000, 500, 100_000, true).unwrap();
    record(&l, "u", 0, 11, 300, false);
    let s = l.metered_finalize("u", usage(900, 200)).unwrap();
    // Existing 300 credit applied at finalize; deposit still unpaid.
    assert_eq!((s.remaining_msat, s.credit_applied_msat), (800, 300));
    paid(11, 300);
    paid(11, 300);
    let s = l.metered_settlement("u").unwrap();
    assert_eq!((s.remaining_msat, s.deposit_paid_msat), (500, 300));
    assert_eq!(l.peer_credit("peer").unwrap(), 0);
    // Usage with final invoice already recorded: deposit becomes credit.
    admit(&l, "f", 1000, 500, 100_000, true).unwrap();
    record(&l, "f", 0, 12, 300, false);
    l.metered_finalize("f", usage(900, 200)).unwrap();
    record(&l, "f", 1, 13, 1100, false);
    assert_eq!(l.peer_credit("peer").unwrap(), 0);
    paid(12, 300);
    assert_eq!(l.peer_credit("peer").unwrap(), 300);
    assert_eq!(l.metered_settlement("f").unwrap().remaining_msat, 1100);
    // UsageMissing: untouched until resolved, which then counts the deposit.
    admit(&l, "m", 1000, 500, 100_000, true).unwrap();
    record(&l, "m", 0, 14, 300, false);
    l.metered_finalize("m", MeteredOutcome::UsageMissing)
        .unwrap();
    paid(14, 300);
    assert_eq!(l.metered_settlement("m").unwrap().deposit_paid_msat, 0);
    let s = l.metered_finalize("m", usage(900, 200)).unwrap();
    // Deposit counted (300) and the 300 credit from "f" applied.
    assert_eq!((s.deposit_paid_msat, s.credit_applied_msat), (300, 300));
    assert_eq!(s.remaining_msat, 500);
}

#[test]
fn dispatch_claim_is_once_only_and_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let l = Ledger::open(dir.path()).unwrap();
        admit(&l, "c", 1000, 500, 10_000, false).unwrap();
        assert!(l.metered_claim_dispatch("c", "other").is_err());
        assert!(l.metered_claim_dispatch("c", "peer").unwrap());
        // Admission replay does not re-authorize dispatch.
        admit(&l, "c", 1000, 500, 10_000, false).unwrap();
        assert!(!l.metered_claim_dispatch("c", "peer").unwrap());
        admit(&l, "n", 1000, 500, 10_000, false).unwrap();
        l.metered_finalize("n", MeteredOutcome::BackendError)
            .unwrap();
        assert!(l.metered_claim_dispatch("n", "peer").is_err());
        assert!(l.metered_claim_dispatch("zz", "peer").is_err());
    }
    let l = Ledger::open(dir.path()).unwrap();
    assert!(!l.metered_claim_dispatch("c", "peer").unwrap());
    let o = l.metered_owner("c").unwrap().unwrap();
    assert_eq!((o.peer.as_str(), o.dispatched), ("peer", true));
    assert!(!l.metered_owner("n").unwrap().unwrap().dispatched);
    assert!(l.metered_owner("zz").unwrap().is_none());
}

fn try_record(l: &Ledger, id: &str, segment: u32, n: u8, amount: u64) -> anyhow::Result<()> {
    l.record_receivable(&Receivable {
        request_id: id.into(),
        peer: "peer".into(),
        segment,
        invoice: invoice(n, amount),
        tokens: 0,
        paid: false,
    })
}

#[test]
fn metered_receivables_must_match_current_state() {
    let (_d, l) = open();
    admit(&l, "s", 1000, 500, 100_000, true).unwrap();
    // Deposit must equal the admitted deposit.
    assert!(try_record(&l, "s", 0, 20, 299).is_err());
    assert!(
        try_record(&l, "s", 1, 21, 1100).is_err(),
        "no final before bill"
    );
    try_record(&l, "s", 0, 22, 300).unwrap();
    let snapshot = l.metered_finalize("s", usage(900, 200)).unwrap();
    assert_eq!(snapshot.remaining_msat, 1100);
    // Late deposit lands between the settlement read and invoice persistence.
    l.mark_received(&invoice(22, 300).payment_hash).unwrap();
    assert!(
        try_record(&l, "s", 1, 23, snapshot.remaining_msat).is_err(),
        "stale snapshot refused"
    );
    let current = l.metered_settlement("s").unwrap().remaining_msat;
    assert_eq!(current, 800);
    try_record(&l, "s", 1, 23, current).unwrap();
    assert_eq!(l.metered_exposure("peer").unwrap(), 800);
    assert_eq!(l.peer_credit("peer").unwrap(), 0);
    // Not-billed states and no-deposit rows cannot carry invoices.
    admit(&l, "n", 1000, 500, 100_000, false).unwrap();
    assert!(try_record(&l, "n", 0, 24, 0).is_err());
    l.metered_finalize("n", MeteredOutcome::BackendError)
        .unwrap();
    assert!(try_record(&l, "n", 1, 25, 1).is_err());
    assert!(try_record(&l, "n", 2, 26, 1).is_err());
}

#[test]
fn startup_closes_interrupted_rows_by_dispatch_marker() {
    let dir = tempfile::tempdir().unwrap();
    {
        let l = Ledger::open(dir.path()).unwrap();
        admit(&l, "d", 1000, 500, 100_000, false).unwrap();
        assert!(l.metered_claim_dispatch("d", "peer").unwrap());
        admit(&l, "u", 1000, 500, 100_000, true).unwrap();
        record(&l, "u", 0, 30, 300, false);
        admit(&l, "p", 1000, 500, 100_000, true).unwrap();
        record(&l, "p", 0, 31, 300, true);
    }
    let l = Ledger::open(dir.path()).unwrap();
    l.close_interrupted_metered().unwrap();
    l.close_interrupted_metered().unwrap();
    // Dispatched: usage unknown, reservation held, no refund.
    assert_eq!(
        l.metered_settlement("d").unwrap().status,
        MeteredStatus::UsageMissing
    );
    assert_eq!(l.metered_exposure("peer").unwrap(), 1500);
    // Never dispatched: paid deposit refunded now, unpaid one on arrival.
    assert_eq!(
        l.metered_settlement("p").unwrap().status,
        MeteredStatus::BackendError
    );
    assert_eq!(l.peer_credit("peer").unwrap(), 300);
    l.mark_received(&invoice(30, 300).payment_hash).unwrap();
    assert_eq!(l.peer_credit("peer").unwrap(), 600);
}

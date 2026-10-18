//! Metered provider ledger: deposit, finalize outcomes, unpaid-final strikes.
use super::invoice;
use crate::ledger::Ledger;
use crate::ledger::metered::{MeteredOutcome, MeteredRequest, MeteredStatus, STRIKE_LIMIT};
use crate::ledger::metered_invoices::{MeteredInvoice, MeteredInvoiceKind};
use crate::pricing::Pricing;
use mesh_llm_payments_types::metering::MeteredUsage;

// 10 msat per token both ways; the deposit is the flat 1000 msat.
fn rates() -> Pricing {
    Pricing {
        input_msat_per_million: 10_000_000,
        output_msat_per_million: 10_000_000,
        cache_discount_percent: 0,
    }
}

fn usage(prompt: u64, completion: u64) -> MeteredOutcome {
    MeteredOutcome::Usage(MeteredUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        cached_tokens: 0,
    })
}

fn record(ledger: &Ledger, id: &str, segment: u32, n: u8, amount: u64, paid: bool) {
    let inv = invoice(n, amount);
    ledger
        .record_metered_invoice(&MeteredInvoice {
            request_id: id.into(),
            peer: "peer".into(),
            kind: if segment == 0 {
                MeteredInvoiceKind::Deposit
            } else {
                MeteredInvoiceKind::Final
            },
            invoice: inv.clone(),
            paid: false,
        })
        .unwrap();
    if paid {
        ledger.mark_metered_received(&inv.payment_hash).unwrap();
    }
}

fn req(id: &str, deposit: bool) -> MeteredRequest<'_> {
    static RATES: std::sync::OnceLock<Pricing> = std::sync::OnceLock::new();
    MeteredRequest {
        id,
        peer: "peer",
        pricing: RATES.get_or_init(rates),
        require_deposit: deposit,
    }
}

fn admit(l: &Ledger, id: &str, deposit: bool) -> anyhow::Result<u64> {
    l.metered_admit(&req(id, deposit))
}

fn open() -> (tempfile::TempDir, Ledger) {
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(dir.path()).unwrap();
    (dir, ledger)
}

#[test]
fn deposit_is_flat_and_admission_is_idempotent() {
    let (_d, l) = open();
    assert_eq!(admit(&l, "a", true).unwrap(), 1000);
    // Idempotent replay; differing terms rejected.
    assert_eq!(admit(&l, "a", true).unwrap(), 1000);
    assert!(admit(&l, "a", false).is_err());
    // Trusted peers owe no deposit; no cap or reservation is involved.
    assert_eq!(admit(&l, "b", false).unwrap(), 0);
    assert_eq!(
        l.metered_settlement("b").unwrap().status,
        MeteredStatus::Admitted
    );
}

#[test]
fn usage_bills_input_plus_output_minus_paid_deposit_until_final_paid() {
    let dir = tempfile::tempdir().unwrap();
    {
        let l = Ledger::open(dir.path()).unwrap();
        admit(&l, "r", true).unwrap();
        record(&l, "r", 0, 1, 1000, true);
        let s = l.metered_finalize("r", usage(900, 200)).unwrap();
        assert_eq!(s.status, MeteredStatus::Billed);
        assert_eq!(
            (s.bill_msat, s.deposit_paid_msat, s.remaining_msat),
            (Some(11000), 1000, 10000)
        );
        // Replay returns the same settlement; a different outcome is refused.
        assert_eq!(l.metered_finalize("r", usage(900, 200)).unwrap(), s);
        assert!(l.metered_finalize("r", usage(900, 201)).is_err());
        assert!(l.metered_finalize("r", MeteredOutcome::Abandoned).is_err());
    }
    // Restart: the bill is durable and the final invoice settles it.
    let l = Ledger::open(dir.path()).unwrap();
    assert_eq!(l.metered_settlement("r").unwrap().remaining_msat, 10000);
    record(&l, "r", 1, 2, 10000, true);
    assert_eq!(
        l.metered_settlement("r").unwrap().status,
        MeteredStatus::Settled
    );
}

#[test]
fn covering_deposit_means_no_final_invoice_and_no_refund() {
    let (_d, l) = open();
    admit(&l, "a", true).unwrap();
    record(&l, "a", 0, 1, 1000, true);
    let s = l.metered_finalize("a", usage(60, 40)).unwrap();
    assert_eq!(
        (s.status, s.bill_msat, s.deposit_paid_msat, s.remaining_msat),
        (MeteredStatus::Settled, Some(1000), 1000, 0)
    );
    // Nothing is carried to the next request.
    admit(&l, "b", false).unwrap();
    let s = l.metered_finalize("b", usage(100, 100)).unwrap();
    assert_eq!((s.deposit_paid_msat, s.remaining_msat), (0, 2000));
}

#[test]
fn every_positive_sub_sat_remainder_gets_a_final_invoice() {
    let (_d, l) = open();
    admit(&l, "small", false).unwrap();
    let s = l.metered_finalize("small", usage(1, 0)).unwrap();
    assert_eq!((s.bill_msat, s.remaining_msat), (Some(10), 1000));
    assert_eq!(s.status, MeteredStatus::Billed);
    admit(&l, "deposit", true).unwrap();
    record(&l, "deposit", 0, 80, 1000, true);
    let s = l.metered_finalize("deposit", usage(100, 1)).unwrap();
    assert_eq!((s.bill_msat, s.remaining_msat), (Some(1010), 1000));
    assert_eq!(s.status, MeteredStatus::Billed);
}

#[test]
fn unpaid_deposit_never_reduces_the_bill() {
    let (_d, l) = open();
    admit(&l, "u", true).unwrap();
    record(&l, "u", 0, 1, 1000, false);
    let s = l.metered_finalize("u", usage(900, 200)).unwrap();
    assert_eq!((s.deposit_paid_msat, s.remaining_msat), (0, 11000));
    // A deposit arriving after finalize changes nothing already fixed.
    l.mark_metered_received(&invoice(1, 300).payment_hash)
        .unwrap();
    assert_eq!(l.metered_settlement("u").unwrap().remaining_msat, 11000);
}

#[test]
fn abandoned_requests_have_no_bill_no_debt_and_keep_a_received_deposit() {
    let (_d, l) = open();
    admit(&l, "e", true).unwrap();
    record(&l, "e", 0, 1, 1000, true);
    let s = l.metered_finalize("e", MeteredOutcome::Abandoned).unwrap();
    assert_eq!(
        (s.status, s.bill_msat, s.usage, s.remaining_msat),
        (MeteredStatus::Abandoned, None, None, 0)
    );
    assert_eq!(s.deposit_paid_msat, 1000);
    // Terminal: later usage cannot bill an abandoned request.
    assert_eq!(
        l.metered_finalize("e", MeteredOutcome::Abandoned).unwrap(),
        s
    );
    assert!(l.metered_finalize("e", usage(900, 200)).is_err());
    assert_eq!(l.metered_unpaid_finals("peer").unwrap(), 0);
}

#[test]
fn three_issued_unpaid_final_invoices_block_admission() {
    let (_d, l) = open();
    for n in 0..STRIKE_LIMIT {
        let id = format!("s{n}");
        admit(&l, &id, false).unwrap();
        l.metered_finalize(&id, usage(100, 100)).unwrap();
        // A billed remainder without an issued invoice is not a strike.
        assert_eq!(l.metered_unpaid_finals("peer").unwrap(), n);
        record(&l, &id, 1, n as u8 + 1, 2000, false);
    }
    assert_eq!(l.metered_unpaid_finals("peer").unwrap(), STRIKE_LIMIT);
    assert!(admit(&l, "blocked", true).is_err());
    // Deposits and abandoned requests never count; paying one final unblocks.
    assert_eq!(l.metered_unpaid_finals("other").unwrap(), 0);
    l.mark_metered_received(&invoice(1, 20).payment_hash)
        .unwrap();
    assert_eq!(l.metered_unpaid_finals("peer").unwrap(), STRIKE_LIMIT - 1);
    admit(&l, "again", true).unwrap();
}

#[test]
fn admission_identity_is_the_exact_terms() {
    let (_d, l) = open();
    let low = Pricing {
        input_msat_per_million: 1,
        output_msat_per_million: 1,
        cache_discount_percent: 0,
    };
    let mk = |require_deposit| MeteredRequest {
        id: "low",
        peer: "peer",
        pricing: &low,
        require_deposit,
    };
    let a = l.metered_admit(&mk(true)).unwrap();
    assert!(l.metered_admit(&mk(false)).is_err());
    assert_eq!(l.metered_admit(&mk(true)).unwrap(), a);
}

#[test]
fn dispatch_claim_is_once_only_and_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let l = Ledger::open(dir.path()).unwrap();
        admit(&l, "c", false).unwrap();
        assert!(l.metered_claim_dispatch("c", "other").is_err());
        assert!(l.metered_claim_dispatch("c", "peer").unwrap());
        // Admission replay does not re-authorize dispatch.
        admit(&l, "c", false).unwrap();
        assert!(!l.metered_claim_dispatch("c", "peer").unwrap());
        admit(&l, "n", false).unwrap();
        l.metered_finalize("n", MeteredOutcome::Abandoned).unwrap();
        assert!(l.metered_claim_dispatch("n", "peer").is_err());
        assert!(l.metered_claim_dispatch("zz", "peer").is_err());
    }
    let l = Ledger::open(dir.path()).unwrap();
    assert!(!l.metered_claim_dispatch("c", "peer").unwrap());
    assert_eq!(l.metered_owner("c").unwrap().as_deref(), Some("peer"));
    assert!(l.metered_owner("zz").unwrap().is_none());
}

fn try_record(l: &Ledger, id: &str, segment: u32, n: u8, amount: u64) -> anyhow::Result<()> {
    l.record_metered_invoice(&MeteredInvoice {
        request_id: id.into(),
        peer: "peer".into(),
        kind: if segment == 0 {
            MeteredInvoiceKind::Deposit
        } else {
            MeteredInvoiceKind::Final
        },
        invoice: invoice(n, amount),
        paid: false,
    })
}

#[test]
fn metered_receivables_must_match_current_state() {
    let (_d, l) = open();
    admit(&l, "s", true).unwrap();
    // Deposit must equal the admitted deposit.
    assert!(try_record(&l, "s", 0, 20, 999).is_err());
    assert!(
        try_record(&l, "s", 1, 21, 11000).is_err(),
        "no final before bill"
    );
    try_record(&l, "s", 0, 22, 1000).unwrap();
    l.mark_metered_received(&invoice(22, 300).payment_hash)
        .unwrap();
    let s = l.metered_finalize("s", usage(900, 200)).unwrap();
    assert_eq!(s.remaining_msat, 10000);
    assert!(try_record(&l, "s", 1, 23, 11000).is_err());
    try_record(&l, "s", 1, 23, 10000).unwrap();
    // Abandoned and no-deposit rows cannot carry invoices.
    admit(&l, "n", false).unwrap();
    assert!(try_record(&l, "n", 0, 24, 0).is_err());
    l.metered_finalize("n", MeteredOutcome::Abandoned).unwrap();
    assert!(try_record(&l, "n", 1, 25, 1).is_err());
    assert!(try_record(&l, "n", 2, 26, 1).is_err());
}

#[test]
fn startup_abandons_interrupted_rows() {
    let dir = tempfile::tempdir().unwrap();
    {
        let l = Ledger::open(dir.path()).unwrap();
        admit(&l, "d", false).unwrap();
        assert!(l.metered_claim_dispatch("d", "peer").unwrap());
        admit(&l, "p", true).unwrap();
        record(&l, "p", 0, 31, 1000, true);
        admit(&l, "b", false).unwrap();
        l.metered_finalize("b", usage(100, 100)).unwrap();
    }
    let l = Ledger::open(dir.path()).unwrap();
    l.close_interrupted_metered().unwrap();
    l.close_interrupted_metered().unwrap();
    for id in ["d", "p"] {
        assert_eq!(
            l.metered_settlement(id).unwrap().status,
            MeteredStatus::Abandoned
        );
    }
    assert_eq!(l.metered_settlement("p").unwrap().deposit_paid_msat, 1000);
    assert_eq!(
        l.metered_settlement("b").unwrap().status,
        MeteredStatus::Billed
    );
}

#[test]
fn forgiving_a_peer_clears_its_strikes_but_a_late_payment_is_still_recorded() {
    let (_d, l) = open();
    for n in 0..STRIKE_LIMIT {
        let id = format!("s{n}");
        admit(&l, &id, false).unwrap();
        l.metered_finalize(&id, usage(100, 100)).unwrap();
        record(&l, &id, 1, n as u8 + 1, 2000, false);
    }
    assert!(admit(&l, "blocked", true).is_err());
    let listed = l.blocked_peers().unwrap();
    assert_eq!(listed.len(), 1);
    assert!(listed[0].blocked);
    assert_eq!(listed[0].debts.len(), STRIKE_LIMIT as usize);
    assert_eq!(listed[0].debts[0].amount_msat, Some(2000));
    assert_eq!(l.wallet_switch_blockers().unwrap().unpaid_invoices.len(), 3);
    assert!(l.unblock_peer("nobody").is_err());
    let forgiven = l.unblock_peer("peer").unwrap();
    assert_eq!(forgiven.invoices, STRIKE_LIMIT as usize);
    assert_eq!(l.metered_unpaid_finals("peer").unwrap(), 0);
    assert!(l.blocked_peers().unwrap().is_empty());
    assert!(
        l.wallet_switch_blockers()
            .unwrap()
            .unpaid_invoices
            .is_empty()
    );
    admit(&l, "again", true).unwrap();
    // Forgiven is not paid: the bill stays billed until a receipt turns up.
    assert_eq!(
        l.metered_settlement("s0").unwrap().status,
        MeteredStatus::Billed
    );
    l.mark_metered_received(&invoice(1, 20).payment_hash)
        .unwrap();
    assert_eq!(
        l.metered_settlement("s0").unwrap().status,
        MeteredStatus::Settled
    );
    assert!(l.metered_invoices(Some("s0")).unwrap()[0].paid);
}

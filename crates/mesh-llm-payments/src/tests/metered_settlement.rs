//! Metered provider ledger: deposit, finalize outcomes, unpaid-final strikes.
use super::invoice;
use crate::ledger::Ledger;
use crate::ledger::metered::{MeteredOutcome, MeteredRequest, MeteredStatus, STRIKE_LIMIT};
use crate::ledger::metered_invoices::{MeteredInvoice, MeteredInvoiceKind};
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

fn req(id: &str, input: u64, deposit: bool) -> MeteredRequest<'_> {
    static RATES: std::sync::OnceLock<Pricing> = std::sync::OnceLock::new();
    MeteredRequest {
        id,
        peer: "peer",
        pricing: RATES.get_or_init(rates),
        estimated_input_tokens: input,
        require_deposit: deposit,
    }
}

fn admit(l: &Ledger, id: &str, input: u64, deposit: bool) -> anyhow::Result<u64> {
    l.metered_admit(&req(id, input, deposit))
}

fn open() -> (tempfile::TempDir, Ledger) {
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(dir.path()).unwrap();
    (dir, ledger)
}

#[test]
fn deposit_is_three_tenths_of_estimated_input_and_admission_is_idempotent() {
    let (_d, l) = open();
    assert_eq!(admit(&l, "a", 1000, true).unwrap(), 300);
    // Idempotent replay; differing terms rejected.
    assert_eq!(admit(&l, "a", 1000, true).unwrap(), 300);
    assert!(admit(&l, "a", 1001, true).is_err());
    assert!(admit(&l, "a", 1000, false).is_err());
    // Trusted peers owe no deposit; no cap or reservation is involved.
    assert_eq!(admit(&l, "b", 1_000_000, false).unwrap(), 0);
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
        admit(&l, "r", 1000, true).unwrap();
        record(&l, "r", 0, 1, 300, true);
        let s = l.metered_finalize("r", usage(900, 200)).unwrap();
        assert_eq!(s.status, MeteredStatus::Billed);
        assert_eq!(
            (s.bill_msat, s.deposit_paid_msat, s.remaining_msat),
            (Some(1100), 300, 800)
        );
        // Replay returns the same settlement; a different outcome is refused.
        assert_eq!(l.metered_finalize("r", usage(900, 200)).unwrap(), s);
        assert!(l.metered_finalize("r", usage(900, 201)).is_err());
        assert!(l.metered_finalize("r", MeteredOutcome::Abandoned).is_err());
    }
    // Restart: the bill is durable and the final invoice settles it.
    let l = Ledger::open(dir.path()).unwrap();
    assert_eq!(l.metered_settlement("r").unwrap().remaining_msat, 800);
    record(&l, "r", 1, 2, 800, true);
    assert_eq!(
        l.metered_settlement("r").unwrap().status,
        MeteredStatus::Settled
    );
}

#[test]
fn covering_deposit_means_no_final_invoice_and_no_refund() {
    let (_d, l) = open();
    admit(&l, "a", 1000, true).unwrap();
    record(&l, "a", 0, 1, 300, true);
    let s = l.metered_finalize("a", usage(100, 50)).unwrap();
    assert_eq!(
        (s.status, s.bill_msat, s.deposit_paid_msat, s.remaining_msat),
        (MeteredStatus::Settled, Some(150), 300, 0)
    );
    // Nothing is carried to the next request.
    admit(&l, "b", 10, false).unwrap();
    let s = l.metered_finalize("b", usage(100, 100)).unwrap();
    assert_eq!((s.deposit_paid_msat, s.remaining_msat), (0, 200));
}

#[test]
fn unpaid_deposit_never_reduces_the_bill() {
    let (_d, l) = open();
    admit(&l, "u", 1000, true).unwrap();
    record(&l, "u", 0, 1, 300, false);
    let s = l.metered_finalize("u", usage(900, 200)).unwrap();
    assert_eq!((s.deposit_paid_msat, s.remaining_msat), (0, 1100));
    // A deposit arriving after finalize changes nothing already fixed.
    l.mark_metered_received(&invoice(1, 300).payment_hash)
        .unwrap();
    assert_eq!(l.metered_settlement("u").unwrap().remaining_msat, 1100);
}

#[test]
fn abandoned_requests_have_no_bill_no_debt_and_keep_a_received_deposit() {
    let (_d, l) = open();
    admit(&l, "e", 1000, true).unwrap();
    record(&l, "e", 0, 1, 300, true);
    let s = l.metered_finalize("e", MeteredOutcome::Abandoned).unwrap();
    assert_eq!(
        (s.status, s.bill_msat, s.usage, s.remaining_msat),
        (MeteredStatus::Abandoned, None, None, 0)
    );
    assert_eq!(s.deposit_paid_msat, 300);
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
        admit(&l, &id, 10, false).unwrap();
        l.metered_finalize(&id, usage(10, 10)).unwrap();
        // A billed remainder without an issued invoice is not a strike.
        assert_eq!(l.metered_unpaid_finals("peer").unwrap(), n);
        record(&l, &id, 1, n as u8 + 1, 20, false);
    }
    assert_eq!(l.metered_unpaid_finals("peer").unwrap(), STRIKE_LIMIT);
    assert!(admit(&l, "blocked", 10, true).is_err());
    // Deposits and abandoned requests never count; paying one final unblocks.
    assert_eq!(l.metered_unpaid_finals("other").unwrap(), 0);
    l.mark_metered_received(&invoice(1, 20).payment_hash)
        .unwrap();
    assert_eq!(l.metered_unpaid_finals("peer").unwrap(), STRIKE_LIMIT - 1);
    admit(&l, "again", 10, true).unwrap();
}

#[test]
fn admission_identity_is_exact_token_terms_not_rounded_amounts() {
    let (_d, l) = open();
    let low = Pricing {
        input_msat_per_million: 1,
        output_msat_per_million: 1,
        minimum_invoice_msat: 1,
    };
    let mk = |input| MeteredRequest {
        id: "low",
        peer: "peer",
        pricing: &low,
        estimated_input_tokens: input,
        require_deposit: true,
    };
    let a = l.metered_admit(&mk(10)).unwrap();
    // Same rounded charge, different token terms.
    assert!(l.metered_admit(&mk(11)).is_err());
    assert_eq!(l.metered_admit(&mk(10)).unwrap(), a);
}

#[test]
fn dispatch_claim_is_once_only_and_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let l = Ledger::open(dir.path()).unwrap();
        admit(&l, "c", 1000, false).unwrap();
        assert!(l.metered_claim_dispatch("c", "other").is_err());
        assert!(l.metered_claim_dispatch("c", "peer").unwrap());
        // Admission replay does not re-authorize dispatch.
        admit(&l, "c", 1000, false).unwrap();
        assert!(!l.metered_claim_dispatch("c", "peer").unwrap());
        admit(&l, "n", 1000, false).unwrap();
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
    admit(&l, "s", 1000, true).unwrap();
    // Deposit must equal the admitted deposit.
    assert!(try_record(&l, "s", 0, 20, 299).is_err());
    assert!(
        try_record(&l, "s", 1, 21, 1100).is_err(),
        "no final before bill"
    );
    try_record(&l, "s", 0, 22, 300).unwrap();
    l.mark_metered_received(&invoice(22, 300).payment_hash)
        .unwrap();
    let s = l.metered_finalize("s", usage(900, 200)).unwrap();
    assert_eq!(s.remaining_msat, 800);
    assert!(try_record(&l, "s", 1, 23, 1100).is_err());
    try_record(&l, "s", 1, 23, 800).unwrap();
    // Abandoned and no-deposit rows cannot carry invoices.
    admit(&l, "n", 1000, false).unwrap();
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
        admit(&l, "d", 1000, false).unwrap();
        assert!(l.metered_claim_dispatch("d", "peer").unwrap());
        admit(&l, "p", 1000, true).unwrap();
        record(&l, "p", 0, 31, 300, true);
        admit(&l, "b", 1000, false).unwrap();
        l.metered_finalize("b", usage(10, 10)).unwrap();
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
    assert_eq!(l.metered_settlement("p").unwrap().deposit_paid_msat, 300);
    assert_eq!(
        l.metered_settlement("b").unwrap().status,
        MeteredStatus::Billed
    );
}

#[test]
fn retire_only_never_submitted_metered_deposit() {
    use crate::ledger::{ApprovalMode, Charge, Policy, RequestTerms};
    let (_d, l) = open();
    l.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(100_000),
    })
    .unwrap();
    for (n, id) in [(40u8, "v2a"), (41, "v2b"), (42, "legacy")] {
        l.propose(&RequestTerms {
            exchange_id: None,
            id: id.into(),
            peer: "peer".into(),
            payee: None,
            model: "m".into(),
            pricing: rates(),
            input_tokens: 10,
            max_output_tokens: 10,
            max_total_msat: 1000,
            expires_at_ms: u64::MAX,
        })
        .unwrap();
        if id != "legacy" {
            l.set_setting(&format!("metered-payer/{id}"), "{}").unwrap();
        }
        l.approve(id, 100_000, 1).unwrap();
        l.prepare_charge(&Charge {
            request_id: id.into(),
            segment: 0,
            invoice: invoice(n, 300),
            amount_msat: 300,
            max_total_msat: 400,
        })
        .unwrap();
    }
    // v2b was submitted: must stay unresolved.
    l.begin_submission(&invoice(41, 300).payment_hash).unwrap();
    assert!(l.retire_unsubmitted_metered_deposit("v2a").unwrap());
    assert!(!l.retire_unsubmitted_metered_deposit("v2a").unwrap());
    assert!(!l.retire_unsubmitted_metered_deposit("v2b").unwrap());
    assert!(l.retire_unsubmitted_metered_deposit("legacy").is_err());
    assert_eq!(
        l.charge_state(&invoice(40, 300).payment_hash)
            .unwrap()
            .as_deref(),
        Some("retired")
    );
    assert_eq!(
        l.charge_state(&invoice(41, 300).payment_hash)
            .unwrap()
            .as_deref(),
        Some("pending")
    );
    // No failed cascade: request still approved, and completes cleanly.
    assert_eq!(l.request_state("v2a").unwrap().as_deref(), Some("approved"));
    l.finish("v2a").unwrap();
    assert!(l.finish("v2b").is_err());
    assert!(
        l.pending_charges()
            .unwrap()
            .iter()
            .all(|c| c.request_id != "v2a")
    );
}

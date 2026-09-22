//! Host-owned invoice lifetimes and payment wait deadlines.
//!
//! The wallet contract requires the caller to state how long an invoice stays
//! payable (`WalletProvider::create_invoice(.., expiry_secs)`); no provider
//! default is ever used. Expiry is a safety bound enforced by the payee's node:
//! an HTLC for an expired invoice is rejected, so a late payment fails at the
//! payer instead of landing after the seller has given up. The seller's wait
//! for an input payment therefore runs for exactly the invoice lifetime: a
//! payer cannot cancel an in-flight HTLC, so any moment where the seller has
//! stopped waiting but its node would still claim the payment is a window in
//! which the payer pays for output that has already been discarded.

use std::time::Duration;

/// Input (pre-delivery) inference invoice. The payer pays as soon as it sees
/// the invoice, so a few minutes is generous; it also bounds how long an
/// unpaid input invoice keeps that peer blocked and how long the seller's
/// prefill state can be worth paying for.
pub const INPUT_INVOICE_EXPIRY_SECS: u32 = 5 * 60;

/// Output (post-delivery) inference invoice. Tokens have already been
/// delivered, so the debt exists regardless of the invoice; the expiry only
/// bounds how long a payer that crashed mid-exchange has to recover and pay
/// this exact invoice before the receivable can no longer be settled.
pub const OUTPUT_INVOICE_EXPIRY_SECS: u32 = 60 * 60;

/// `wallet fund-wallet` invoices are paid by a human from another wallet; keep
/// the day the SDK used to default to.
pub const FUNDING_INVOICE_EXPIRY_SECS: u32 = 24 * 60 * 60;

/// How long the seller waits for the input payment to arrive before failing
/// the delivery gate and releasing the backend. Equal to
/// [`INPUT_INVOICE_EXPIRY_SECS`] on purpose: once the invoice has expired the
/// payee's node rejects the HTLC, so "the seller gave up" and "the payment can
/// no longer land" coincide and a payer is never charged for discarded output.
/// The cost is that an unpaid request can pin a backend slot for the whole
/// invoice lifetime; shorten both constants together, never this one alone.
pub const INPUT_ARRIVAL_WAIT: Duration = Duration::from_secs(INPUT_INVOICE_EXPIRY_SECS as u64);

// The seller keeps waiting for as long as the invoice it issued stays payable,
// and lifetimes grow with how long the other side may reasonably take.
const _: () = {
    assert!(INPUT_ARRIVAL_WAIT.as_secs() == INPUT_INVOICE_EXPIRY_SECS as u64);
    assert!(INPUT_INVOICE_EXPIRY_SECS < OUTPUT_INVOICE_EXPIRY_SECS);
    assert!(OUTPUT_INVOICE_EXPIRY_SECS <= FUNDING_INVOICE_EXPIRY_SECS);
};

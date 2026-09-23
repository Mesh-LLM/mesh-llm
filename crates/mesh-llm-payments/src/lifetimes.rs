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
/// the invoice, so a minute is generous for a Lightning payment; it also
/// bounds how long an unpaid request can hold a backend slot (the seller
/// pauses decode while it waits, see [`PRE_PAYMENT_OUTPUT_TOKENS`]), how long
/// an unpaid input invoice keeps that peer blocked, and how long the seller's
/// prefill state can be worth paying for.
pub const INPUT_INVOICE_EXPIRY_SECS: u32 = 60;

/// Output tokens the seller may decode ahead of the input payment. Decode
/// overlaps the payment wait so a paid response is ready the moment the
/// payment arrives; at this many tokens the seller's payment gate pauses decode
/// until the payment arrives, fails, is cancelled or the invoice expires. This
/// also bounds the unpaid work a peer can obtain per request. It must stay well
/// inside the seller's buffered-output cap so the buffer never fills while
/// unpaid (a full buffer stalls the backend stream until its receiver-stall
/// timeout cancels generation).
pub const PRE_PAYMENT_OUTPUT_TOKENS: u64 = 512;

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

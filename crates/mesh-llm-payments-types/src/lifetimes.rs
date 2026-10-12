//! Host-owned invoice lifetimes.
//!
//! The wallet contract requires the caller to state how long an invoice stays
//! payable (`WalletProvider::create_invoice(.., expiry_secs)`); no provider
//! default is ever used. Expiry is a safety bound enforced by the payee's node:
//! an HTLC for an expired invoice is rejected, so a late payment fails at the
//! payer instead of landing after the seller has given up.

/// Metered deposit invoice. The payer pays as soon as it sees the invoice, so
/// a minute is generous for a Lightning payment and bounds how long an unpaid
/// admission can hold a backend slot.
pub const INPUT_INVOICE_EXPIRY_SECS: u32 = 60;

/// Metered final invoice. Tokens have already been delivered, so the debt
/// exists regardless of the invoice; the expiry only bounds how long a payer
/// that crashed mid-exchange has to recover and pay this exact invoice.
pub const OUTPUT_INVOICE_EXPIRY_SECS: u32 = 60 * 60;

/// `wallet fund-wallet` invoices are paid by a human from another wallet; keep
/// the day the SDK used to default to.
pub const FUNDING_INVOICE_EXPIRY_SECS: u32 = 24 * 60 * 60;

// Lifetimes grow with how long the other side may reasonably take.
const _: () = {
    assert!(INPUT_INVOICE_EXPIRY_SECS < OUTPUT_INVOICE_EXPIRY_SECS);
    assert!(OUTPUT_INVOICE_EXPIRY_SECS <= FUNDING_INVOICE_EXPIRY_SECS);
};

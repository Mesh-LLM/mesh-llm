//! Host-owned invoice lifetimes and payment wait deadlines.
//!
//! The wallet contract requires the caller to state how long an invoice stays
//! payable (`WalletProvider::create_invoice(.., expiry_secs)`); no provider
//! default is ever used. Expiry is enforced by the payee's node: an HTLC for an
//! expired invoice is rejected, so a late payment fails at the payer instead of
//! landing after the seller has given up.

use std::time::Duration;

/// Deposit invoice for a cold peer. The payer pays as soon as it sees the
/// invoice and no backend work starts until it lands, so a minute is generous.
pub const DEPOSIT_INVOICE_EXPIRY_SECS: u32 = 60;

/// Final invoice for the metered remainder. The tokens are already delivered,
/// so the debt exists regardless; the expiry only bounds how long a payer that
/// crashed mid-exchange has to recover and pay this exact invoice.
pub const FINAL_INVOICE_EXPIRY_SECS: u32 = 60 * 60;

/// `wallet fund-wallet` invoices are paid by a human from another wallet; keep
/// the day the SDK used to default to.
pub const FUNDING_INVOICE_EXPIRY_SECS: u32 = 24 * 60 * 60;

/// How long the seller waits for the deposit to arrive. Equal to the deposit
/// invoice lifetime: once it has expired the payment can no longer land, so
/// "the seller gave up" and "the payer can no longer be charged" coincide.
pub const DEPOSIT_ARRIVAL_WAIT: Duration = Duration::from_secs(DEPOSIT_INVOICE_EXPIRY_SECS as u64);

const _: () = {
    assert!(DEPOSIT_ARRIVAL_WAIT.as_secs() == DEPOSIT_INVOICE_EXPIRY_SECS as u64);
    assert!(DEPOSIT_INVOICE_EXPIRY_SECS < FINAL_INVOICE_EXPIRY_SECS);
    assert!(FINAL_INVOICE_EXPIRY_SECS <= FUNDING_INVOICE_EXPIRY_SECS);
};

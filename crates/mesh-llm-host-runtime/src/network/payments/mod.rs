//! Invoice exchange over authenticated peer tunnels. Wallet authorization is
//! owned by local ingress; remote forwarding cannot spend the relay's wallet.

mod delivery;
mod gate;
pub(crate) mod request;
mod server;
pub(crate) mod wallet_plugin;

pub(crate) use server::serve;

pub(crate) fn is_payment_upgrade(prefix: &[u8]) -> bool {
    prefix.starts_with(b"POST /mesh/payment/v1 HTTP/1.1\r\n")
}

#[cfg(test)]
mod tests;

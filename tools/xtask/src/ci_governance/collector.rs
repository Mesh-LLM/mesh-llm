//! Fixed-request GitHub collection. App custody and workflow authority are external.
mod collect;
mod pagination;
mod reconcile;
pub(crate) mod request;
pub(crate) mod transport;
mod wire;

pub(crate) use collect::{Clock, SystemClock, collect, read_identity};
pub(crate) use reconcile::open_pr_numbers;
pub(crate) use request::Request;
pub(crate) use transport::{
    ApiResponse, GithubClient, GithubRead, SecretToken, checkout_activation, checkout_head,
};
pub(crate) use wire::timestamp as parse_timestamp;

#[cfg(test)]
mod tests;

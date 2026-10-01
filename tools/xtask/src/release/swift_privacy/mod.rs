//! Generic privacy policy and the experimental supervised native-lint adapter.

pub(super) mod adapter;
mod adapter_args;
mod adapter_error;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Generic verifier is retained unchanged beside the interleaved adapter"
    )
)]
mod embedding;
mod error;
mod native_lint;
mod policy;

#[cfg(test)]
pub(crate) use embedding::verify_files;
pub(crate) use error::PrivacyError;

#[cfg(test)]
mod adapter_tests;
#[cfg(test)]
mod binary_tests;
#[cfg(test)]
mod embedding_tests;
#[cfg(test)]
mod native_lint_tests;
#[cfg(test)]
mod policy_tests;

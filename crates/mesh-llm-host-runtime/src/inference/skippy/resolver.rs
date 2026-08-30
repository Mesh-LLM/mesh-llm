mod request_defaults;
mod resolution;
mod speculative;
mod support;
mod translation;
mod types;

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod native_mtp_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod speculative_tests;

pub(crate) use resolution::resolve_skippy_config;
pub(crate) use types::{
    ResolvedEmbeddedOpenAiArgs, ResolvedSkippyConfig, SkippyConfigResolveRequest,
};

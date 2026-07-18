pub(crate) mod consult;
pub(crate) mod election;
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub(crate) mod mlx;
pub(crate) mod pipeline;
pub(crate) mod skippy;
pub(crate) mod virtual_llm;

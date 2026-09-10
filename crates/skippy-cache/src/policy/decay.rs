//! Exponential decay of reuse statistics (#1650 first slice).

/// Decay tunables. `factor` is the per-observation retention weight applied to
/// past history: 0.9 keeps 90% of each entry's accumulated ratio weight per
/// new observation. Higher `pressure_decay_sensitivity` (via
/// `BenefitPolicy::observe_pressure`) pulls retention toward 0 faster under
/// churn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecayConfig {
    /// Per-observation retention factor in `(0, 1)`.
    pub factor: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self { factor: 0.9 }
    }
}

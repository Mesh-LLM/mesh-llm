const DECIMAL_GB_BYTES: f64 = 1_000_000_000.0;
const GIB_BYTES: f64 = 1024.0 * 1024.0 * 1024.0;

const RATED_CAPACITY_GB_CLASSES: &[u64] = &[
    1, 2, 3, 4, 6, 8, 10, 11, 12, 16, 18, 20, 22, 24, 32, 36, 40, 44, 48, 64, 80, 96, 128, 144,
    160, 192, 256, 384, 512,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramCapacity {
    pub system_reported_bytes: u64,
    pub reserved_bytes: Option<u64>,
}

impl VramCapacity {
    pub const fn new(system_reported_bytes: u64, reserved_bytes: Option<u64>) -> Self {
        Self {
            system_reported_bytes,
            reserved_bytes,
        }
    }

    pub fn allocatable_bytes(self) -> u64 {
        allocatable_bytes(self.system_reported_bytes, self.reserved_bytes)
    }

    pub fn rated_capacity_gb(self) -> Option<u64> {
        rated_capacity_gb(self.system_reported_bytes)
    }
}

pub fn allocatable_bytes(system_reported_bytes: u64, reserved_bytes: Option<u64>) -> u64 {
    system_reported_bytes.saturating_sub(reserved_bytes.unwrap_or(0))
}

pub fn decimal_gb(system_reported_bytes: u64) -> f64 {
    system_reported_bytes as f64 / DECIMAL_GB_BYTES
}

pub fn rated_capacity_gb(system_reported_bytes: u64) -> Option<u64> {
    if system_reported_bytes == 0 {
        return None;
    }

    let decimal = best_rated_candidate(system_reported_bytes, DECIMAL_GB_BYTES);
    let binary = best_rated_candidate(system_reported_bytes, GIB_BYTES);
    Some(if decimal.relative_error <= binary.relative_error {
        decimal.capacity_gb
    } else {
        binary.capacity_gb
    })
}

pub fn format_rated_capacity(system_reported_bytes: u64) -> String {
    rated_capacity_gb(system_reported_bytes)
        .map(|gb| format!("{gb} GB"))
        .unwrap_or_else(|| "unknown".to_string())
}

pub fn format_decimal_gb(system_reported_bytes: u64) -> String {
    if system_reported_bytes == 0 {
        "unknown".to_string()
    } else {
        format!("{:.1} GB", decimal_gb(system_reported_bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RatedCandidate {
    capacity_gb: u64,
    relative_error: f64,
}

fn best_rated_candidate(system_reported_bytes: u64, unit_bytes: f64) -> RatedCandidate {
    let bytes = system_reported_bytes as f64;
    RATED_CAPACITY_GB_CLASSES
        .iter()
        .copied()
        .map(|capacity_gb| {
            let candidate_bytes = capacity_gb as f64 * unit_bytes;
            RatedCandidate {
                capacity_gb,
                relative_error: ((bytes - candidate_bytes) / candidate_bytes).abs(),
            }
        })
        .min_by(|left, right| {
            left.relative_error
                .partial_cmp(&right.relative_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("rated capacity classes must not be empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rated_capacity_prefers_binary_sized_driver_totals() {
        assert_eq!(rated_capacity_gb(32 * 1024 * 1024 * 1024), Some(32));
        assert_eq!(format_rated_capacity(32 * 1024 * 1024 * 1024), "32 GB");
    }

    #[test]
    fn rated_capacity_preserves_decimal_sized_totals() {
        assert_eq!(rated_capacity_gb(24_000_000_000), Some(24));
    }

    #[test]
    fn rated_capacity_maps_near_decimal_totals_to_the_product_class() {
        assert_eq!(rated_capacity_gb(32_359_738_368), Some(32));
    }

    #[test]
    fn allocatable_capacity_subtracts_true_reserved_memory() {
        let capacity = VramCapacity::new(32 * 1024 * 1024 * 1024, Some(512 * 1024 * 1024));
        assert_eq!(capacity.allocatable_bytes(), 33_822_867_456);
    }

    #[test]
    fn allocatable_capacity_saturates_when_reserved_exceeds_total() {
        assert_eq!(allocatable_bytes(1_000, Some(2_000)), 0);
    }
}

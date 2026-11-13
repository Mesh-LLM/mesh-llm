//! Comparing a check's error with its tolerance so that a value that is
//! not a number fails.
//!
//! `error > tolerance` is false for NaN, and `f64::max` drops a NaN
//! operand. A check written with either passes a node that opens a NaN.
//! Every comparison of an opened value goes through these instead.

/// Whether `error` is past `tolerance`. An error that is not a number is.
pub(crate) fn exceeds(error: f64, tolerance: f64) -> bool {
    error.is_nan() || error > tolerance
}

/// Whether `error` is within `tolerance`. An error that is not a number
/// is not.
pub(crate) fn within(error: f64, tolerance: f64) -> bool {
    !exceeds(error, tolerance)
}

/// The larger of two errors; infinite when either is not a number.
pub(crate) fn worst(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::INFINITY
    } else {
        a.max(b)
    }
}

/// The largest of `errors`, from 0; infinite when one is not a number.
pub(crate) fn worst_of(errors: impl IntoIterator<Item = f64>) -> f64 {
    errors.into_iter().fold(0.0, worst)
}

/// Whether every value is a number.
pub(crate) fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_error_that_is_not_a_number_is_past_every_tolerance() {
        assert!(exceeds(f64::NAN, 1e-3));
        assert!(exceeds(f64::INFINITY, 1e-3));
        assert!(exceeds(2e-3, 1e-3));
        assert!(within(1e-3, 1e-3));
        assert!(!within(f64::NAN, f64::INFINITY));
    }

    #[test]
    fn the_worst_error_keeps_a_value_that_is_not_a_number() {
        assert_eq!(worst(1.0, 2.0), 2.0);
        assert_eq!(worst(f64::NAN, 2.0), f64::INFINITY);
        assert_eq!(worst(2.0, f64::NAN), f64::INFINITY);
        assert_eq!(worst_of([0.5, f64::NAN, 0.25]), f64::INFINITY);
        assert_eq!(worst_of([0.5, 0.25]), 0.5);
        assert_eq!(worst_of([]), 0.0);
        // `f64::max`, which the checks used, drops it
        assert_eq!([0.5, f64::NAN].into_iter().fold(0.0, f64::max), 0.5);
    }

    #[test]
    fn finite_values_hold_no_nan_or_infinity() {
        assert!(all_finite(&[0.0, -1.5, f32::MAX]));
        assert!(!all_finite(&[0.0, f32::NAN]));
        assert!(!all_finite(&[f32::NEG_INFINITY]));
    }
}

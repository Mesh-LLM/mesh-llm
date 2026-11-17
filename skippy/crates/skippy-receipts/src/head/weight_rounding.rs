//! Profile-specific output weights shared by both head verifiers.

use std::borrow::Cow;

use crate::activation_rounding::{Profile, round_to_f16};

pub(super) fn rows_for(
    profile: Profile,
    rows: &[(usize, Vec<f64>)],
) -> Cow<'_, [(usize, Vec<f64>)]> {
    if profile.rounds_weights() {
        Cow::Owned(rows.iter().map(|(r, w)| (*r, round_to_f16(w))).collect())
    } else {
        Cow::Borrowed(rows)
    }
}

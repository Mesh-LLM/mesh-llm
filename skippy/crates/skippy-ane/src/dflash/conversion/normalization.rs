//! RMS normalization used by DFlash graph construction.

use mil_spec::Block;

pub(super) fn rms(
    b: &mut Block,
    x: &str,
    weight: &str,
    shape: &[i64],
    axis: i32,
    name: &str,
    epsilon_value: f32,
) -> String {
    let mut reduced = shape.to_vec();
    reduced[axis as usize] = 1;
    let absolute = b.abs(x, shape, &format!("{name}_abs"));
    let maximum = b.reduce_max(&absolute, &[axis], true, &reduced, &format!("{name}_max"));
    let inverse = b.inverse(&maximum, 1e-4, &reduced, &format!("{name}_inverse"));
    let scaled = b.mul(x, &inverse, shape, &format!("{name}_scaled"));
    let square = b.mul(&scaled, &scaled, shape, &format!("{name}_square"));
    let mean = b.reduce_mean(&square, &[axis], true, &reduced, &format!("{name}_mean"));
    let epsilon = b.konst_f16(&format!("{name}_epsilon"), epsilon_value);
    let scaled_inverse = b.mul(
        &inverse,
        &epsilon,
        &reduced,
        &format!("{name}_scaled_inverse"),
    );
    let epsilon = b.mul(
        &inverse,
        &scaled_inverse,
        &reduced,
        &format!("{name}_scaled_epsilon"),
    );
    let denominator = b.add(&mean, &epsilon, &reduced, &format!("{name}_denominator"));
    let reciprocal = b.rsqrt(&denominator, 1e-7, &reduced, &format!("{name}_reciprocal"));
    let normalized = b.mul(&scaled, &reciprocal, shape, &format!("{name}_normalized"));
    b.mul(&normalized, weight, shape, name)
}

//! CubeCL feasibility spike for the CacheGen hot kernels (#1652).
//!
//! Evidence, not commitment: this example is feature-gated behind
//! `cachegen-spike`; nothing in the library links CubeCL. Per the agreed
//! gate it reports six numbers separately on CPU and on the best available
//! wgpu backend (Metal on Apple Silicon):
//!
//! 1. cold JIT/compile time (first launch of a kernel specialization)
//! 2. warm dispatch time (steady-state launches)
//! 3. host→device bytes copied
//! 4. device→host bytes copied
//! 5. peak temporary device memory
//! 6. output equality against the CPU reference (bitwise)
//!
//! Measurement honesty, per the review of PR #1752: every timed stage
//! ends with a `client.sync()` *inside* the timer, so the number covers
//! real completion — not unsynchronized enqueue. "Cold JIT" is the first
//! launch of a kernel specialization in this process (compile + pipeline
//! creation + execution); in cubecl 0.10.0 the only on-disk kernel cache
//! is the Vulkan-only SPIR-V cache, which this build does not enable, so
//! per-process first launch is the true cold path for CPU and Metal
//! alike. Warm numbers are the average over synchronized steady-state
//! launches. Equality is exact `==` on symbols and f32 values: dequantized
//! values are exact products of small integers and dyadic floats, so any
//! non-bit-equal output is a real divergence, not rounding noise.
//!
//! Kernel decomposition, stated honestly: quantization is embarrassingly
//! parallel; the token-axis delta is a scan along rows. KV tiles are
//! tall-thin (thousands of token rows, `dims` of one head layout), so the
//! spike assigns one unit per column and walks rows sequentially inside
//! the unit — the same arithmetic as the CPU reference, ordered so the
//! result is bit-exact. A parallel-scan kernel is the performance
//! follow-up; the rANS entropy stage remains CPU in this reference and is
//! measured here only as an encoded-size ratio.
//!
//! Run:
//!   cargo run -p skippy-cache --example cachegen_cubecl_spike \
//!     --features cachegen-spike -- --rows 4096 --dims 128 [--iterations 20]
//!
//! `--iterations` only widens the warm-dispatch sample; the cold-JIT
//! number is a single first launch by definition.

use cubecl::prelude::*;
use skippy_cache::cachegen::reference;
use skippy_protocol::binary::{f16_bits_to_f32, f32_to_f16_bits};

/// One unit per column; walks token rows sequentially so the delta ring
/// matches the CPU reference exactly.
#[cube(launch)]
fn quantize_delta_columns(
    values: &Array<f32>,
    symbols: &mut Array<u32>,
    calib: &Array<f32>,
    #[comptime] rows: usize,
    #[comptime] dims: usize,
) {
    let column = UNIT_POS_X as usize;
    if column < dims {
        let min = calib[0];
        let scale = calib[1];
        let mut prev: u32 = 0;
        for row in 0..rows {
            let index = row * dims + column;
            let scaled = ((values[index] - min) / scale).round();
            let plain = u32::cast_from(scaled.clamp(0.0, 15.0));
            symbols[index] = (plain + 16 - prev) % 16;
            prev = plain;
        }
    }
}

/// Inverse scan: one unit per column re-accumulates the reconstructed
/// symbol sequence, then dequantizes. Bit-exact against the CPU reference.
#[cube(launch)]
fn undelta_dequantize_columns(
    symbols: &Array<u32>,
    values: &mut Array<f32>,
    calib: &Array<f32>,
    #[comptime] rows: usize,
    #[comptime] dims: usize,
) {
    let column = UNIT_POS_X as usize;
    if column < dims {
        let min = calib[0];
        let scale = calib[1];
        let mut prev: u32 = 0;
        for row in 0..rows {
            let index = row * dims + column;
            let plain = (symbols[index] + prev) % 16;
            values[index] = f32::cast_from(plain) * scale + min;
            prev = plain;
        }
    }
}

struct StageTiming {
    /// First synchronized launch of this kernel specialization in this
    /// process: JIT compile + pipeline creation + execution to completion.
    cold_compile_ms: u128,
    /// Average over synchronized steady-state launches (completion, not
    /// enqueue).
    warm_dispatch_us: u128,
}

/// Blocks until the client's stream drains. Called inside every timed
/// region: without it the timer measures launch enqueue only and the
/// actual work lands after the clock stops (the PR #1752 review bug).
fn synchronize<R: Runtime>(client: &ComputeClient<R>) {
    cubecl::future::block_on(client.sync())
        .unwrap_or_else(|error| panic!("device sync failed: {error}"));
}

fn timed_stage<R: Runtime>(
    client: &ComputeClient<R>,
    iterations: u32,
    mut launch: impl FnMut(),
) -> StageTiming {
    synchronize::<R>(client);
    let cold = std::time::Instant::now();
    launch();
    synchronize::<R>(client);
    let cold_compile_ms = cold.elapsed().as_millis();
    // Steady state: the first launch populated the pipeline cache, so
    // every launch here is the warm path. Each launch is individually
    // synchronized inside the timed region; the reported number is total
    // elapsed / iterations, i.e. the steady-state cost per launch
    // including completion.
    let warm_start = std::time::Instant::now();
    for _ in 0..iterations {
        launch();
        synchronize::<R>(client);
    }
    StageTiming {
        cold_compile_ms,
        warm_dispatch_us: warm_start.elapsed().as_micros() / u128::from(iterations),
    }
}

fn run_backend<R: Runtime>(
    backend: &'static str,
    tile: &[u8],
    rows: usize,
    dims: usize,
    expected_symbols: &[u8],
    expected_values: &[f32],
    iterations: u32,
) -> Result<(), String> {
    if dims > 1024 {
        return Err(format!(
            "{backend}: spike shape needs dims <= 1024 units, got {dims}"
        ));
    }

    let client = R::client(&R::Device::default());
    let count = rows * dims;
    let values: Vec<f32> = tile
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f16_bits_to_f32(u16::from_le_bytes(*bytes)))
        .collect();
    let calibration = reference::calibrate(&values).map_err(|error| error.to_string())?;

    // Timed H2D: the honest copy path is every byte the kernel work needs
    // (the f32 tile plus the 8-byte calibration vector), measured to
    // completion.
    let h2d_start = std::time::Instant::now();
    let values_handle = client.create_from_slice(f32::as_bytes(&values));
    let calib_handle = client.create_from_slice(f32::as_bytes(&[
        f32::from_bits(calibration.min_bits),
        f32::from_bits(calibration.scale_bits),
    ]));
    synchronize::<R>(&client);
    let h2d_us = h2d_start.elapsed().as_micros();
    let h2d_bytes = values.len() * core::mem::size_of::<f32>() + 2 * core::mem::size_of::<f32>();
    // Working-set allocations (outputs) are not timed; they are part of
    // the live peak, not the transfer.
    let symbols_handle = client.empty(count * core::mem::size_of::<u32>());
    let rebuilt_handle = client.empty(count * core::mem::size_of::<f32>());
    // Actual live allocation peak while the kernels run: every buffer the
    // harness holds at once — inputs (tile + calibration) and both
    // outputs. Reported rather than derived, per the review.
    let peak_temporary_bytes =
        values_handle.size() + calib_handle.size() + symbols_handle.size() + rebuilt_handle.size();
    if calibration.scale_bits == 0.0f32.to_bits() {
        return Err(
            "flat tile (scale == 0): not exercised by the spike; the CPU reference covers it"
                .to_string(),
        );
    }

    let encode = || unsafe {
        quantize_delta_columns::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(dims as u32),
            ArrayArg::from_raw_parts(values_handle.clone(), count),
            ArrayArg::from_raw_parts(symbols_handle.clone(), count),
            ArrayArg::from_raw_parts(calib_handle.clone(), 2),
            rows,
            dims,
        )
    };
    let encode_timing = timed_stage::<R>(&client, iterations, encode);

    let decode = || unsafe {
        undelta_dequantize_columns::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(dims as u32),
            ArrayArg::from_raw_parts(symbols_handle.clone(), count),
            ArrayArg::from_raw_parts(rebuilt_handle.clone(), count),
            ArrayArg::from_raw_parts(calib_handle.clone(), 2),
            rows,
            dims,
        )
    };
    let decode_timing = timed_stage::<R>(&client, iterations, decode);

    // Timed D2H: both returns measured to completion, like the uploads.
    let d2h_start = std::time::Instant::now();
    let symbols_bytes = client
        .read_one(symbols_handle.clone())
        .map_err(|error| error.to_string())?;
    let rebuilt_bytes = client
        .read_one(rebuilt_handle.clone())
        .map_err(|error| error.to_string())?;
    synchronize::<R>(&client);
    let d2h_us = d2h_start.elapsed().as_micros();
    let d2h_bytes = symbols_bytes.len() + rebuilt_bytes.len();
    let device_symbols_u32 = u32::from_bytes(&symbols_bytes);
    // Range-check before the narrowing cast: `as u8` would silently alias
    // an out-of-alphabet u32 (e.g. 256 -> 0) into a false parity pass.
    let all_in_alphabet = device_symbols_u32.iter().all(|&symbol| symbol < 16);
    let device_symbols: Vec<u8> = device_symbols_u32.iter().map(|&s| s as u8).collect();
    let mut histogram = vec![0u32; reference::TOKEN_COUNT];
    for &symbol in &device_symbols {
        histogram[usize::from(symbol)] += 1;
    }
    let freqs =
        reference::histogram_to_freqs(&histogram, count).map_err(|error| error.to_string())?;
    let table = skippy_cache::cachegen::rans::SymbolTable::from_freqs(&freqs)
        .ok_or("rANS table construction failed")?;
    let mut encoder = skippy_cache::cachegen::rans::RansEncoder::new();
    for &symbol in device_symbols.iter().rev() {
        encoder.put(&table, usize::from(symbol));
    }
    let stream = encoder.finish();
    let ratio = stream.len() as f64 / tile.len() as f64;

    let rebuilt = f32::from_bytes(&rebuilt_bytes);
    let symbols_match = all_in_alphabet && device_symbols == expected_symbols;
    // Exact bitwise comparison: dequantized values are `symbol * scale +
    // min` where symbol is a small integer and scale/min are identical f32
    // bits on both sides, so the f32 words must match exactly. A tolerance
    // here is what let an enqueue-only measurement pass for parity.
    let values_match = rebuilt.len() == expected_values.len()
        && rebuilt
            .iter()
            .zip(expected_values.iter())
            .all(|(device, reference_value)| device.to_bits() == reference_value.to_bits());
    let symbol_mismatches = device_symbols
        .iter()
        .zip(expected_symbols.iter())
        .filter(|(device, expected)| device != expected)
        .count();
    let value_mismatches = rebuilt
        .iter()
        .zip(expected_values.iter())
        .filter(|(device, expected)| device.to_bits() != expected.to_bits())
        .count();

    println!("=== {backend} ===");
    println!(
        "quantize+delta: cold {} ms, warm {} us | undelta+dequantize: cold {} ms, warm {} us",
        encode_timing.cold_compile_ms,
        encode_timing.warm_dispatch_us,
        decode_timing.cold_compile_ms,
        decode_timing.warm_dispatch_us,
    );
    println!(
        "copies: H2D {} bytes in {} us (f32 tile + 8 B calibration, to completion) | D2H {} bytes in {} us (symbols + rebuilt) | live device buffer peak {} bytes (tile + calibration + both outputs)",
        h2d_bytes, h2d_us, d2h_bytes, d2h_us, peak_temporary_bytes,
    );
    println!(
        "encoded-size ratio: rANS {} bytes / raw {} bytes = {:.3}",
        stream.len(),
        tile.len(),
        ratio
    );
    println!("equality vs CPU reference (bitwise): symbols={symbols_match}, values={values_match}");
    println!(
        "mismatch counts: symbols {symbol_mismatches}/{}, values {value_mismatches}/{}",
        device_symbols.len(),
        rebuilt.len()
    );
    if symbols_match && values_match {
        Ok(())
    } else {
        Err(format!(
            "{backend}: device output diverges from the CPU reference (symbols={symbols_match}, values={values_match})"
        ))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parse = |name: &str, default: usize| -> usize {
        let position = args.iter().position(|argument| argument == name);
        match position {
            Some(index) => args
                .get(index + 1)
                .and_then(|value| value.parse().ok())
                .unwrap_or(default),
            None => default,
        }
    };
    let rows = parse("--rows", 4096);
    let dims = parse("--dims", 128);
    let iterations = u32::try_from(parse("--iterations", 20)).unwrap_or(20);

    // Smooth KV-like fixture, deterministic, the shape CacheGen gains come
    // from.
    let count = rows * dims;
    let mut tile = Vec::with_capacity(count * 2);
    for row in 0..rows {
        for column in 0..dims {
            let phase = (row * dims + column) as f32;
            let value = (phase * 0.000_5).sin() * 0.4 + 0.5;
            tile.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
        }
    }

    // CPU reference outputs for the equality gate.
    let values: Vec<f32> = tile
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f16_bits_to_f32(u16::from_le_bytes(*bytes)))
        .collect();
    let calibration = reference::calibrate(&values).expect("calibrate");
    let mut symbols = reference::quantize(&calibration, &values).expect("quantize");
    reference::delta_encode(&mut symbols, dims).expect("delta");
    // The device decode path undeltas before dequantizing; the expected
    // values must follow the same order of operations.
    let mut undeltaed = symbols.clone();
    reference::delta_decode(&mut undeltaed, dims).expect("undelta");
    let expected_values = reference::dequantize(&calibration, &undeltaed);

    println!(
        "tile: {rows} rows x {dims} dims = {count} f16 values ({} raw bytes)",
        tile.len()
    );

    let mut failures = Vec::new();
    if let Err(error) = run_backend::<cubecl::cpu::CpuRuntime>(
        "cubecl-cpu",
        &tile,
        rows,
        dims,
        &symbols,
        &expected_values,
        iterations,
    ) {
        failures.push(error);
    }
    if let Err(error) = run_backend::<cubecl::wgpu::WgpuRuntime>(
        "wgpu(Metal)",
        &tile,
        rows,
        dims,
        &symbols,
        &expected_values,
        iterations,
    ) {
        failures.push(error);
    }

    if failures.is_empty() {
        println!("SPIKE PASS: device outputs match the CPU reference on every available backend");
    } else {
        for failure in &failures {
            eprintln!("SPIKE FAILURE: {failure}");
        }
        std::process::exit(1);
    }
}

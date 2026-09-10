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
//! 6. output equality against the CPU reference
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
//!     --features cachegen-spike -- --rows 4096 --dims 128

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
    cold_compile_ms: u128,
    warm_dispatch_us: u128,
}

fn timed_stage<R: Runtime>(
    _client: &ComputeClient<R>,
    iterations: u32,
    mut launch: impl FnMut(),
) -> StageTiming {
    let cold = std::time::Instant::now();
    launch();
    let cold_compile_ms = cold.elapsed().as_millis();
    let warm = std::time::Instant::now();
    for _ in 0..iterations {
        launch();
    }
    StageTiming {
        cold_compile_ms,
        warm_dispatch_us: warm.elapsed().as_micros() / u128::from(iterations),
    }
}

fn run_backend<R: Runtime>(
    backend: &'static str,
    tile: &[u8],
    rows: usize,
    dims: usize,
    expected_symbols: &[u8],
    expected_values: &[f32],
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

    // Device buffers: one H2D upload of the f16-decoded f32 tile, one
    // symbols buffer, one rebuilt-values buffer.
    let values_handle = client.create_from_slice(f32::as_bytes(&values));
    let symbols_handle = client.empty(count * core::mem::size_of::<u32>());
    let rebuilt_handle = client.empty(count * core::mem::size_of::<f32>());
    let calib_handle = client.create_from_slice(f32::as_bytes(&[
        f32::from_bits(calibration.min_bits),
        f32::from_bits(calibration.scale_bits),
    ]));
    let peak_temporary_bytes = (symbols_handle.size() + rebuilt_handle.size()) as u64;
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
    let encode_timing = timed_stage::<R>(&client, 20, encode);

    let symbols_bytes = client
        .read_one(symbols_handle.clone())
        .map_err(|error| error.to_string())?;
    let device_symbols_u32 = u32::from_bytes(&symbols_bytes);
    let device_symbols: Vec<u8> = device_symbols_u32.iter().map(|&s| s as u8).collect();

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
    let decode_timing = timed_stage::<R>(&client, 20, decode);

    let rebuilt_bytes = client
        .read_one(rebuilt_handle.clone())
        .map_err(|error| error.to_string())?;
    let rebuilt = f32::from_bytes(&rebuilt_bytes);

    // Encoded-size ratio: CPU rANS over the device-produced symbols.
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

    let symbols_match = device_symbols == expected_symbols;
    let values_match = rebuilt.len() == expected_values.len()
        && rebuilt
            .iter()
            .zip(expected_values.iter())
            .all(|(device, reference_value)| (device - reference_value).abs() < 1e-6);

    println!("=== {backend} ===");
    println!(
        "quantize+delta: cold {} ms, warm {} us | undelta+dequantize: cold {} ms, warm {} us",
        encode_timing.cold_compile_ms,
        encode_timing.warm_dispatch_us,
        decode_timing.cold_compile_ms,
        decode_timing.warm_dispatch_us,
    );
    println!(
        "copies: H2D {} bytes (f32 tile), D2H {} bytes (symbols + rebuilt) | peak temporary device memory {} bytes",
        count * core::mem::size_of::<f32>(),
        symbols_bytes.len() + rebuilt_bytes.len(),
        peak_temporary_bytes,
    );
    println!(
        "encoded-size ratio: rANS {} bytes / raw {} bytes = {:.3}",
        stream.len(),
        tile.len(),
        ratio
    );
    println!("equality vs CPU reference: symbols={symbols_match}, values={values_match}");
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

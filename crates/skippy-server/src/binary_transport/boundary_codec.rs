//! Learned low-rank activation codec for split boundaries.
//!
//! A codec is a per-boundary orthonormal projection fitted offline (PCA over
//! calibration activations). Encoding projects each d-wide activation row to
//! k coefficients and int8-quantizes them with a per-row scale; decoding
//! reconstructs `x̂ = Qᵀ·y + μ`. Both stages of a boundary must load
//! byte-identical codec files.

use std::fs;
use std::io::{self, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};

const CODEC_MAGIC: &[u8; 8] = b"SKBC0001";

/// Maximum supported width/rank; a codec file above this is rejected as
/// corrupt rather than allocating unbounded memory.
const MAX_CODEC_DIM: usize = 65_536;

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryCodec {
    d: usize,
    k: usize,
    /// Mean activation, length `d`.
    mean: Vec<f32>,
    /// `k` orthonormal component rows, each of length `d` (row-major).
    components: Vec<f32>,
}

impl BoundaryCodec {
    pub fn new(d: usize, k: usize, mean: Vec<f32>, components: Vec<f32>) -> Result<Self> {
        if d == 0 || k == 0 || k > d || d > MAX_CODEC_DIM {
            bail!("boundary codec dimensions out of range: d={d} k={k}");
        }
        if mean.len() != d {
            bail!("boundary codec mean length {} != d {d}", mean.len());
        }
        if components.len() != k * d {
            bail!(
                "boundary codec component length {} != k*d {}",
                components.len(),
                k * d
            );
        }
        Ok(Self {
            d,
            k,
            mean,
            components,
        })
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Encodes `token_count` f32 rows (little-endian, `token_count * d`
    /// values) into the lowrank wire layout:
    /// `[token_count f32 scales][token_count * k i8 coefficients]`.
    pub fn encode(&self, f32_payload: &[u8], token_count: usize) -> io::Result<Vec<u8>> {
        let expected = token_count
            .checked_mul(self.d)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| invalid_data("lowrank source byte count overflow"))?;
        if f32_payload.len() != expected {
            return Err(invalid_data("lowrank source payload size mismatch"));
        }
        let mut scales = Vec::with_capacity(token_count * 4);
        let mut packed = Vec::with_capacity(token_count * self.k);
        let mut centered = vec![0.0_f32; self.d];
        let mut coeffs = vec![0.0_f32; self.k];
        for token_index in 0..token_count {
            let row_offset = token_index * self.d * 4;
            let row = &f32_payload[row_offset..row_offset + self.d * 4];
            for (index, chunk) in row.chunks_exact(4).enumerate() {
                let value = f32::from_le_bytes(chunk.try_into().expect("chunks_exact size"));
                centered[index] = value - self.mean[index];
            }
            let mut max_abs = 0.0_f32;
            for (component_index, coeff) in coeffs.iter_mut().enumerate() {
                let component = &self.components
                    [component_index * self.d..(component_index + 1) * self.d];
                let mut dot = 0.0_f32;
                for (value, weight) in centered.iter().zip(component) {
                    dot += value * weight;
                }
                *coeff = dot;
                max_abs = max_abs.max(dot.abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales.extend_from_slice(&scale.to_le_bytes());
            for coeff in &coeffs {
                packed.push((coeff / scale).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
        scales.extend_from_slice(&packed);
        Ok(scales)
    }

    /// Decodes a lowrank wire payload back into `token_count * d` f32 values
    /// (little-endian bytes).
    pub fn decode(&self, payload: &[u8], token_count: usize) -> io::Result<Vec<u8>> {
        let scale_bytes = token_count
            .checked_mul(4)
            .ok_or_else(|| invalid_data("lowrank scale byte count overflow"))?;
        let coeff_bytes = token_count
            .checked_mul(self.k)
            .ok_or_else(|| invalid_data("lowrank coefficient byte count overflow"))?;
        let expected = scale_bytes
            .checked_add(coeff_bytes)
            .ok_or_else(|| invalid_data("lowrank payload byte count overflow"))?;
        if payload.len() != expected {
            return Err(invalid_data("lowrank payload size mismatch"));
        }
        let mut out = Vec::with_capacity(token_count * self.d * 4);
        let mut row = vec![0.0_f32; self.d];
        for token_index in 0..token_count {
            let scale_offset = token_index * 4;
            let scale = f32::from_le_bytes(
                payload[scale_offset..scale_offset + 4]
                    .try_into()
                    .expect("slice length"),
            );
            row.copy_from_slice(&self.mean);
            let coeff_offset = scale_bytes + token_index * self.k;
            for component_index in 0..self.k {
                let coeff = (payload[coeff_offset + component_index] as i8) as f32 * scale;
                if coeff == 0.0 {
                    continue;
                }
                let component = &self.components
                    [component_index * self.d..(component_index + 1) * self.d];
                for (value, weight) in row.iter_mut().zip(component) {
                    *value += coeff * weight;
                }
            }
            for value in &row {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// Fits a codec by PCA over calibration rows (`samples.len() / d` rows)
    /// using orthogonal iteration. Deterministic: the starting subspace is
    /// seeded from a fixed splitmix64 stream.
    pub fn fit(samples: &[f32], d: usize, k: usize, iterations: usize) -> Result<Self> {
        if d == 0 || k == 0 || k > d || d > MAX_CODEC_DIM {
            bail!("boundary codec dimensions out of range: d={d} k={k}");
        }
        if samples.is_empty() || !samples.len().is_multiple_of(d) {
            bail!("calibration sample length {} is not a multiple of d {d}", samples.len());
        }
        let n = samples.len() / d;
        if n < k {
            bail!("need at least k={k} calibration rows, got {n}");
        }
        let mut mean = vec![0.0_f32; d];
        for row in samples.chunks_exact(d) {
            for (accumulator, value) in mean.iter_mut().zip(row) {
                *accumulator += value;
            }
        }
        for value in &mut mean {
            *value /= n as f32;
        }

        // Orthogonal iteration on the covariance action: Z = Aᵀ(A·Q).
        let mut q = deterministic_matrix(d, k);
        orthonormalize(&mut q, d, k);
        let mut projected = vec![0.0_f32; n * k];
        for _ in 0..iterations.max(1) {
            // projected = centered · Qᵀ  (n×k)
            for (row_index, row) in samples.chunks_exact(d).enumerate() {
                for component_index in 0..k {
                    let component = &q[component_index * d..(component_index + 1) * d];
                    let mut dot = 0.0_f32;
                    for ((value, mean_value), weight) in row.iter().zip(&mean).zip(component) {
                        dot += (value - mean_value) * weight;
                    }
                    projected[row_index * k + component_index] = dot;
                }
            }
            // q = projectedᵀ · centered  (k×d)
            let mut next = vec![0.0_f32; k * d];
            for (row_index, row) in samples.chunks_exact(d).enumerate() {
                for component_index in 0..k {
                    let coeff = projected[row_index * k + component_index];
                    if coeff == 0.0 {
                        continue;
                    }
                    let target = &mut next[component_index * d..(component_index + 1) * d];
                    for ((accumulator, value), mean_value) in
                        target.iter_mut().zip(row).zip(&mean)
                    {
                        *accumulator += coeff * (value - mean_value);
                    }
                }
            }
            q = next;
            orthonormalize(&mut q, d, k);
        }
        Self::new(d, k, mean, q)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut bytes = Vec::with_capacity(16 + (self.d + self.k * self.d) * 4);
        bytes.extend_from_slice(CODEC_MAGIC);
        bytes.extend_from_slice(&(self.d as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.k as u32).to_le_bytes());
        for value in &self.mean {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.components {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(path, bytes).with_context(|| format!("write boundary codec {}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file =
            fs::File::open(path).with_context(|| format!("open boundary codec {}", path.display()))?;
        let mut reader = io::BufReader::new(file);
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic).context("read codec magic")?;
        if &magic != CODEC_MAGIC {
            bail!("{} is not a boundary codec file", path.display());
        }
        let d = read_u32(&mut reader)? as usize;
        let k = read_u32(&mut reader)? as usize;
        if d == 0 || k == 0 || k > d || d > MAX_CODEC_DIM {
            bail!("boundary codec {} has invalid dimensions d={d} k={k}", path.display());
        }
        let mean = read_f32_vec(&mut reader, d)?;
        let components = read_f32_vec(&mut reader, k * d)?;
        Self::new(d, k, mean, components)
    }
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).context("read codec u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32_vec(reader: &mut impl Read, len: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0_u8; len * 4];
    reader
        .read_exact(&mut bytes)
        .context("read codec tensor data")?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunks_exact size")))
        .collect())
}

/// Deterministic pseudo-random starting subspace (splitmix64 → uniform).
fn deterministic_matrix(d: usize, k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k * d);
    let mut state = 0x1234_5678_9ABC_DEF0_u64;
    for _ in 0..k * d {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push(((z >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5);
    }
    out
}

/// Modified Gram-Schmidt over `k` row vectors of width `d`.
fn orthonormalize(rows: &mut [f32], d: usize, k: usize) {
    for row_index in 0..k {
        for prior_index in 0..row_index {
            let mut dot = 0.0_f32;
            for column in 0..d {
                dot += rows[row_index * d + column] * rows[prior_index * d + column];
            }
            for column in 0..d {
                rows[row_index * d + column] -= dot * rows[prior_index * d + column];
            }
        }
        let mut norm = 0.0_f32;
        for column in 0..d {
            norm += rows[row_index * d + column] * rows[row_index * d + column];
        }
        let norm = norm.sqrt();
        if norm > 1e-12 {
            for column in 0..d {
                rows[row_index * d + column] /= norm;
            }
        }
    }
}

/// CLI entry: fits a codec from a raw f32 activation dump and writes it out.
pub fn fit_boundary_codec_cli(args: &crate::cli::FitBoundaryCodecArgs) -> Result<()> {
    let bytes = fs::read(&args.input)
        .with_context(|| format!("read calibration dump {}", args.input.display()))?;
    if !bytes.len().is_multiple_of(4) {
        bail!("calibration dump is not a whole number of f32 values");
    }
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunks_exact size")))
        .collect();
    let rows = samples.len() / args.width.max(1);
    eprintln!(
        "fitting boundary codec: {} rows of width {}, rank {}, {} iterations",
        rows, args.width, args.rank, args.iterations
    );
    let codec = BoundaryCodec::fit(&samples, args.width, args.rank, args.iterations)?;
    codec.save(&args.output)?;
    eprintln!(
        "wrote {} ({}x compression vs f16 at the boundary)",
        args.output.display(),
        (args.width * 2) as f64 / (args.rank as f64 + 4.0 / 1.0f64.max(1.0)),
    );
    Ok(())
}

/// Appends f32 activation rows to the capture file named by
/// `SKIPPY_CAPTURE_BOUNDARY_ACTIVATIONS`, bounded by
/// `SKIPPY_CAPTURE_BOUNDARY_MAX_MB` (default 512). Failures are logged and
/// ignored: capture must never break serving.
pub(crate) fn maybe_capture_boundary_activations(payload: &[u8]) {
    use std::io::Write as _;
    let Some(path) = std::env::var_os("SKIPPY_CAPTURE_BOUNDARY_ACTIVATIONS") else {
        return;
    };
    let max_bytes = std::env::var("SKIPPY_CAPTURE_BOUNDARY_MAX_MB")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(512)
        .saturating_mul(1024 * 1024);
    let path = std::path::PathBuf::from(path);
    if let Ok(metadata) = fs::metadata(&path)
        && metadata.len().saturating_add(payload.len() as u64) > max_bytes
    {
        return;
    }
    let result = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| file.write_all(payload));
    if let Err(error) = result {
        eprintln!("boundary activation capture failed: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn low_rank_samples(n: usize, d: usize, rank: usize) -> Vec<f32> {
        // Rows are combinations of `rank` fixed basis directions plus a mean
        // offset, so a rank-k codec should reconstruct them almost exactly.
        let basis = deterministic_matrix(d, rank);
        let coefficients = deterministic_matrix(rank, n);
        let mut out = Vec::with_capacity(n * d);
        for row_index in 0..n {
            for column in 0..d {
                let mut value = 0.5; // constant mean offset
                for basis_index in 0..rank {
                    value += coefficients[row_index * rank + basis_index]
                        * basis[basis_index * d + column]
                        * 3.0;
                }
                out.push(value);
            }
        }
        out
    }

    fn payload_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn payload_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn fit_encode_decode_roundtrips_low_rank_data() {
        let (n, d, rank) = (64, 32, 4);
        let samples = low_rank_samples(n, d, rank);
        let codec = BoundaryCodec::fit(&samples, d, rank, 12).unwrap();

        let token_count = 3;
        let original = &samples[..token_count * d];
        let encoded = codec.encode(&payload_bytes(original), token_count).unwrap();
        assert_eq!(encoded.len(), token_count * 4 + token_count * rank);

        let decoded = payload_values(&codec.decode(&encoded, token_count).unwrap());
        assert_eq!(decoded.len(), original.len());
        let mut max_error = 0.0_f32;
        let mut max_abs = 0.0_f32;
        for (reconstructed, value) in decoded.iter().zip(original) {
            max_error = max_error.max((reconstructed - value).abs());
            max_abs = max_abs.max(value.abs());
        }
        // Exactly-low-rank data must reconstruct to int8 quantization noise.
        assert!(
            max_error <= max_abs * 0.02,
            "max_error {max_error} vs max_abs {max_abs}"
        );
    }

    #[test]
    fn save_load_roundtrips_exactly() {
        let samples = low_rank_samples(32, 16, 3);
        let codec = BoundaryCodec::fit(&samples, 16, 3, 8).unwrap();
        let dir = std::env::temp_dir().join("skippy-boundary-codec-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("codec.skbc");
        codec.save(&path).unwrap();
        let loaded = BoundaryCodec::load(&path).unwrap();
        assert_eq!(codec, loaded);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn encode_rejects_mismatched_payload_sizes() {
        let samples = low_rank_samples(32, 16, 3);
        let codec = BoundaryCodec::fit(&samples, 16, 3, 4).unwrap();
        assert!(codec.encode(&[0_u8; 12], 1).is_err());
        assert!(codec.decode(&[0_u8; 5], 1).is_err());
    }

    #[test]
    fn fit_rejects_degenerate_shapes() {
        assert!(BoundaryCodec::fit(&[], 4, 2, 4).is_err());
        assert!(BoundaryCodec::fit(&[0.0; 8], 4, 8, 4).is_err());
        assert!(BoundaryCodec::fit(&[0.0; 9], 4, 2, 4).is_err());
    }
}

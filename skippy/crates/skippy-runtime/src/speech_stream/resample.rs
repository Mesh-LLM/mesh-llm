//! Streaming conversion of 16-bit PCM at any rate to the 16 kHz float PCM
//! Whisper takes.

use crate::speech_recognition::SPEECH_SAMPLE_RATE;

/// Input samples on each side of an output sample that the filter reads.
const HALF_TAPS: usize = 16;

/// Converts chunked 16-bit mono PCM to 16 kHz `f32`, carrying filter history
/// across chunks so chunk boundaries leave no seams. Downsampling low-passes
/// at the output Nyquist rate with a Hann-windowed sinc, so content above
/// 8 kHz does not alias into the speech band.
///
/// The last [`HALF_TAPS`] input samples of a stream wait for look-ahead and
/// are never emitted: about 1 ms, which a stream that ends a turn on silence
/// does not miss.
#[derive(Debug, Clone)]
pub struct PcmResampler {
    input_rate: u32,
    /// Input samples not yet fully consumed, starting at `history_start`.
    history: Vec<f32>,
    /// Absolute input index of `history[0]`.
    history_start: u64,
    /// Absolute input position, in input samples, of the next output sample.
    next_output: f64,
}

impl PcmResampler {
    pub fn new(input_rate: u32) -> Self {
        Self {
            input_rate: input_rate.max(1),
            history: Vec::new(),
            history_start: 0,
            next_output: 0.0,
        }
    }

    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    /// Converts one chunk; output for its last few input samples waits for
    /// the next chunk, which the filter needs to look ahead into.
    pub fn push(&mut self, input: &[i16]) -> Vec<f32> {
        self.history
            .extend(input.iter().map(|sample| f32::from(*sample) / 32_768.0));
        if self.input_rate == SPEECH_SAMPLE_RATE {
            let out = std::mem::take(&mut self.history);
            self.history_start += out.len() as u64;
            return out;
        }
        let step = f64::from(self.input_rate) / f64::from(SPEECH_SAMPLE_RATE);
        // Cutoff as a fraction of the input Nyquist rate; 1.0 when upsampling.
        let cutoff = (f64::from(SPEECH_SAMPLE_RATE) / f64::from(self.input_rate)).min(1.0);
        let available_end = self.history_start + self.history.len() as u64;
        let mut out = Vec::new();
        while (self.next_output.floor() as u64) + (HALF_TAPS as u64) < available_end {
            out.push(self.sample_at(self.next_output, cutoff));
            self.next_output += step;
        }
        // Drop history the next output can no longer reach.
        let keep_from = (self.next_output.floor() as u64).saturating_sub(HALF_TAPS as u64);
        if keep_from > self.history_start {
            let drop = ((keep_from - self.history_start) as usize).min(self.history.len());
            self.history.drain(..drop);
            self.history_start += drop as u64;
        }
        out
    }

    fn sample_at(&self, position: f64, cutoff: f64) -> f32 {
        let center = position.floor() as i64;
        let mut acc = 0.0f64;
        let mut gain = 0.0f64;
        for index in (center - HALF_TAPS as i64 + 1)..=(center + HALF_TAPS as i64) {
            if index < self.history_start as i64 {
                continue;
            }
            let Some(sample) = self
                .history
                .get((index - self.history_start as i64) as usize)
            else {
                continue;
            };
            let distance = position - index as f64;
            let window =
                0.5 + 0.5 * (std::f64::consts::PI * distance / (HALF_TAPS as f64 + 1.0)).cos();
            let weight = cutoff * sinc(cutoff * distance) * window;
            acc += f64::from(*sample) * weight;
            gain += weight;
        }
        // Normalize per output position: the truncated kernel's gain varies
        // with the fractional phase, which would ripple the amplitude.
        if gain.abs() > f64::EPSILON {
            acc /= gain;
        }
        acc as f32
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f64, seconds: f64) -> Vec<i16> {
        (0..(f64::from(rate) * seconds) as usize)
            .map(|n| {
                let t = n as f64 / f64::from(rate);
                ((2.0 * std::f64::consts::PI * hz * t).sin() * 16_000.0) as i16
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f64 {
        (samples.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / samples.len() as f64).sqrt()
    }

    #[test]
    fn sixteen_khz_passes_through_unchanged() {
        let mut resampler = PcmResampler::new(16_000);
        assert_eq!(resampler.push(&[0, 16_384, -32_768]), vec![0.0, 0.5, -1.0]);
    }

    #[test]
    fn downsampling_keeps_the_rate_ratio_across_chunks() {
        let input = tone(24_000, 440.0, 1.0);
        let mut resampler = PcmResampler::new(24_000);
        let out = input
            .chunks(480)
            .flat_map(|chunk| resampler.push(chunk))
            .collect::<Vec<_>>();
        // All but the filter's look-ahead tail is emitted: 2/3 of the input.
        assert!((15_980..=16_000).contains(&out.len()), "{}", out.len());
        // A 440 Hz tone survives at its original level (0.49 amplitude RMS).
        let level = rms(&out[100..out.len() - 100]);
        assert!((level - 0.345).abs() < 0.01, "{level}");
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let input = tone(24_000, 1_000.0, 0.25);
        let mut whole = PcmResampler::new(24_000);
        let a = whole.push(&input);
        let mut split = PcmResampler::new(24_000);
        let b = input
            .chunks(7)
            .flat_map(|chunk| split.push(chunk))
            .collect::<Vec<_>>();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn forty_eight_khz_downsamples_and_eight_khz_upsamples_at_the_right_level() {
        for rate in [48_000, 8_000] {
            let input = tone(rate, 440.0, 1.0);
            let mut resampler = PcmResampler::new(rate);
            let out = input
                .chunks(rate as usize / 50)
                .flat_map(|chunk| resampler.push(chunk))
                .collect::<Vec<_>>();
            assert!(
                (15_950..=16_000).contains(&out.len()),
                "{rate}: {}",
                out.len()
            );
            let level = rms(&out[200..out.len() - 200]);
            assert!((level - 0.345).abs() < 0.01, "{rate}: {level}");
        }
    }

    #[test]
    fn a_constant_signal_stays_constant_at_every_phase() {
        let mut resampler = PcmResampler::new(24_000);
        let out = resampler.push(&vec![16_384; 4_800]);
        for sample in &out[40..out.len() - 40] {
            assert!((sample - 0.5).abs() < 1e-4, "{sample}");
        }
    }

    #[test]
    fn content_above_the_output_nyquist_is_filtered_out() {
        let input = tone(24_000, 11_000.0, 0.5);
        let mut resampler = PcmResampler::new(24_000);
        let out = resampler.push(&input);
        assert!(rms(&out[100..out.len() - 100]) < 0.02);
    }
}

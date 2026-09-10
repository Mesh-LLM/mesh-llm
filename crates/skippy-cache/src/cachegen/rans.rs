//! Byte-aligned rANS entropy coder, the static-CDF variant CacheGen uses.
//!
//! Ported from Fabian Giesen's public-domain `ryg_rans` (`rans_byte.h`),
//! which is also the coder LMCache ships. This is the deterministic CPU
//! reference: the GPU slices in the later CubeCL work must produce
//! byte-identical streams. The straightforward divide/mod operations are
//! used on purpose — this crate is the correctness oracle, not the
//! performance path.
//!
//! Encoding processes symbols in reverse order and emits bytes backwards;
//! [`RansEncoder::finish`] returns the stream the decoder consumes
//! forwards.

/// Lower bound of the normalization interval (`RANS_BYTE_L`).
pub const RANS_L: u32 = 1 << 23;
/// CDF precision. Frequencies sum to `1 << SCALE_BITS` (CacheGen uses 12).
pub const SCALE_BITS: u32 = 12;
/// Total frequency, `2^SCALE_BITS`.
pub const SCALE: u32 = 1 << SCALE_BITS;

/// Forward symbol table: cumulative start and frequency per token.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Cumulative frequency at each symbol's range start.
    pub starts: Vec<u32>,
    /// Frequency of each symbol. `starts[0] == 0` and
    /// `starts.last() + freq.last() == SCALE`.
    pub freqs: Vec<u32>,
}

impl SymbolTable {
    /// Builds the table from per-symbol frequencies. An all-zero histogram
    /// is refused rather than silently producing an undecodable table.
    pub fn from_freqs(freqs: &[u32]) -> Option<Self> {
        if freqs.is_empty() || freqs.iter().any(|&freq| freq > SCALE) {
            return None;
        }
        let total: u64 = freqs.iter().map(|&freq| u64::from(freq)).sum();
        if total != u64::from(SCALE) {
            return None;
        }
        let mut starts = Vec::with_capacity(freqs.len());
        let mut running = 0u32;
        for &freq in freqs {
            starts.push(running);
            running = running.checked_add(freq)?;
        }
        Some(Self {
            starts,
            freqs: freqs.to_vec(),
        })
    }

    /// The symbol whose range contains `value` (`RansDecGet` output).
    pub fn symbol_for(&self, value: u32) -> usize {
        let position = self
            .starts
            .partition_point(|&start| start <= value)
            .saturating_sub(1);
        position.min(self.freqs.len() - 1)
    }
}

/// Static-frequency byte-rANS encoder.
#[derive(Debug)]
pub struct RansEncoder {
    state: u32,
    /// Emitted bytes in time order; reversed into the final stream.
    emitted: Vec<u8>,
}

impl RansEncoder {
    pub fn new() -> Self {
        Self {
            state: RANS_L,
            emitted: Vec::new(),
        }
    }

    /// Encodes one symbol. Symbols must be pushed in reverse stream order
    /// (last symbol first).
    pub fn put(&mut self, table: &SymbolTable, symbol: usize) {
        let start = table.starts[symbol];
        let freq = table.freqs[symbol];
        let x_max = ((RANS_L >> SCALE_BITS) << 8) * freq;
        while self.state >= x_max {
            self.emitted.push((self.state & 0xff) as u8);
            self.state >>= 8;
        }
        self.state = ((self.state / freq) << SCALE_BITS) + (self.state % freq) + start;
    }

    /// Flushes the state and returns the wire stream.
    pub fn finish(mut self) -> Vec<u8> {
        let mut stream = self.state.to_le_bytes().to_vec();
        stream.extend(self.emitted.drain(..).rev());
        stream
    }
}

impl Default for RansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Static-frequency byte-rANS decoder over a complete stream.
pub struct RansDecoder<'a> {
    state: u32,
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> RansDecoder<'a> {
    /// Starts decoding; consumes the initial 4-byte little-endian state.
    pub fn new(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let state = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Some(Self {
            state,
            bytes,
            cursor: 4,
        })
    }

    /// Decodes the next symbol (stream order).
    pub fn get(&mut self, table: &SymbolTable) -> Option<usize> {
        let value = self.state & (SCALE - 1);
        let symbol = table.symbol_for(value);
        let start = table.starts[symbol];
        let freq = table.freqs[symbol];
        let mut x = freq * (self.state >> SCALE_BITS) + value - start;
        while x < RANS_L {
            let byte = *self.bytes.get(self.cursor)?;
            self.cursor += 1;
            x = (x << 8) | u32::from(byte);
        }
        self.state = x;
        Some(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_symbol_table() -> SymbolTable {
        SymbolTable::from_freqs(&[SCALE / 2, SCALE - SCALE / 2]).expect("table")
    }

    #[test]
    fn round_trips_a_skewed_two_symbol_stream() {
        let table = two_symbol_table();
        // Skewed pattern with long runs in both directions.
        let symbols: Vec<usize> = (0..1000)
            .map(|index| if (index / 7) % 5 == 0 { 1 } else { 0 })
            .collect();
        let mut encoder = RansEncoder::new();
        for symbol in symbols.iter().rev() {
            encoder.put(&table, *symbol);
        }
        let stream = encoder.finish();

        let mut decoder = RansDecoder::new(&stream).expect("decoder");
        let mut decoded = Vec::with_capacity(symbols.len());
        for _ in 0..symbols.len() {
            decoded.push(decoder.get(&table).expect("symbol"));
        }
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn rejects_frequencies_that_do_not_sum_to_scale() {
        assert!(SymbolTable::from_freqs(&[1, 2, 3]).is_none());
        assert!(SymbolTable::from_freqs(&[]).is_none());
        assert!(SymbolTable::from_freqs(&[SCALE + 1]).is_none());
        assert!(SymbolTable::from_freqs(&[SCALE]).is_some());
    }

    #[test]
    fn single_symbol_stream_still_round_trips() {
        let table = SymbolTable::from_freqs(&[SCALE]).expect("table");
        let mut encoder = RansEncoder::new();
        for _ in 0..64 {
            encoder.put(&table, 0);
        }
        let stream = encoder.finish();
        let mut decoder = RansDecoder::new(&stream).expect("decoder");
        for _ in 0..64 {
            assert_eq!(decoder.get(&table).expect("symbol"), 0);
        }
    }
}

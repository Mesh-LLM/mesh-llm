//! A split GGUF (`name-00001-of-00003.gguf` and its siblings) read as one
//! file: the shards' bytes end to end. Each shard has its own header and
//! the tensors it holds; [`WeightFile::open_shards`] reads them all and
//! places each tensor at its offset in the whole.
//!
//! The name of the first shard says how many there are, and a node chooses
//! the name: more than [`MAX_SHARDS`] is refused before any is opened.
//!
//! [`WeightFile::open_shards`]: crate::weights::WeightFile::open_shards

use anyhow::{Context, Result, bail};

use crate::weights::ByteSource;
use crate::weights::limits::{MAX_SHARDS, refuse};

/// The shards of a split GGUF, end to end.
pub(crate) struct Shards {
    /// Each shard with the offset of its first byte in the whole.
    parts: Vec<(u64, Box<dyn ByteSource>)>,
    size: u64,
}

impl Shards {
    pub fn new(shards: Vec<Box<dyn ByteSource>>) -> Result<Self> {
        let mut size = 0u64;
        let mut parts = Vec::with_capacity(shards.len());
        for shard in shards {
            let base = size;
            size = size
                .checked_add(shard.size())
                .context("the shards are larger than any file")?;
            parts.push((base, shard));
        }
        Ok(Self { parts, size })
    }

    /// Each shard and the offset of its first byte in the whole.
    pub fn parts(&self) -> impl Iterator<Item = (u64, &dyn ByteSource)> {
        self.parts
            .iter()
            .map(|(base, shard)| (*base, shard.as_ref()))
    }

    /// The pieces of `[offset, offset + len)` in each shard: the shard's
    /// index, the offset in the shard, and the length.
    fn pieces(&self, offset: u64, len: u64) -> Result<Vec<(usize, u64, u64)>> {
        let end = offset.saturating_add(len);
        if end > self.size {
            bail!("bytes {offset}..{end} are past the end of the shards");
        }
        Ok(self
            .parts
            .iter()
            .enumerate()
            .filter_map(|(index, (base, shard))| {
                let start = offset.max(*base);
                // the shards' sizes sum to `size`, so this does not wrap
                let stop = end.min(base + shard.size());
                (start < stop).then(|| (index, start - base, stop - start))
            })
            .collect())
    }

    /// `ranges` as the ranges of each shard they lie in.
    fn per_shard(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<(u64, u64)>>> {
        let mut per_shard = vec![Vec::new(); self.parts.len()];
        for &(offset, len) in ranges {
            for (index, at, n) in self.pieces(offset, len)? {
                per_shard[index].push((at, n));
            }
        }
        Ok(per_shard)
    }

    fn read_with(
        &self,
        offset: u64,
        len: u64,
        read: impl Fn(&dyn ByteSource, u64, u64) -> Result<Vec<u8>>,
    ) -> Result<Vec<u8>> {
        // the pieces first: a length past the shards allocates nothing
        let pieces = self.pieces(offset, len)?;
        let mut out = Vec::with_capacity(usize::try_from(len)?);
        for (index, at, n) in pieces {
            out.extend(read(self.parts[index].1.as_ref(), at, n)?);
        }
        Ok(out)
    }
}

impl ByteSource for Shards {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.read_with(offset, len, |shard, at, n| shard.read_at(at, n))
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn prefetch(&self, ranges: &[(u64, u64)]) -> Result<()> {
        for ((_, shard), ranges) in self.parts.iter().zip(self.per_shard(ranges)?) {
            if !ranges.is_empty() {
                shard.prefetch(&ranges)?;
            }
        }
        Ok(())
    }

    fn read_once(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.read_with(offset, len, |shard, at, n| shard.read_once(at, n))
    }

    fn missing_bytes(&self, ranges: &[(u64, u64)]) -> Result<u64> {
        let mut missing = 0u64;
        for ((_, shard), ranges) in self.parts.iter().zip(self.per_shard(ranges)?) {
            if !ranges.is_empty() {
                missing = missing.saturating_add(shard.missing_bytes(&ranges)?);
            }
        }
        Ok(missing)
    }
}

/// The files of the split GGUF whose first shard is `file`, in order, or
/// `None` when `file` is not the first shard of a split. A split of more
/// than [`MAX_SHARDS`] is refused: each shard costs requests and a header.
pub fn shard_files(file: &str) -> Result<Option<Vec<String>>> {
    let Some((prefix, total)) = first_shard(file) else {
        return Ok(None);
    };
    let count: usize = total.parse()?;
    if count <= 1 {
        return Ok(None);
    }
    if count > MAX_SHARDS {
        return refuse(format!(
            "{file} names a split of {count} shards, more than {MAX_SHARDS}"
        ));
    }
    Ok(Some(
        (1..=count)
            .map(|n| format!("{prefix}-{n:05}-of-{total}.gguf"))
            .collect(),
    ))
}

/// The prefix and the five-digit shard count of `file`, when it names the
/// first shard of a split.
fn first_shard(file: &str) -> Option<(&str, &str)> {
    let stem = file.strip_suffix(".gguf")?;
    let (prefix_and_part, total) = stem.rsplit_once("-of-")?;
    let (prefix, part) = prefix_and_part.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    (digits(part) && digits(total) && part == "00001").then_some((prefix, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::limits::CannotAudit;
    use crate::weights::test_gguf::Memory;

    fn shards() -> Shards {
        Shards::new(vec![
            Box::new(Memory((0u8..10).collect())),
            Box::new(Memory((10u8..15).collect())),
            Box::new(Memory((15u8..30).collect())),
        ])
        .unwrap()
    }

    #[test]
    fn reads_cross_shard_boundaries() {
        let shards = shards();
        assert_eq!(shards.size(), 30);
        assert_eq!(shards.read_at(8, 9).unwrap(), (8u8..17).collect::<Vec<_>>());
        assert_eq!(
            shards.read_at(10, 5).unwrap(),
            (10u8..15).collect::<Vec<_>>()
        );
        assert!(shards.read_at(25, 6).is_err());
        // a length that wraps is past the end, and allocates nothing
        assert!(shards.read_at(25, u64::MAX).is_err());
        assert!(shards.read_once(u64::MAX, 2).is_err());
    }

    /// A shard that says it is as large as a file can be.
    struct Huge;

    impl ByteSource for Huge {
        fn read_at(&self, _offset: u64, _len: u64) -> Result<Vec<u8>> {
            bail!("never read")
        }

        fn size(&self) -> u64 {
            u64::MAX
        }
    }

    #[test]
    fn shards_larger_than_any_file_are_refused() {
        assert!(Shards::new(vec![Box::new(Huge), Box::new(Memory(vec![0]))]).is_err());
    }

    #[test]
    fn the_first_shard_names_its_siblings() {
        assert_eq!(
            shard_files("q/Model-Q4_K_M-00001-of-00003.gguf")
                .unwrap()
                .unwrap(),
            vec![
                "q/Model-Q4_K_M-00001-of-00003.gguf",
                "q/Model-Q4_K_M-00002-of-00003.gguf",
                "q/Model-Q4_K_M-00003-of-00003.gguf",
            ]
        );
        let whole = |file: &str| shard_files(file).unwrap().is_none();
        assert!(whole("Model-00002-of-00003.gguf"));
        assert!(whole("Model-00001-of-00001.gguf"));
        assert!(whole("Model-Q8_0.gguf"));
    }

    #[test]
    fn a_split_of_too_many_shards_is_refused() {
        let most = format!("m-00001-of-{MAX_SHARDS:05}.gguf");
        assert_eq!(shard_files(&most).unwrap().unwrap().len(), MAX_SHARDS);
        for file in [
            format!("m-00001-of-{:05}.gguf", MAX_SHARDS + 1),
            "m-00001-of-99999.gguf".to_owned(),
        ] {
            let refused = shard_files(&file).unwrap_err();
            assert!(CannotAudit::caused(&refused), "{refused:#}");
        }
    }
}

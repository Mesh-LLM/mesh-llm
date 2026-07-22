use std::ptr::{self, NonNull};

use anyhow::{Context, Result, bail};

/// llama.cpp's stateful N-gram cache supports match windows up to four tokens.
/// The proposal length remains independently configurable.
pub const NGRAM_CACHE_MAX_NGRAM: usize = 4;

/// Stateful adapter for llama.cpp's cache-based N-gram proposer.
///
/// Callers must feed only target-committed history through [`Self::reset`] or
/// [`Self::append`]. `draft_after` may include provisional tokens, but native
/// state is not mutated while producing that candidate.
pub struct Cache {
    raw: NonNull<skippy_ffi::NgramCache>,
}

impl Cache {
    pub fn new(ngram_min: usize, ngram_max: usize) -> Result<Self> {
        if ngram_min == 0 || ngram_min > ngram_max {
            bail!("cache N-gram proposer requires 0 < ngram_min <= ngram_max");
        }
        if ngram_max > NGRAM_CACHE_MAX_NGRAM {
            bail!(
                "cache N-gram proposer ngram_max {ngram_max} exceeds llama.cpp limit {NGRAM_CACHE_MAX_NGRAM}"
            );
        }
        let ngram_min = u16::try_from(ngram_min).context("cache N-gram minimum exceeds limit")?;
        let ngram_max = u16::try_from(ngram_max).context("cache N-gram maximum exceeds limit")?;
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_ngram_cache_create(ngram_min, ngram_max, &mut raw, &mut error)
        };
        super::ensure_ok(status, error)?;
        let raw = NonNull::new(raw).context("llama.cpp created a null N-gram cache")?;
        Ok(Self { raw })
    }

    pub fn reset(&mut self, history: &[i32]) -> Result<()> {
        self.update(history, true)
    }

    pub fn append(&mut self, committed_tokens: &[i32]) -> Result<()> {
        if committed_tokens.is_empty() {
            return Ok(());
        }
        self.update(committed_tokens, false)
    }

    pub fn draft_after(
        &mut self,
        continuation_prefix: &[i32],
        max_draft_tokens: usize,
    ) -> Result<Vec<i32>> {
        if max_draft_tokens == 0 {
            return Ok(Vec::new());
        }
        let max_draft_tokens =
            u16::try_from(max_draft_tokens).context("cache N-gram draft limit exceeds limit")?;
        let mut output_tokens = vec![0_i32; usize::from(max_draft_tokens)];
        let mut output_token_count = 0_usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_ngram_cache_draft(
                self.raw.as_ptr(),
                continuation_prefix.as_ptr(),
                continuation_prefix.len(),
                max_draft_tokens,
                output_tokens.as_mut_ptr(),
                output_tokens.len(),
                &mut output_token_count,
                &mut error,
            )
        };
        super::ensure_ok(status, error)?;
        if output_token_count > output_tokens.len() {
            bail!("llama.cpp cache N-gram proposer exceeded its requested draft limit");
        }
        output_tokens.truncate(output_token_count);
        Ok(output_tokens)
    }

    fn update(&mut self, tokens: &[i32], reset: bool) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            if reset {
                skippy_ffi::skippy_ngram_cache_reset(
                    self.raw.as_ptr(),
                    tokens.as_ptr(),
                    tokens.len(),
                    &mut error,
                )
            } else {
                skippy_ffi::skippy_ngram_cache_append(
                    self.raw.as_ptr(),
                    tokens.as_ptr(),
                    tokens.len(),
                    &mut error,
                )
            }
        };
        super::ensure_ok(status, error)
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        unsafe { skippy_ffi::skippy_ngram_cache_free(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::{Cache, NGRAM_CACHE_MAX_NGRAM};

    #[test]
    fn cache_drafts_from_committed_history_and_never_mutates_for_a_prefix() {
        let mut cache = Cache::new(2, 2).unwrap();
        cache.reset(&[1, 2, 3, 1, 2, 3, 1, 2]).unwrap();

        assert_eq!(cache.draft_after(&[], 2).unwrap(), vec![3, 1]);
        assert_eq!(cache.draft_after(&[9], 2).unwrap(), Vec::<i32>::new());
        assert_eq!(cache.draft_after(&[], 2).unwrap(), vec![3, 1]);
    }

    #[test]
    fn cache_append_extends_the_committed_history() {
        let mut cache = Cache::new(2, 2).unwrap();
        cache.reset(&[1, 9, 7, 1, 9, 7, 1]).unwrap();

        assert_eq!(cache.draft_after(&[9], 1).unwrap(), vec![7]);
        cache.append(&[9, 7, 1]).unwrap();
        assert_eq!(cache.draft_after(&[9], 1).unwrap(), vec![7]);
    }

    #[test]
    fn cache_rejects_match_windows_above_the_llama_limit() {
        let error = match Cache::new(2, NGRAM_CACHE_MAX_NGRAM + 1) {
            Ok(_) => panic!("must reject max > 4"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("exceeds llama.cpp limit 4"));
    }
}

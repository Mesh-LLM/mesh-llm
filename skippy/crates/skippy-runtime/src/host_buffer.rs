/// A zeroed buffer for native code to fill, with every page already faulted in.
///
/// `vec![0; len]` maps a large allocation lazily, so the first write to each page is
/// the device copy itself. On unified-memory GPUs (DGX Spark) a device-to-host copy into
/// pages the CPU never touched faults page by page: a 64 MiB state export ran at 0.16 GB/s
/// against 53 GB/s into resident pages. One volatile write per page faults them in on the
/// CPU first; a plain zero fill could be folded back into the lazy zeroed allocation.
pub(crate) fn native_output_buffer(len: usize) -> Vec<u8> {
    const PAGE: usize = 4096;
    let mut buffer = vec![0_u8; len];
    let base = buffer.as_mut_ptr();
    for offset in (0..len).step_by(PAGE) {
        // SAFETY: `offset < len`, so the write stays inside the allocation.
        unsafe { std::ptr::write_volatile(base.add(offset), 0) };
    }
    buffer
}

#[cfg(test)]
mod tests {
    use super::native_output_buffer;

    #[test]
    fn buffer_is_zeroed_and_sized() {
        let buffer = native_output_buffer((3 << 20) + 17);
        assert_eq!(buffer.len(), (3 << 20) + 17);
        assert!(buffer.iter().all(|byte| *byte == 0));
        assert!(native_output_buffer(0).is_empty());
    }
}

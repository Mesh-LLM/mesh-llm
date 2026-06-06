pub struct PagedKV {
    block_size: usize,
    num_blocks: usize,
}

impl PagedKV {
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        Self { block_size, num_blocks }
    }
}

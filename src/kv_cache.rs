pub struct ContinuousBatching {
    max_batch_size: usize,
}

impl ContinuousBatching {
    pub fn new(max_batch_size: usize) -> Self {
        Self { max_batch_size }
    }
}

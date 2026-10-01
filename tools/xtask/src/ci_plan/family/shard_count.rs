pub(super) struct ShardCount(u64);

impl ShardCount {
    pub(super) fn parse(text: &str) -> Option<Self> {
        serde_json::from_str::<u64>(text).ok().map(Self)
    }

    pub(super) fn effective(&self, selected: usize) -> Result<usize, String> {
        if self.0 == 0 {
            return Err("--shard-count must be positive".into());
        }
        let selected = u64::try_from(selected).map_err(|error| error.to_string())?;
        usize::try_from(self.0.min(selected)).map_err(|error| error.to_string())
    }
}

impl From<usize> for ShardCount {
    fn from(value: usize) -> Self {
        Self(u64::try_from(value).unwrap_or(u64::MAX))
    }
}

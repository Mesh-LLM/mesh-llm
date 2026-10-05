#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MultipartDownloadProgress {
    label: String,
    completed: usize,
    total: usize,
}

impl MultipartDownloadProgress {
    pub(crate) fn new(label: impl Into<String>, total: usize) -> Self {
        Self {
            label: label.into(),
            completed: 0,
            total,
        }
    }

    pub(crate) fn is_multipart(&self) -> bool {
        self.total > 1
    }

    pub(crate) fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn snapshot(&self) -> (usize, usize) {
        (self.completed, self.total)
    }

    pub(crate) fn complete_optional_metadata(&mut self) {}

    pub(crate) fn complete_required_part(&mut self) {
        self.completed = self.completed.saturating_add(1).min(self.total);
    }
}

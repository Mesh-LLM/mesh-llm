//! Cold exports read immutable cache owners, never a borrowed execution lane.
use super::*;

impl RuntimeState {
    pub(crate) fn resident_kv_page_desc(
        &mut self,
        session: &str,
        sequence: i32,
        tokens: u64,
    ) -> Result<skippy_runtime::RuntimeKvPageDesc> {
        let layers = (
            i32::try_from(self.model_layer_start())?,
            i32::try_from(self.model_layer_end())?,
        );
        self.session(session)?
            .resident_kv_page_desc(sequence, layers, tokens)
    }

    pub(crate) fn export_resident_kv_page(
        &mut self,
        session: &str,
        sequence: i32,
        tokens: u64,
        bytes: usize,
    ) -> Result<skippy_runtime::RuntimeKvPage> {
        let layers = (
            i32::try_from(self.model_layer_start())?,
            i32::try_from(self.model_layer_end())?,
        );
        self.session(session)?
            .export_resident_kv_page(sequence, layers, tokens, bytes)
    }

    pub(crate) fn resident_recurrent_state_bytes(
        &mut self,
        session: &str,
        sequence: i32,
    ) -> Result<usize> {
        self.session(session)?
            .resident_recurrent_state_bytes(sequence)
    }

    pub(crate) fn export_resident_recurrent_state(
        &mut self,
        session: &str,
        sequence: i32,
        bytes: usize,
    ) -> Result<Vec<u8>> {
        self.session(session)?
            .export_resident_recurrent_state(sequence, bytes)
    }
}

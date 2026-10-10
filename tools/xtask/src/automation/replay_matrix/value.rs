use crate::automation::codepoint_json::value::Value;

impl Value {
    pub(super) fn repr(&self) -> String {
        super::serialization::render(self)
            .trim_end_matches('\n')
            .to_owned()
    }
}

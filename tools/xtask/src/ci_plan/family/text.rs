#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct FamilyString(Vec<u32>);

impl From<&str> for FamilyString {
    fn from(text: &str) -> Self {
        Self(text.chars().map(u32::from).collect())
    }
}

impl From<String> for FamilyString {
    fn from(text: String) -> Self {
        Self::from(text.as_str())
    }
}

impl PartialEq<str> for FamilyString {
    fn eq(&self, text: &str) -> bool {
        self.0.iter().copied().eq(text.chars().map(u32::from))
    }
}

impl FamilyString {
    pub(super) fn codes(&self) -> &[u32] {
        &self.0
    }

    pub(super) fn scalar_text(&self) -> Option<String> {
        self.0.iter().copied().map(char::from_u32).collect()
    }

    pub(super) fn diagnostic(&self) -> String {
        let text = self
            .scalar_text()
            .expect("JSON input strings contain Unicode scalars");
        serde_json::to_string(&text).expect("JSON strings serialize")
    }

    pub(super) fn write_json(&self, output: &mut String) {
        output.push('"');
        for &code in &self.0 {
            let Some(character) = char::from_u32(code) else {
                output.push_str(&format!("\\u{code:04x}"));
                continue;
            };
            match character {
                '"' => output.push_str("\\\""),
                '\\' => output.push_str("\\\\"),
                '\n' => output.push_str("\\n"),
                '\r' => output.push_str("\\r"),
                '\t' => output.push_str("\\t"),
                '\u{8}' => output.push_str("\\b"),
                '\u{c}' => output.push_str("\\f"),
                ' '..='~' => output.push(character),
                _ => {
                    let mut units = [0_u16; 2];
                    for unit in character.encode_utf16(&mut units) {
                        output.push_str(&format!("\\u{unit:04x}"));
                    }
                }
            }
        }
        output.push('"');
    }
}

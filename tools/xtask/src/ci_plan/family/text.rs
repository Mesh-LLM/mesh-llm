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
        let mut text = String::new();
        for &code in &self.0 {
            match char::from_u32(code) {
                Some(character) => text.push(character),
                None => text.push_str(&format!("\\u{code:04x}")),
            }
        }
        text
    }

    pub(super) fn repr(&self) -> String {
        let quote = if self.0.contains(&u32::from('\'')) && !self.0.contains(&u32::from('"')) {
            '"'
        } else {
            '\''
        };
        let mut output = String::from(quote);
        for &code in &self.0 {
            match char::from_u32(code) {
                None => output.push_str(&format!("\\u{code:04x}")),
                Some('\\') => output.push_str("\\\\"),
                Some('\n') => output.push_str("\\n"),
                Some('\r') => output.push_str("\\r"),
                Some('\t') => output.push_str("\\t"),
                Some(character) if character == quote => {
                    output.push('\\');
                    output.push(character);
                }
                Some(character)
                    if character < ' ' || ('\u{7f}'..='\u{a0}').contains(&character) =>
                {
                    output.push_str(&format!("\\x{code:02x}"));
                }
                Some(character) => output.push(character),
            }
        }
        output.push(quote);
        output
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

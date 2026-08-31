pub(in super::super) fn tokenize(sql: &str) -> Vec<String> {
    let bytes = sql.as_bytes();
    let mut tokens = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        if bytes[start].is_ascii_whitespace() || bytes[start] == b',' {
            start += 1;
            continue;
        }
        if matches!(bytes[start], b'(' | b')') {
            tokens.push(char::from(bytes[start]).to_string());
            start += 1;
            continue;
        }
        if matches!(bytes[start], b'\'' | b'"' | b'`' | b'[') {
            let end = quoted_end(bytes, start);
            tokens.push(sql[start..end].to_owned());
            start = end;
            continue;
        }
        let mut end = start + 1;
        if matches!(bytes[start], b'<' | b'>' | b'=' | b'!') {
            if end < bytes.len() && matches!(bytes[end], b'=' | b'>') {
                end += 1;
            }
        } else {
            while end < bytes.len()
                && !bytes[end].is_ascii_whitespace()
                && !matches!(bytes[end], b',' | b'(' | b')' | b'<' | b'>' | b'=' | b'!')
            {
                end += 1;
            }
        }
        tokens.push(sql[start..end].to_ascii_uppercase());
        start = end;
    }
    tokens
}

fn quoted_end(bytes: &[u8], start: usize) -> usize {
    let opening = bytes[start];
    let closing = if opening == b'[' { b']' } else { opening };
    let mut end = start + 1;
    while end < bytes.len() {
        if bytes[end] != closing {
            end += 1;
            continue;
        }
        if closing != b']' && bytes.get(end + 1) == Some(&closing) {
            end += 2;
            continue;
        }
        return end + 1;
    }
    bytes.len()
}

pub(super) fn contains(actual: &[String], expected: &[String]) -> bool {
    actual
        .windows(expected.len())
        .any(|window| window == expected)
}

pub(super) fn tail(tokens: &[String], start: &str) -> Vec<String> {
    tokens
        .iter()
        .position(|token| token == start)
        .map_or_else(Vec::new, |position| tokens[position..].to_vec())
}

pub(super) fn checks(tokens: &[String]) -> Option<Vec<Vec<String>>> {
    let mut expressions = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = tokens[cursor..].iter().position(|token| token == "CHECK") {
        let check = cursor + offset;
        if tokens.get(check + 1).map(String::as_str) != Some("(") {
            return None;
        }
        let mut depth = 1_u32;
        let mut end = check + 2;
        while depth > 0 {
            match tokens.get(end).map(String::as_str) {
                Some("(") => depth += 1,
                Some(")") => depth -= 1,
                Some(_) => {}
                None => return None,
            }
            end += 1;
        }
        expressions.push(tokens[check + 2..end - 1].to_vec());
        cursor = end;
    }
    Some(expressions)
}

#[cfg(test)]
mod tests {
    use super::tokenize;

    #[test]
    fn quoted_sql_string_is_one_case_preserving_token_when_it_contains_sql_punctuation() {
        // Given
        let sql = "state IN ('Active value, (pending)! It''s exact.')";

        // When
        let tokens = tokenize(sql);

        // Then
        assert_eq!(
            tokens,
            [
                "STATE",
                "IN",
                "(",
                "'Active value, (pending)! It''s exact.'",
                ")"
            ]
        );
    }
}

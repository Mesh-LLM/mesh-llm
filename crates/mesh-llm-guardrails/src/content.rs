/// Removes hidden reasoning blocks from model-visible text.
///
/// This is content hygiene only. Tool calls are parsed by the model runtime's
/// native chat parser and are never recovered from assistant text here.
pub fn strip_thinking_blocks(content: &str) -> String {
    let stripped_html = strip_tag_pairs(content, "<think>", "</think>");
    let stripped_brackets = strip_tag_pairs(&stripped_html, "[THINK]", "[/THINK]");
    stripped_brackets.trim().to_owned()
}

fn strip_tag_pairs(content: &str, start_tag: &str, end_tag: &str) -> String {
    let mut remainder = content;
    let mut result = String::new();
    while let Some(start_index) = remainder.find(start_tag) {
        result.push_str(&remainder[..start_index]);
        let Some(end_index) = matching_tag_end(&remainder[start_index..], start_tag, end_tag)
        else {
            return result;
        };
        remainder = &remainder[start_index + end_index..];
    }
    result.push_str(remainder);
    result
}

fn matching_tag_end(content: &str, start_tag: &str, end_tag: &str) -> Option<usize> {
    let mut offset = 0;
    let mut depth = 0usize;
    while offset < content.len() {
        let next_start = content[offset..]
            .find(start_tag)
            .map(|index| offset + index);
        let next_end = content[offset..].find(end_tag).map(|index| offset + index);
        match (next_start, next_end) {
            (Some(start), Some(end)) if start < end => {
                depth += 1;
                offset = start + start_tag.len();
            }
            (_, Some(end)) if depth > 0 => {
                depth -= 1;
                offset = end + end_tag.len();
                if depth == 0 {
                    return Some(offset);
                }
            }
            _ => return None,
        }
    }
    None
}

fn matching_tag_end(content: &str, start_tag: &str, end_tag: &str) -> Option<usize> {
    let mut offset = 0;
    let mut depth = 0usize;
    while offset < content.len() {
        let next_start = content[offset..]
            .find(start_tag)
            .map(|index| offset + index);
        let next_end = content[offset..].find(end_tag).map(|index| offset + index);
        match (next_start, next_end) {
            (Some(start), Some(end)) if start < end => {
                depth += 1;
                offset = start + start_tag.len();
            }
            (_, Some(end)) if depth > 0 => {
                depth -= 1;
                offset = end + end_tag.len();
                if depth == 0 {
                    return Some(offset);
                }
            }
            _ => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_supported_thinking_blocks() {
        assert_eq!(
            strip_thinking_blocks("<think>hidden</think>Visible answer"),
            "Visible answer"
        );
        assert_eq!(
            strip_thinking_blocks("[THINK]hidden[/THINK]Visible answer"),
            "Visible answer"
        );
    }

    #[test]
    fn drops_unterminated_thinking_blocks_without_duplicating_prefix() {
        assert_eq!(strip_thinking_blocks("hello <think>truncated"), "hello");
    }

    #[test]
    fn strips_nested_thinking_blocks_completely() {
        assert_eq!(
            strip_thinking_blocks("before<think>outer<think>inner</think>tail</think>after"),
            "beforeafter"
        );
    }
}

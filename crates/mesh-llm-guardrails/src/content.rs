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
        let after_start = &remainder[start_index + start_tag.len()..];
        if let Some(end_index) = after_start.find(end_tag) {
            remainder = &after_start[end_index + end_tag.len()..];
        } else {
            return result;
        }
    }
    result.push_str(remainder);
    result
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
}

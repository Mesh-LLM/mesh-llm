use crate::command::DynResult;
use caseless::Caseless;
use unicode_normalization::UnicodeNormalization;

fn normalized(value: &str) -> DynResult<String> {
    let mut result = String::new();
    let mut separator = false;
    for character in value.nfkc().default_case_fold() {
        if character.is_alphanumeric() || character == '_' {
            if separator && !result.is_empty() {
                result.push(' ');
            }
            result.push(character);
            separator = false;
        } else {
            separator = true;
        }
    }
    if result.is_empty() {
        return Err("returned empty text".into());
    }
    Ok(result)
}
fn transcript(value: &str) -> DynResult<String> {
    let normalized = normalized(value)?;
    let text = ["the text is ", "the audio is "]
        .into_iter()
        .find_map(|prefix| normalized.strip_prefix(prefix))
        .unwrap_or(&normalized);
    if text.is_empty() || matches!(text, "the text is" | "the audio is") {
        return Err("returned empty transcript".into());
    }
    if [
        "i can t fulfill",
        "i cannot fulfill",
        "i can help you with transcribing",
        "you can use various tools to transcribe",
    ]
    .into_iter()
    .any(|prefix| text.starts_with(prefix))
    {
        return Err("returned a refusal or generic ASR advice, not a transcript".into());
    }
    Ok(text.into())
}
pub(super) fn compare(
    candidate: &str,
    reference: &str,
    expected: Option<&str>,
    asr: bool,
) -> DynResult<String> {
    let normalize = if asr { transcript } else { normalized };
    let candidate = normalize(candidate)?;
    let reference = normalize(reference)?;
    if candidate != reference {
        return Err("text differs from monolithic reference".into());
    }
    if let Some(expected) = expected {
        let expected = normalized(expected)?;
        if candidate != expected {
            return Err("output does not exactly match independently known fixture text".into());
        }
        return Ok(format!(
            "identical normalized text exactly matching {expected}"
        ));
    }
    Ok("identical normalized text; unlabeled fixture, no accuracy claim".into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_presentation_and_independent_labels() {
        assert!(compare("ＭＥＳＨ—42", "mesh 42", Some("MESH 42"), false).is_ok());
        assert!(compare("Straße Σς", "STRASSE σσ", None, true).is_ok());
        assert!(compare("The audio is: Ready!", "The text is ready", None, true).is_ok());
        assert!(compare("mesh 43", "mesh 43", Some("MESH 42"), false).is_err());
        assert!(compare("The audio is:", "The text is:", None, true).is_err());
        assert!(compare("I cannot fulfill that", "I cannot fulfill that", None, true).is_err());
        assert!(compare("---", "---", None, false).is_err());
    }
}

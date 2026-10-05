pub(crate) fn append_external_inference_models(
    models: &mut Vec<String>,
    external_models: &[String],
) {
    for model in external_models {
        if model.trim().is_empty() || models.iter().any(|existing| existing == model) {
            continue;
        }
        models.push(model.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::append_external_inference_models;

    #[test]
    fn append_external_inference_models_skips_blanks_and_duplicates() {
        let mut models = vec!["local".to_string(), "external".to_string()];
        let external_models = vec![
            String::new(),
            "  ".to_string(),
            "external".to_string(),
            "plugin".to_string(),
        ];

        append_external_inference_models(&mut models, &external_models);

        assert_eq!(models, vec!["local", "external", "plugin"]);
    }
}

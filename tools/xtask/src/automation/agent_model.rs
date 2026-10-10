use crate::command::DynResult;
use serde::Deserialize;
use std::io::Read;

#[derive(Deserialize)]
struct Models {
    #[serde(default)]
    data: Vec<Model>,
}

#[derive(Deserialize)]
struct Model {
    #[serde(default)]
    id: String,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!("usage: cargo xtool automation agent-pick-model < models.json");
        return Ok(());
    }
    if !args.is_empty() {
        return Err("usage: cargo xtool automation agent-pick-model < models.json".into());
    }
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes)?;
    let models: Models = serde_json::from_slice(&bytes)?;
    println!("{}", select(&models));
    Ok(())
}

fn select(models: &Models) -> &str {
    for needle in ["minimax", "glm", "qwen", "coder", "hermes"] {
        if let Some(model) = models
            .data
            .iter()
            .find(|model| model.id.to_lowercase().contains(needle))
        {
            return &model.id;
        }
    }
    models
        .data
        .iter()
        .find(|model| !model.id.is_empty())
        .map_or("", |model| &model.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_precedes_input_order() {
        let models: Models =
            serde_json::from_str(r#"{"data":[{"id":"qwen"},{"id":"GLM"},{"id":"MiniMax-2"}]}"#)
                .unwrap();
        assert_eq!(select(&models), "MiniMax-2");
    }

    #[test]
    fn fallback_uses_first_nonempty_id() {
        let models: Models =
            serde_json::from_str(r#"{"data":[{}, {"id":"first"},{"id":"second"}]}"#).unwrap();
        assert_eq!(select(&models), "first");
    }

    #[test]
    fn empty_roster_has_no_selection() {
        let models: Models = serde_json::from_str("{}").unwrap();
        assert_eq!(select(&models), "");
    }
}

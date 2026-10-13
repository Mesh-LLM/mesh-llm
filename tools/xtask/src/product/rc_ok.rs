use crate::repository::check_report::CheckReport;
use serde::{Deserialize, Serialize};
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
#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: [Message; 1],
    max_tokens: u32,
    temperature: u32,
}
#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: &'static str,
}
#[derive(Deserialize)]
struct Response {
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    message: Content,
}
#[derive(Deserialize)]
struct Content {
    content: String,
}

fn evaluate(args: &[String], input: &[u8]) -> Result<String, String> {
    match args {
        [verb] if verb == "model" => {
            let models: Models =
                serde_json::from_slice(input).map_err(|error| error.to_string())?;
            Ok(format!(
                "{}\n",
                models.data.first().map_or("", |model| &model.id)
            ))
        }
        [verb, model] if verb == "request" => serde_json::to_string(&Request {
            model,
            messages: [Message {
                role: "user",
                content: "Reply with exactly: rc-ok",
            }],
            max_tokens: 16,
            temperature: 0,
        })
        .map(|json| format!("{json}\n"))
        .map_err(|error| error.to_string()),
        [verb] if verb == "verify" => {
            let response: Response =
                serde_json::from_slice(input).map_err(|error| error.to_string())?;
            match response.choices.first() {
                Some(choice) if choice.message.content.trim() == "rc-ok" => Ok(String::new()),
                _ => Err("RC chat response must contain exactly rc-ok".to_owned()),
            }
        }
        _ => Err("usage: product rc-ok {model|request MODEL|verify}".to_owned()),
    }
}

pub(super) fn run(args: &[String]) -> CheckReport {
    let mut input = Vec::new();
    if args.first().is_none_or(|verb| verb != "request")
        && let Err(error) = std::io::stdin().read_to_end(&mut input)
    {
        return CheckReport::failure(String::new(), format!("{error}\n"));
    }
    match evaluate(args, &input) {
        Ok(output) => CheckReport::success(output),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rc_verification_requires_exact_first_choice_content() {
        for (content, accepted) in [(" rc-ok\n", true), ("rc-ok extra", false), ("", false)] {
            let response = serde_json::json!({"choices":[{"message":{"content":content}}]});
            assert_eq!(
                super::evaluate(
                    &["verify".to_owned()],
                    &serde_json::to_vec(&response).unwrap()
                )
                .is_ok(),
                accepted
            );
        }
        assert!(super::evaluate(&["verify".to_owned()], br#"{"choices":[]}"#).is_err());
    }

    #[test]
    fn request_preserves_model_and_deterministic_generation_parameters() {
        let model = "model\"with quotes";
        let output = super::evaluate(&["request".to_owned(), model.to_owned()], &[]).unwrap();
        let request: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(request["model"], model);
        assert_eq!(request["max_tokens"], 16);
        assert_eq!(request["temperature"], 0);
    }
}

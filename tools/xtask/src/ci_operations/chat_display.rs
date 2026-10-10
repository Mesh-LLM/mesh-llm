use crate::command::DynResult;
use serde::Deserialize;

#[derive(Deserialize)]
struct Response {
    choices: Vec<Choice>,
    timings: Timings,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct Timings {
    prompt_per_second: f64,
    predicted_per_second: f64,
    predicted_n: u64,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        return crate::repository::check_report::CheckReport::success(
            "usage: ci-ops chat-display < response.json\n".to_owned(),
        )
        .emit();
    }
    if !args.is_empty() {
        return Err("usage: ci-ops chat-display < response.json".into());
    }
    let response: Response = serde_json::from_reader(std::io::stdin().lock())?;
    let choice = response
        .choices
        .first()
        .ok_or("chat response has no choices")?;
    let content: String = choice.message.content.chars().take(200).collect();
    let timings = response.timings;
    crate::repository::check_report::CheckReport::success(format!(
        "{content}\n  prompt: {:.1} tok/s  gen: {:.1} tok/s ({} tok)\n",
        timings.prompt_per_second, timings.predicted_per_second, timings.predicted_n
    ))
    .emit()
}

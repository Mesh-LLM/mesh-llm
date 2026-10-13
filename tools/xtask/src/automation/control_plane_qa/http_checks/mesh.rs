use super::{Context, Fact, decode, request, save, validation};
use serde::Deserialize;
use std::time::{Duration, Instant};

#[derive(Deserialize)]
struct Status {
    #[serde(default)]
    token: String,
    #[serde(default)]
    peers: Vec<serde_json::Value>,
}
#[derive(Deserialize)]
struct Models {
    data: Vec<Model>,
}
#[derive(Deserialize)]
struct Model {
    id: String,
}

pub(super) fn ready(console: u16, context: &Context<'_>) -> Result<Fact, String> {
    wait(context, || {
        let response = request(console, "/api/status", Vec::new(), Duration::from_secs(3))?;
        if response.status >= 400 {
            return Ok(None);
        }
        validation::privacy(&response.body)?;
        #[derive(serde::Serialize, Deserialize)]
        struct Evidence {
            #[serde(default)]
            node_id: String,
            #[serde(default)]
            mesh_id: String,
        }
        let projected: Evidence = decode(&response)?;
        let bytes = serde_json::to_vec(&projected).map_err(|_| "status evidence encoding")?;
        save(context.directory, &format!("status-{console}.json"), &bytes)?;
        Ok(Some(Fact::Done))
    })
}
pub(super) fn invite(console: u16, context: &Context<'_>) -> Result<Fact, String> {
    wait(context, || {
        let response = request(console, "/api/status", Vec::new(), Duration::from_secs(3))?;
        validation::privacy(&response.body)?;
        let status: Status = decode(&response)?;
        Ok((!status.token.is_empty()).then_some(Fact::Token(status.token)))
    })
}
pub(super) fn pair(
    ports: (u16, u16, u16),
    chat: bool,
    context: &Context<'_>,
) -> Result<Fact, String> {
    let mut stable = 0;
    wait(context, || {
        let left = request(ports.0, "/api/status", Vec::new(), Duration::from_secs(3))?;
        let right = request(ports.1, "/api/status", Vec::new(), Duration::from_secs(3))?;
        validation::privacy(&left.body)?;
        validation::privacy(&right.body)?;
        let left: Status = decode(&left)?;
        let right: Status = decode(&right)?;
        stable = if !left.peers.is_empty() && !right.peers.is_empty() {
            stable + 1
        } else {
            0
        };
        Ok((stable >= context.options.stable).then_some(Fact::Done))
    })?;
    if chat {
        chat_probe(ports.2, context)?;
    }
    Ok(Fact::Done)
}
pub(super) fn public(console: u16, api: u16, context: &Context<'_>) -> Result<Fact, String> {
    validation::privacy(
        &request(console, "/api/status", Vec::new(), Duration::from_secs(3))?.body,
    )?;
    match wait(context, || {
        let models: Models = decode(&request(
            api,
            "/v1/models",
            Vec::new(),
            Duration::from_secs(3),
        )?)?;
        Ok(models
            .data
            .first()
            .filter(|model| !model.id.is_empty())
            .map(|_| Fact::Done))
    }) {
        Ok(_) => {
            chat_probe(api, context)?;
            Ok(Fact::Done)
        }
        Err(_) if !context.options.public_required && !context.cancel.is_cancelled() => {
            Ok(Fact::Prerequisite("public-models"))
        }
        Err(error) => Err(error),
    }
}
fn wait(
    context: &Context<'_>,
    mut probe: impl FnMut() -> Result<Option<Fact>, String>,
) -> Result<Fact, String> {
    let until = Instant::now() + context.options.wait;
    loop {
        if context.cancel.is_cancelled() {
            return Err("mixed-version check cancelled".into());
        }
        match probe() {
            Ok(Some(fact)) => return Ok(fact),
            Ok(None) => (),
            Err(error) if error == "mixed-version control data leaked" => return Err(error),
            Err(_) => (),
        }
        if Instant::now() >= until {
            return Err("mixed-version observation deadline".into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}
fn chat_probe(api: u16, context: &Context<'_>) -> Result<(), String> {
    let models: Models = decode(&request(
        api,
        "/v1/models",
        Vec::new(),
        Duration::from_secs(5),
    )?)?;
    let model = models
        .data
        .first()
        .filter(|model| !model.id.is_empty())
        .ok_or("routed model unavailable")?;
    let body = serde_json::to_vec(&serde_json::json!({"model":model.id,"messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"temperature":0})).map_err(|_| "chat encoding")?;
    let response = request(api, "/v1/chat/completions", body, context.options.chat)?;
    #[derive(Deserialize)]
    struct Chat {
        object: String,
        choices: Vec<Choice>,
    }
    #[derive(Deserialize)]
    struct Choice {
        message: Message,
    }
    #[derive(Deserialize)]
    struct Message {
        content: String,
    }
    let chat: Chat = decode(&response)?;
    if response.status >= 400
        || chat.object != "chat.completion"
        || chat
            .choices
            .first()
            .is_none_or(|choice| choice.message.content.is_empty())
    {
        return Err("routed chat failed".into());
    }
    save(
        context.directory,
        &format!("chat-{api}.json"),
        &response.body,
    )
}

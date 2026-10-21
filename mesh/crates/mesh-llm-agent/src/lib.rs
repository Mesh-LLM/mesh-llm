//! A minimal one-shot agent on goose's unrolled loop (`goose-agent`).
//!
//! Steps, in order: max turns, tool execution (`shell`, `load_skill`), then
//! inference through a small OpenAI client (goose's request/response mapping) against a mesh endpoint. The
//! loop, message types and provider are goose's; the two tools, skill
//! discovery and the short system prompt live here, so mesh links only
//! goose's split-out crates and not the `goose` crate.

mod fenced;
mod prompt;
mod provider;
pub mod shell;
pub mod skills;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use goose_agent::events::AgentEvent;
use goose_agent::inference::{InferenceEffect, InferenceRunner};
use goose_agent::machine::{EffectHandler, MachineSession, SessionLoader, StateMachine, Step};
use goose_agent::operation::{
    Emitter, MachineEffect, Operation, OperationResult, assistant_turn_count,
    messages_since_kickoff, not_applicable, yielded_with,
};
use goose_agent::tool::{ToolOperation, ToolProvider};
use goose_provider_types::conversation::Conversation;
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::token_usage::ProviderUsage;
use goose_provider_types::model::ModelConfig;
use rmcp::model::{CallToolRequestParams, CallToolResult, ErrorData, Tool};
use tokio_util::sync::CancellationToken;

pub struct AgentOptions {
    /// Mesh API origin, e.g. `http://127.0.0.1:9337`.
    pub base_url: String,
    pub model: String,
    pub cwd: PathBuf,
    pub max_turns: u32,
}

/// What the run produced, for the caller to print.
pub enum AgentUpdate {
    Text(String),
    ToolCall { name: String, arguments: String },
    ToolResult { error: bool, text: String },
}

pub struct AgentOutcome {
    pub final_text: String,
    pub turns: u32,
}

struct Session {
    conversation: Conversation,
}

impl MachineSession for Session {
    fn id(&self) -> &str {
        "mesh-agent"
    }
    fn conversation(&self) -> Option<&Conversation> {
        Some(&self.conversation)
    }
}

enum Effect {
    Append(Message),
    Usage,
}

impl MachineEffect for Effect {
    fn ensure_message_ids(&mut self) {
        if let Effect::Append(message) = self
            && message.id.is_none()
        {
            *message = message.clone().with_generated_id();
        }
    }
}

impl From<Message> for Effect {
    fn from(message: Message) -> Self {
        Effect::Append(message)
    }
}

impl InferenceEffect for Effect {
    fn record_usage(_usage: ProviderUsage) -> Self {
        Effect::Usage
    }
}

/// The conversation lives in memory for one run; nothing is persisted.
struct Store(Mutex<Conversation>);

#[async_trait]
impl SessionLoader<Session> for Store {
    async fn load(&self, _session_id: &str) -> Result<Session> {
        let conversation = self.0.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Ok(Session { conversation })
    }
}

#[async_trait]
impl EffectHandler<Session, Effect> for Store {
    async fn apply_effects(&self, _: &Session, effects: &mut [Effect], _: &Emitter) -> Result<()> {
        let mut conversation = self.0.lock().unwrap_or_else(|e| e.into_inner());
        for effect in effects {
            if let Effect::Append(message) = effect {
                conversation.push(message.clone());
            }
        }
        Ok(())
    }
}

/// Stops the run once the model has had `max` assistant turns.
struct MaxTurns(u32);

#[async_trait]
impl Operation<Session, Effect> for MaxTurns {
    fn name(&self) -> &'static str {
        "max_turns"
    }
    async fn run(
        &self,
        _: &Session,
        conversation: &Conversation,
        _: &Emitter,
    ) -> Result<OperationResult<Effect>> {
        let turn = messages_since_kickoff(conversation)?;
        let last_is_user_or_tool = turn
            .last()
            .is_some_and(|m| m.role != rmcp::model::Role::Assistant);
        if last_is_user_or_tool && assistant_turn_count(turn) >= self.0 {
            let note = Message::assistant()
                .with_text(format!("Stopped after {} turns (--max-turns).", self.0));
            return yielded_with([Effect::Append(note)]);
        }
        not_applicable()
    }
}

struct MeshTools {
    cwd: PathBuf,
    skills: Vec<skills::Skill>,
}

#[async_trait]
impl ToolProvider<Session> for MeshTools {
    async fn tools(&self, _: &Session) -> Result<Vec<Tool>> {
        let mut tools = vec![shell::tool()];
        if !self.skills.is_empty() {
            tools.push(skills::tool());
        }
        Ok(tools)
    }

    async fn call(
        &self,
        _: &Session,
        _request_id: &str,
        call: CallToolRequestParams,
        _: &Emitter,
    ) -> Result<CallToolResult, ErrorData> {
        let args = call.arguments.unwrap_or_default();
        let text = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or_default();
        match call.name.as_ref() {
            shell::NAME => {
                let timeout = args.get("timeout_secs").and_then(|v| v.as_u64());
                Ok(shell::run(&self.cwd, text("command"), timeout).await)
            }
            skills::NAME => Ok(skills::load(&self.skills, text("name"))),
            other => Err(ErrorData::invalid_params(
                format!("unknown tool {other}"),
                None,
            )),
        }
    }
}

/// Runs a fenced `sh` block (or unparsed raw tool call) from a plain-text
/// reply, for models whose native tool calls don't come through.
struct FencedShell {
    cwd: PathBuf,
}

#[async_trait]
impl Operation<Session, Effect> for FencedShell {
    fn name(&self) -> &'static str {
        "fenced_shell"
    }
    async fn run(
        &self,
        _: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<Effect>> {
        let Some(command) = conversation.last().and_then(fenced::extract) else {
            return not_applicable();
        };
        emit.emit(AgentEvent::Message(
            Message::assistant().with_text(format!("[fenced] $ {command}")),
        ));
        let reply = emit.message(fenced::run(&self.cwd, &command).await);
        goose_agent::operation::applied([Effect::Append(reply)])
    }
}

/// Lean system prompt, contributed as a prompt part so the loop assembles it.
struct SystemPrompt(String);

#[async_trait]
impl Operation<Session, Effect> for SystemPrompt {
    fn name(&self) -> &'static str {
        "system_prompt"
    }
    async fn prompt_parts(&self, _: &Session, _: &Conversation) -> Result<Vec<(String, String)>> {
        Ok(vec![("system".to_string(), self.0.clone())])
    }
}

pub async fn run(
    options: AgentOptions,
    instructions: &str,
    mut on_update: impl FnMut(AgentUpdate),
) -> Result<AgentOutcome> {
    let home = dirs::home_dir();
    let skills = skills::discover(&options.cwd, home.as_deref());
    let system = prompt::system_prompt(&options.cwd, &skills);

    let provider = Arc::new(provider::MeshProvider::new(&options.base_url));
    let model_config = ModelConfig::new(&options.model);

    let tools = ToolOperation::new().with_provider(Arc::new(MeshTools {
        cwd: options.cwd.clone(),
        skills,
    }));
    let steps: Vec<Step<'_, Session, Effect>> = vec![
        Step::Operation(Arc::new(SystemPrompt(system))),
        Step::Operation(Arc::new(MaxTurns(options.max_turns))),
        Step::Operation(Arc::new(tools)),
        Step::Operation(Arc::new(FencedShell {
            cwd: options.cwd.clone(),
        })),
        Step::Inference(Arc::new(InferenceRunner::new(provider, model_config))),
    ];
    let cancel = CancellationToken::new();
    let machine = StateMachine::new(steps, cancel.clone());

    let kickoff = Message::user().with_text(instructions);
    let store = Store(Mutex::new(Conversation::new_unvalidated([kickoff])));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let emit = Emitter::new(tx, cancel);

    let session = {
        let run = machine.run(&store, "mesh-agent", &emit);
        tokio::pin!(run);
        loop {
            tokio::select! {
                result = &mut run => break result?,
                Some(event) = rx.recv() => report(event, &mut on_update),
            }
        }
    };
    while let Ok(event) = rx.try_recv() {
        report(event, &mut on_update);
    }
    finish(&session.conversation)
}

fn report(event: AgentEvent, on_update: &mut impl FnMut(AgentUpdate)) {
    let AgentEvent::Message(message) = event else {
        return;
    };
    use goose_provider_types::conversation::message::MessageContent;
    for content in &message.content {
        match content {
            MessageContent::ToolRequest(request) => {
                if let Ok(call) = &request.tool_call {
                    on_update(AgentUpdate::ToolCall {
                        name: call.name.to_string(),
                        arguments: call
                            .arguments
                            .as_ref()
                            .map(|a| serde_json::Value::Object(a.clone()).to_string())
                            .unwrap_or_default(),
                    });
                }
            }
            MessageContent::ToolResponse(response) => {
                let (error, text) = match &response.tool_result {
                    Ok(result) => (
                        result.is_error == Some(true),
                        result
                            .content
                            .iter()
                            .filter_map(|c| c.as_text().map(|t| t.text.clone()))
                            .collect::<Vec<_>>()
                            .join("\n"),
                    ),
                    Err(error) => (true, error.message.to_string()),
                };
                on_update(AgentUpdate::ToolResult { error, text });
            }
            MessageContent::Text(text) if message.role == rmcp::model::Role::Assistant => {
                on_update(AgentUpdate::Text(text.text.clone()));
            }
            _ => {}
        }
    }
}

fn finish(conversation: &Conversation) -> Result<AgentOutcome> {
    let turn = messages_since_kickoff(conversation)?;
    if let Some(kind) = conversation.last().and_then(Message::error_kind) {
        let text = conversation
            .last()
            .map(Message::as_concat_text)
            .unwrap_or_default();
        anyhow::bail!("model error ({kind:?}): {text}");
    }
    let final_text = turn
        .iter()
        .rev()
        .find(|m| m.role == rmcp::model::Role::Assistant)
        .map(Message::as_concat_text)
        .unwrap_or_default();
    Ok(AgentOutcome {
        final_text,
        turns: assistant_turn_count(turn),
    })
}

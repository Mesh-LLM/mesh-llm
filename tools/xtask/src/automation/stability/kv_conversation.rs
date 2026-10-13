//! complete growing-history KV conversations and overlap follow-ups.
use super::{
    kv_requests::{self, Key, PIN, TOOL},
    kv_tool_calls,
    transport::{Http, Reply},
};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Serialize)]
pub(super) struct Record {
    pub phase: String,
    pub status_code: Option<u16>,
    pub call_id: Option<String>,
    pub error: Option<String>,
}
pub(super) struct Outcome {
    pub records: Vec<Record>,
    pub status_code: Option<u16>,
    pub result: Result<String, String>,
}
struct Conversation<'a> {
    http: &'a Http,
    model: &'a str,
    messages: Vec<Value>,
    records: Vec<Record>,
    status: Option<u16>,
}
impl<'a> Conversation<'a> {
    fn new(http: &'a Http, model: &'a str, messages: Vec<Value>) -> Self {
        Self {
            http,
            model,
            messages,
            records: vec![],
            status: None,
        }
    }
    async fn fetch(&mut self, phase: &str, payload: Value) -> Result<Value, String> {
        let reply = self.http.chat(&payload, false).await;
        match reply {
            Ok(reply) => self.accept(phase, reply),
            Err(error) => {
                self.status = error.status;
                self.records.push(Record {
                    phase: phase.into(),
                    status_code: error.status,
                    call_id: None,
                    error: Some(error.detail.clone()),
                });
                Err(error.detail)
            }
        }
    }
    fn accept(&mut self, phase: &str, reply: Reply) -> Result<Value, String> {
        self.status = Some(reply.status);
        self.records.push(Record {
            phase: phase.into(),
            status_code: Some(reply.status),
            call_id: None,
            error: None,
        });
        reply
            .json
            .ok_or_else(|| "KV conversation requires a nonstreaming JSON response".into())
    }
    fn record_tool(&mut self, response: &Value, expected: Key) -> Result<(), String> {
        let call = kv_tool_calls::extract(response, expected)?;
        kv_tool_calls::append_tool(&mut self.messages, response, &call)?;
        if let Some(record) = self.records.last_mut() {
            record.call_id = Some(call.id);
        }
        Ok(())
    }
    async fn answer(
        &mut self,
        phase: &str,
        prompt: String,
        expected: &[&str],
        tokens: u32,
    ) -> Result<(), String> {
        self.messages.push(json!({"role":"user","content":prompt}));
        let payload = kv_requests::text(self.model, &self.messages, tokens);
        let response = self.fetch(phase, payload).await?;
        self.messages
            .push(kv_tool_calls::text_message(&response, expected)?);
        Ok(())
    }
    async fn second_tool(&mut self, key: Key) -> Result<(), String> {
        self.messages.push(json!({"role":"user","content":format!("Now call {TOOL} with key={}. Do not answer directly before the tool call.",key.name())}));
        let payload = kv_requests::tool(self.model, &self.messages);
        let response = self.fetch("second_tool_call", payload).await?;
        self.record_tool(&response, key)
    }
    async fn finish_history(&mut self, initial: Key, pressure_turns: u32) -> Result<(), String> {
        self.answer(
            "first_tool_result",
            format!("Answer with the tool fact and include {PIN}."),
            &[initial.fact(), PIN],
            128,
        )
        .await?;
        for turn in 1..=pressure_turns {
            self.answer(&format!("pressure_turn_{turn}"),format!("Pressure turn {turn}: keep the pinned value stable. Return {PIN} and a short confirmation."),&[PIN],64).await?;
        }
        let second = match initial {
            Key::Primary => Key::Secondary,
            Key::Secondary => Key::Primary,
        };
        self.second_tool(second).await?;
        self.answer(
            "final_recall",
            "Final recall: include both tool facts and the pinned KV value.".into(),
            &[PIN, Key::Primary.fact(), Key::Secondary.fact()],
            128,
        )
        .await
    }
    fn outcome(mut self, result: Result<String, String>) -> Outcome {
        if let Err(error) = &result {
            self.records.push(Record {
                phase: "failure".into(),
                status_code: self.status,
                call_id: None,
                error: Some(error.clone()),
            });
        }
        Outcome {
            records: self.records,
            status_code: self.status,
            result,
        }
    }
}

pub(super) async fn run(http: &Http, model: &str, attempt: u32, pressure_turns: u32) -> Outcome {
    let mut conversation =
        Conversation::new(http, model, kv_requests::initial(attempt, Key::Primary));
    let result = async {
        let payload = kv_requests::tool(model, &conversation.messages);
        let response = conversation.fetch("first_tool_call", payload).await?;
        conversation.record_tool(&response, Key::Primary)?;
        conversation
            .finish_history(Key::Primary, pressure_turns)
            .await?;
        Ok(format!(
            "completed {pressure_turns} pressure turns and two tool calls with both-fact/PIN recall"
        ))
    }
    .await;
    conversation.outcome(result)
}

pub(super) async fn continue_overlap(
    http: &Http,
    model: &str,
    context: &kv_requests::Overlap,
    reply: Reply,
) -> Outcome {
    let messages = context.payload["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut conversation = Conversation::new(http, model, messages);
    let result = async {
        let response = conversation.accept(&format!("overlap_{}", context.label), reply)?;
        match context.key {
            Some(key) => {
                conversation.record_tool(&response, key)?;
                conversation.finish_history(key, 0).await?;
                Ok(format!(
                    "completed overlap {} with both-fact/PIN recall",
                    context.label
                ))
            }
            None => {
                kv_tool_calls::text_message(&response, &[PIN])?;
                Ok("overlap title included the pinned value".into())
            }
        }
    }
    .await;
    conversation.outcome(result)
}

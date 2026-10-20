use crate::{
    broker::Broker,
    model::{ModelClient, Stage},
    protocol::{Challenge, Intent, ProbeInput, ProbeResponse, ProbeStatus, ToolRecord, validate},
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::json;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use wasmi::{Caller, Config, Engine, Linker, Module, Store};

pub const WASM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/controller.wasm"));

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_tool_calls: u32,
    pub max_model_calls: u32,
    pub max_total_bytes: usize,
    pub max_model_output_bytes: usize,
    pub max_wall: Duration,
    pub fuel: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_tool_calls: 6,
            max_model_calls: 8,
            max_total_bytes: 16 * 1024,
            max_model_output_bytes: 4096,
            max_wall: Duration::from_secs(120),
            fuel: 10_000,
        }
    }
}

struct State {
    input: ProbeInput,
    broker: Box<dyn Broker>,
    model: Arc<dyn ModelClient>,
    trace: Vec<ToolRecord>,
    model_actions: Vec<String>,
    answer: String,
    status: ProbeStatus,
    error: Option<String>,
    model_calls: u32,
    tool_calls: u32,
    total_bytes: usize,
    started: Instant,
    limits: Limits,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Action {
    tool: String,
    #[serde(default)]
    answer: String,
}

impl State {
    fn spend_bytes(&mut self, bytes: usize) -> Result<()> {
        ensure!(
            self.total_bytes.saturating_add(bytes) <= self.limits.max_total_bytes,
            "total byte budget exhausted"
        );
        self.total_bytes += bytes;
        Ok(())
    }

    fn check_time(&self) -> Result<()> {
        ensure!(
            self.started.elapsed() <= self.limits.max_wall,
            "wall-time budget exhausted"
        );
        Ok(())
    }

    fn fail(&mut self, error: impl std::fmt::Display) -> i32 {
        let error = error.to_string();
        self.status = if error.starts_with("invalid_model_output:") {
            ProbeStatus::InvalidModelOutput
        } else {
            ProbeStatus::Failed
        };
        self.error = Some(error);
        0
    }

    fn next_action(&mut self) -> i32 {
        match self.next_action_inner() {
            Ok(code) => code,
            Err(error) => self.fail(error),
        }
    }

    fn next_action_inner(&mut self) -> Result<i32> {
        self.check_time()?;
        // A one-intent round needs the model to choose the observation, not to
        // re-select it merely to produce a summary. The trace is authoritative.
        if self.input.challenge.intents.len() == 1
            && self
                .input
                .challenge
                .intents
                .iter()
                .all(|intent| self.trace.iter().any(|record| record.tool == *intent))
        {
            self.answer = "Required observation attempted; see host-recorded trace.".into();
            return Ok(0);
        }
        let mut invalid_action = None;
        for _ in 0..=1 {
            self.check_time()?;
            ensure!(
                self.model_calls < self.limits.max_model_calls,
                "model-call budget exhausted"
            );
            let remaining: Vec<_> = self
                .input
                .challenge
                .intents
                .iter()
                .filter(|intent| !self.trace.iter().any(|record| record.tool == **intent))
                .map(|intent| intent.as_str())
                .collect();
            let allowed_next_actions = if remaining.is_empty() {
                vec!["finish"]
            } else {
                remaining.clone()
            };
            let prompt = json!({"challenge":self.input.challenge,"question":self.input.question,
                "trace":self.trace,"tool_menu":remaining,"allowed_next_actions":allowed_next_actions,
                "invalid_action":invalid_action})
            .to_string();
            self.spend_bytes(prompt.len())?;
            self.model_calls += 1;
            let output = self.model.complete(Stage::Planner, &prompt)?;
            self.check_time()?;
            ensure!(
                output.len() <= self.limits.max_model_output_bytes,
                "model output byte budget exhausted"
            );
            self.spend_bytes(output.len())?;
            self.model_actions.push(output.clone());
            let action: Action = match serde_json::from_str(&output) {
                Ok(action) => action,
                Err(error) => {
                    invalid_action = Some(format!("malformed JSON: {error}"));
                    continue;
                }
            };
            match self.action_code(action) {
                Ok(code) => return Ok(code),
                Err(error) => invalid_action = Some(error.to_string()),
            }
        }
        anyhow::bail!(
            "invalid_model_output: {}",
            invalid_action.unwrap_or_else(|| "unknown action".into())
        )
    }

    fn action_code(&mut self, action: Action) -> Result<i32> {
        match action.tool.as_str() {
            "finish" => {
                ensure!(
                    self.input
                        .challenge
                        .intents
                        .iter()
                        .all(|intent| self.trace.iter().any(|record| record.tool == *intent)),
                    "finish before required observations"
                );
                ensure!(!action.answer.trim().is_empty(), "empty answer");
                ensure!(
                    action.answer.len() <= self.limits.max_model_output_bytes,
                    "answer too long"
                );
                self.answer = action.answer;
                Ok(0)
            }
            "unavailable" => {
                ensure!(
                    !action.answer.trim().is_empty(),
                    "unavailable needs a reason"
                );
                self.status = ProbeStatus::Unavailable;
                self.answer = action.answer;
                Ok(0)
            }
            "process" => self.selected_code(Intent::Process, 1),
            "executable" => self.selected_code(Intent::Executable, 2),
            "model" => self.selected_code(Intent::Model, 3),
            other => anyhow::bail!("invented or disallowed tool: {other}"),
        }
    }

    fn selected_code(&self, intent: Intent, code: i32) -> Result<i32> {
        ensure!(
            self.input.challenge.intents.contains(&intent),
            "unselected tool"
        );
        ensure!(
            !self.trace.iter().any(|record| record.tool == intent),
            "repeated tool"
        );
        if intent != Intent::Process && self.input.challenge.round == 1 {
            ensure!(
                self.trace
                    .iter()
                    .any(|record| record.tool == Intent::Process),
                "process must be observed first"
            );
        }
        Ok(code)
    }

    fn run_tool(&mut self, code: i32) -> i32 {
        match self.run_tool_inner(code) {
            Ok(()) => 1,
            Err(error) => self.fail(error),
        }
    }

    fn run_tool_inner(&mut self, code: i32) -> Result<()> {
        self.check_time()?;
        ensure!(
            self.tool_calls < self.limits.max_tool_calls,
            "tool-call budget exhausted"
        );
        let tool = match code {
            1 => Intent::Process,
            2 => Intent::Executable,
            3 => Intent::Model,
            _ => anyhow::bail!("invalid WASM tool code"),
        };
        ensure!(
            self.input.challenge.intents.contains(&tool),
            "unselected tool"
        );
        ensure!(
            !self.trace.iter().any(|record| record.tool == tool),
            "repeated tool"
        );
        let record = self.broker.observe(&self.input.challenge, tool)?;
        self.check_time()?;
        ensure!(record.tool == tool, "broker returned wrong tool");
        let bytes = serde_json::to_vec(&record)?.len();
        self.spend_bytes(bytes)?;
        self.tool_calls += 1;
        self.trace.push(record);
        Ok(())
    }
}

pub fn execute(
    input: ProbeInput,
    broker: Box<dyn Broker>,
    model: Arc<dyn ModelClient>,
    limits: Limits,
) -> Result<ProbeResponse> {
    validate(&input.challenge)?;
    ensure!(
        input.question.len() <= 4096,
        "question byte budget exhausted"
    );
    ensure!(
        WASM.starts_with(b"\0asm"),
        "controller is not a WASM artifact"
    );
    let start = Instant::now();
    let mut config = Config::default();
    config.consume_fuel(true);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, WASM)?;
    let mut linker: Linker<State> = Linker::new(&engine);
    linker.func_wrap(
        "probe",
        "next_action",
        |mut caller: Caller<'_, State>| -> i32 { caller.data_mut().next_action() },
    )?;
    linker.func_wrap(
        "probe",
        "run_tool",
        |mut caller: Caller<'_, State>, code: i32| -> i32 { caller.data_mut().run_tool(code) },
    )?;
    let state = State {
        input,
        broker,
        model,
        trace: vec![],
        model_actions: vec![],
        answer: String::new(),
        status: ProbeStatus::Completed,
        error: None,
        model_calls: 0,
        tool_calls: 0,
        total_bytes: 0,
        started: start,
        limits,
    };
    let mut store = Store::new(&engine, state);
    store.set_fuel(limits.fuel)?;
    let instance = linker.instantiate_and_start(&mut store, &module)?;
    let run = instance.get_typed_func::<(), i32>(&store, "run")?;
    let wasm_error = run
        .call(&mut store, ())
        .err()
        .map(|error| format!("WASM trap: {error}"));
    let state = store.into_data();
    let challenge: Challenge = state.input.challenge;
    Ok(ProbeResponse {
        challenge_id: challenge.id,
        peer: challenge.peer,
        nonce: challenge.nonce,
        answer: state.answer,
        status: if wasm_error.is_some() {
            ProbeStatus::Failed
        } else {
            state.status
        },
        trace: state.trace,
        model_actions: state.model_actions,
        error: state.error.or(wasm_error),
        wasm_executed: true,
        model_calls: state.model_calls,
        tool_calls: state.tool_calls,
        elapsed_ms: start.elapsed().as_millis(),
    })
}

use super::{
    agents,
    options::{Mode, Options},
};
use serde_json::{Value, json};

pub(super) fn build(options: &Options) -> Value {
    let phases = if options.streaming {
        vec![
            "tool_call",
            "tool_result",
            "stream_tool_call",
            "stream_tool_result",
        ]
    } else {
        vec!["tool_call", "tool_result"]
    };
    let mut checks = Vec::new();
    for model in &options.models {
        for attempt in 1..=options.attempts {
            checks.push(json!({"model":model,"attempt":attempt,"phases":phases}));
        }
    }
    let tool = json!({"name":"agent-tool-call-reliability","endpoint":options.base.as_str(),
        "checks":checks,"evidence":["results.jsonl"]});
    if options.mode == Mode::ToolCall {
        return tool;
    }
    let mut steps = vec![
        json!({"name":"openai-surface-probe","models":options.models,
        "attempts":options.attempts,"phases":if options.streaming { vec!["models","chat","stream_chat"] }
            else { vec!["models","chat"] },"output":options.output.join("results.jsonl")}),
        json!({"name":"tool-call-reliability","owner":"native","plan":tool,
            "output":options.output.join("agent-tool-call-reliability/results.jsonl")}),
    ];
    for agent in &options.agents {
        steps.push(json!({"name":format!("{}-agent-smoke",agent.name()),
            "adapter":format!("scripts/ci-{}-smoke.sh",agent.name()),
            "env":agents::environment(*agent,options),"prerequisite":agent.name(),
            "log":format!("logs/{}-agent-smoke.log",agent.name())}));
    }
    if let Some(binary) = &options.binary {
        steps.push(json!({"name":"release-attestation-inspect","binary":binary,
            "expected_status":options.expected_attestation,"output":options.output.join("release-attestation.json")}));
    }
    json!({"name":"nightly-stability","endpoint":options.base.as_str(),"models":options.models,
        "attempts":options.attempts,"output_dir":options.output,
        "evidence":["manifest.json","commands.jsonl","results.jsonl","release-attestation.json",
            "summary.json","summary.md","logs/"],"steps":steps})
}

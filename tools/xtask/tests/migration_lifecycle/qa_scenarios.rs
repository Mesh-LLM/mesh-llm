use super::protocol::{Behavior, Destination, Plan, Record};

pub(super) fn plan(name: &str) -> super::Result<Plan> {
    let (behavior, bytes): (_, &[u8]) = match name {
        "success" => (
            Behavior::Clean,
            b"{\"role\":\"client\",\"tokens\":0,\"status\":\"ready\",\"event\":\"passive_mode\"}\n",
        ),
        "string" => (Behavior::Clean, b"{\"message\":\"prefix cLiEnT ReAdY suffix\"}\n"),
        "structured-object" => (
            Behavior::Clean,
            b"{\"role\":\"client\",\"status\":\"ready\",\"event\":\"passive_mode\",\"message\":{\"detail\":\"starting\"}}\n",
        ),
        "nonstring-only" => (
            Behavior::Clean,
            concat!(
                "{\"message\":[\"Client ready\"]}\n",
                "{\"message\":{\"Client ready\":false}}\n",
                "{\"message\":{\"detail\":\"Client ready\"}}\n",
                "{\"message\":42}\n{\"message\":false}\n{\"message\":null}\n",
            ).as_bytes(),
        ),
        "escaped-containers" => (
            Behavior::Clean,
            b"{\"message\":[\"\\fLIENT ready\"]}\n{\"message\":{\"\\u001cLIENT ready\":false}}\n",
        ),
        "unicode-container" => (Behavior::Clean, b"{\"message\":[\"\\u0c5cLIENT ready\"]}\n"),
        "misleading" => (
            Behavior::Clean,
            b"Client ready\n{\"detail\":\"Client ready\"}\n{\"event\":\"passive_mode\",\"status\":\"starting\",\"role\":\"client\"}\n",
        ),
        "timeout" => (Behavior::Clean, b""),
        "malformed" => (Behavior::Clean, b"{\"message\":\"Client ready a\"b\"}\n"),
        "eof" => (Behavior::UnterminatedEof, b"{\"message\":\"Client ready\"}"),
        "early-exit" => (Behavior::EarlyNonzero, b""),
        "stubborn" => (Behavior::Stubborn, b"{\"message\":\"Client ready\"}\n"),
        "nonzero" => (Behavior::Nonzero, b"{\"message\":\"Client ready\"}\n"),
        "deletion" => (
            Behavior::StateDeletionFailure,
            b"{\"message\":\"Client ready\"}\n",
        ),
        "descendant" => (Behavior::DescendantAfterExit, b""),
        "late-ready" => (Behavior::LateReady, b""),
        _ => return Err("unknown QA scenario".into()),
    };
    Ok(Plan {
        behavior,
        records: vec![Record {
            stream: Destination::Stdout,
            bytes: bytes.to_vec(),
        }],
    })
}

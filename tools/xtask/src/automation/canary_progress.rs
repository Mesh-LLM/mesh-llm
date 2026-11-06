//! Compact projection of Goose events; the caller retains the original stream.
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

#[path = "canary_progress/commands.rs"]
mod commands;
#[path = "canary_progress/events.rs"]
mod events;
#[cfg(test)]
#[path = "canary_progress/tests.rs"]
mod tests;

const MAX_EVENT: usize = 1024 * 1024;
const HEARTBEAT: Duration = Duration::from_secs(60);

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation canary-progress --commands DIRECTORY",
        values: &["--commands"],
        flags: &["--help"],
    };
    let parsed = GRAMMAR.parse(args).map_err(|_| GRAMMAR.usage)?;
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return Err("unexpected canary-progress arguments".into());
    }
    let directory = Path::new(parsed.last("--commands").ok_or("missing --commands")?);
    let mut output = crate::cli_output::stdout();
    let input = events::NonblockingInput::stdin()?;
    project(input, directory, &mut output)
}

fn project(input: impl Read + Send, directory: &Path, output: &mut impl Write) -> DynResult<()> {
    let mut progress =
        Progress::new(std::env::var("GITHUB_ACTIONS").is_ok_and(|value| value == "true"));
    let mut commands = commands::Commands::default();
    writeln!(
        output,
        "canary: concise Goose progress; full events: agent.log; command output: {}/<label>/output.log",
        directory.display()
    )?;
    let cancelled = AtomicBool::new(false);
    std::thread::scope(|scope| -> DynResult<()> {
        let (send, receive) = mpsc::sync_channel(4);
        let reader = scope.spawn(|| events::read_events(input, send, &cancelled));
        let result = (|| -> DynResult<()> {
            commands.report(directory, output)?;
            let mut heartbeat = Instant::now();
            loop {
                match receive.recv_timeout(Duration::from_secs(1)) {
                    Ok(line) => progress.line(&line, output)?,
                    Err(mpsc::RecvTimeoutError::Timeout) => (),
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if heartbeat.elapsed() >= HEARTBEAT {
                    progress.heartbeat(output)?;
                    commands.report(directory, output)?;
                    heartbeat = Instant::now();
                }
                output.flush()?;
            }
            commands.report(directory, output)?;
            Ok(())
        })();
        cancelled.store(true, Ordering::Relaxed);
        // Drop the receiver before joining: a queued reader must not block on send.
        drop(receive);
        let read_result = reader.join().map_err(|_| "Goose event reader panicked")?;
        result?;
        read_result?;
        output.flush()?;
        Ok(())
    })
}

struct Progress {
    github_groups: bool,
    started: Instant,
    last_event: Instant,
    tools: BTreeMap<String, (String, Instant)>,
}
impl Progress {
    fn new(github_groups: bool) -> Self {
        Self {
            github_groups,
            started: Instant::now(),
            last_event: Instant::now(),
            tools: BTreeMap::new(),
        }
    }
    fn line(&mut self, bytes: &[u8], output: &mut impl Write) -> io::Result<()> {
        self.last_event = Instant::now();
        let Ok(event) = serde_json::from_slice::<Value>(bytes) else {
            return writeln!(
                output,
                "goose diagnostic: {}",
                compact(&String::from_utf8_lossy(bytes), 400)
            );
        };
        match event["type"].as_str() {
            Some("message") => self.message(&event["message"], output),
            Some("error") => writeln!(
                output,
                "goose error: {}",
                compact(event["error"].as_str().unwrap_or("unspecified error"), 600)
            ),
            Some("complete") => writeln!(
                output,
                "goose: turn complete — {}s elapsed; harness determines the next gate",
                self.started.elapsed().as_secs()
            ),
            Some("notification") => Ok(()),
            _ => writeln!(output, "goose: unknown event retained in agent.log"),
        }
    }
    fn message(&mut self, message: &Value, output: &mut impl Write) -> io::Result<()> {
        if message["metadata"]["user_visible"] == false {
            return Ok(());
        }
        let Some(content) = message["content"].as_array() else {
            return Ok(());
        };
        for part in content {
            match part["type"].as_str() {
                Some("text") if message["role"] == "assistant" => {
                    writeln!(
                        output,
                        "goose: {}",
                        compact(part["text"].as_str().unwrap_or_default(), 600)
                    )?;
                }
                Some("toolRequest") => self.request(part, output)?,
                Some("toolResponse") => self.response(part, output)?,
                Some("error") => {
                    writeln!(output, "goose error: {}", compact(&part.to_string(), 600))?
                }
                _ => (),
            }
        }
        Ok(())
    }
    fn request(&mut self, part: &Value, output: &mut impl Write) -> io::Result<()> {
        if part["toolCall"]["status"] == "error" {
            return writeln!(
                output,
                "goose tool parse error: {}",
                compact(&part["toolCall"].to_string(), 600)
            );
        }
        let id = part["id"].as_str().unwrap_or("unknown");
        let call = &part["toolCall"]["value"];
        let name = call["name"].as_str().unwrap_or("tool");
        let args = &call["arguments"];
        let detail = args["command"]
            .as_str()
            .or_else(|| args["path"].as_str())
            .unwrap_or("");
        let label = compact(&format!("{name}: {detail}"), 240);
        if self.tools.len() < 64 {
            self.tools
                .insert(id.to_owned(), (label.clone(), Instant::now()));
        }
        writeln!(output, "goose running: {label}")?;
        self.details(
            &format!("Goose command: {}", compact(name, 80)),
            detail,
            output,
        )
    }
    fn response(&mut self, part: &Value, output: &mut impl Write) -> io::Result<()> {
        let id = part["id"].as_str().unwrap_or("unknown");
        let (name, started) = self
            .tools
            .remove(id)
            .unwrap_or_else(|| ("tool".into(), Instant::now()));
        let result = &part["toolResult"];
        let failed = result["status"] == "error" || result["value"]["isError"] == true;
        writeln!(
            output,
            "goose {}: {name} — {}s",
            if failed { "tool failed" } else { "returned" },
            started.elapsed().as_secs()
        )?;
        if failed {
            writeln!(
                output,
                "goose failure: {}",
                compact(&result.to_string(), 1200)
            )?;
        }
        self.details(
            &format!("Goose tool result: {}", compact(&name, 80)),
            &tool_details(result),
            output,
        )
    }
    fn details(&self, title: &str, text: &str, output: &mut impl Write) -> io::Result<()> {
        if !self.github_groups {
            return Ok(());
        }
        writeln!(output, "::group::{title} (full original in agent.log)")?;
        for line in text.lines() {
            // Wrap long JSON lines before the outer capture's line-size bound.
            let characters: Vec<_> = line.chars().filter(|c| !c.is_control()).collect();
            for chunk in characters.chunks(1000) {
                let text: String = chunk.iter().collect();
                writeln!(output, "goose detail: {}", escape_commands(&text))?;
            }
        }
        writeln!(output, "::endgroup::")
    }
    fn heartbeat(&self, output: &mut impl Write) -> io::Result<()> {
        writeln!(
            output,
            "goose activity: {}s elapsed; last event {}s ago",
            self.started.elapsed().as_secs(),
            self.last_event.elapsed().as_secs()
        )?;
        for (name, started) in self.tools.values() {
            writeln!(
                output,
                "goose still running: {name} — {}s elapsed",
                started.elapsed().as_secs()
            )?;
        }
        Ok(())
    }
}

fn compact(text: &str, limit: usize) -> String {
    let flattened = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut result: String = flattened
        .chars()
        .filter(|c| !c.is_control())
        .take(limit)
        .collect();
    if flattened.chars().count() > limit {
        result.push('…');
    }
    escape_commands(&result)
}

fn escape_commands(text: &str) -> String {
    // GitHub recognizes commands even after a prefix. Never let model or tool
    // text close a group, set an output, or emit a workflow command.
    text.replace("::", ": :")
        .replace("::", ": :")
        .replace("##[", "# #[")
}

fn tool_details(result: &Value) -> String {
    let text = result["value"]["content"]
        .as_array()
        .map(|content| {
            content
                .iter()
                .filter_map(|part| part["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    if text.is_empty() {
        result["error"]
            .as_str()
            .map_or_else(|| result.to_string(), str::to_owned)
    } else {
        text
    }
}

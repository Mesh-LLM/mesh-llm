//! A plugin that follows what its node executes and shows a summary in the
//! node's `GET /api/status`: the two halves of #2581 together, with no
//! measurement-specific code in the host.
//!
//! It subscribes to the local inference observation feed
//! (`PluginContext::subscribe_inference_observation`, host capability
//! `local_inference_observation.v1`) and every 5 seconds publishes, as its
//! status contribution (`PluginContext::publish_status`, `plugin_status.v1`),
//! how many executions are running and what finished in the last minute.
//!
//! Build it with `cargo build -p mesh-llm-plugin --example inference_activity`,
//! add it to a node's config:
//!
//! ```toml
//! [[plugin]]
//! name = "inference-activity"
//! command = "/path/to/target/debug/examples/inference_activity"
//! allow_inference_observation = true
//! ```
//!
//! and read `/api/status.plugin_status.inference-activity`.

use mesh_llm_plugin::inference_observation::{
    InferenceObservationFeed, InferenceObservationItem, inference_observation,
};
use mesh_llm_plugin::plugin_status::plugin_status;
use mesh_llm_plugin::{
    PluginContext, PluginMetadata, PluginRuntime, SimplePlugin, plugin_manifest,
    plugin_server_info, proto,
};
use proto::inference_observation_event::Kind;
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

const NAME: &str = "inference-activity";
const WINDOW: Duration = Duration::from_secs(60);
const PUBLISH_EVERY: Duration = Duration::from_secs(5);

/// The snapshot this plugin publishes, version 1 of its schema.
#[derive(Serialize)]
struct Summary {
    running: usize,
    window_seconds: u64,
    executions_finished: usize,
    output_tokens: u64,
    output_tokens_per_second: f64,
    /// Events the host dropped because this plugin read too slowly; the
    /// figures around a loss undercount.
    lost_events: u64,
    /// Host time the figures are complete up to, Unix nanoseconds.
    as_of_unix_ns: u64,
}

#[derive(Default)]
struct Activity {
    /// Running executions, by id: output tokens reported so far.
    running: BTreeMap<u64, u64>,
    /// Finished in the window: (end time, output tokens).
    finished: VecDeque<(u64, u64)>,
    lost_events: u64,
    watermark: u64,
}

impl Activity {
    fn restart_from(&mut self, snapshot: &proto::InferenceObservationSnapshot) {
        self.running = snapshot
            .active
            .iter()
            .map(|execution| (execution.execution_id, 0))
            .collect();
        self.watermark = self.watermark.max(snapshot.taken_at_unix_ns);
    }

    fn apply(&mut self, batch: proto::InferenceObservationBatch) {
        if let Some(loss) = &batch.loss {
            self.lost_events += loss.lost_events;
        }
        if let Some(resync) = &batch.resync {
            self.restart_from(resync);
        }
        for event in batch.events {
            match event.kind {
                Some(Kind::Started(_)) => {
                    self.running.insert(event.execution_id, 0);
                }
                Some(Kind::Usage(usage)) => {
                    if let Some(tokens) = self.running.get_mut(&event.execution_id) {
                        *tokens = u64::from(usage.completion_tokens);
                    }
                }
                Some(Kind::Ended(_)) => {
                    if let Some(tokens) = self.running.remove(&event.execution_id) {
                        self.finished.push_back((event.at_unix_ns, tokens));
                    }
                }
                _ => {}
            }
        }
        self.watermark = self.watermark.max(batch.watermark_unix_ns);
    }

    fn summary(&mut self) -> Summary {
        let window_ns = u64::try_from(WINDOW.as_nanos()).unwrap_or(u64::MAX);
        let start = self.watermark.saturating_sub(window_ns);
        while self.finished.front().is_some_and(|(at, _)| *at < start) {
            self.finished.pop_front();
        }
        let output_tokens = self.finished.iter().map(|(_, tokens)| tokens).sum::<u64>();
        Summary {
            running: self.running.len(),
            window_seconds: WINDOW.as_secs(),
            executions_finished: self.finished.len(),
            output_tokens,
            output_tokens_per_second: output_tokens as f64 / WINDOW.as_secs_f64(),
            lost_events: self.lost_events,
            as_of_unix_ns: self.watermark,
        }
    }
}

async fn subscribe(context: &mut PluginContext<'static>) -> Option<InferenceObservationFeed> {
    match context.subscribe_inference_observation().await {
        Ok(feed) => Some(feed),
        Err(error) => {
            eprintln!("{NAME}: no inference observation: {error}");
            None
        }
    }
}

async fn follow(mut context: PluginContext<'static>) {
    let Some(mut feed) = subscribe(&mut context).await else {
        return;
    };
    let mut activity = Activity::default();
    activity.restart_from(feed.snapshot());
    let mut publish = tokio::time::interval(PUBLISH_EVERY);
    loop {
        tokio::select! {
            item = feed.next() => match item {
                Some(InferenceObservationItem::Batch(batch)) => activity.apply(batch),
                Some(InferenceObservationItem::Overflowed) => {
                    // Batches were dropped here: start again from a snapshot.
                    let Some(next) = subscribe(&mut context).await else { return };
                    feed = next;
                    activity.restart_from(feed.snapshot());
                }
                None => return,
            },
            _ = publish.tick() => {
                let summary = activity.summary();
                if let Err(error) = context.publish_status(&summary, PUBLISH_EVERY * 3).await {
                    eprintln!("{NAME}: status not published: {error}");
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let metadata = PluginMetadata::new(
        NAME,
        env!("CARGO_PKG_VERSION"),
        plugin_server_info(
            NAME,
            env!("CARGO_PKG_VERSION"),
            "Inference activity",
            "Shows what this node executed in the last minute in /api/status",
            None::<String>,
        ),
    )
    .with_manifest(
        plugin_manifest()
            .item(inference_observation())
            .item(plugin_status(1))
            .build(),
    );
    let plugin = SimplePlugin::new(metadata).on_initialized(|context| {
        let context = context.owned();
        Box::pin(async move {
            tokio::spawn(follow(context));
            Ok(())
        })
    });
    PluginRuntime::run(plugin).await
}

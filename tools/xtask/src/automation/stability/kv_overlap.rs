use super::{
    kv_requests::Overlap,
    transport::{Failure, Http, Reply},
};
use std::sync::Arc;
use tokio::{sync::Barrier, task::JoinSet};

pub(super) struct Started {
    pub context: Overlap,
    pub reply: Result<Reply, Failure>,
}

// Every task reaches the start barrier, even when cancellation has arrived.
// This prevents a cancelled participant from stranding its sibling requests.
// Individual request failures remain rows; they do not erase successful siblings.
pub(super) async fn dispatch(
    http: Arc<Http>,
    contexts: Vec<Overlap>,
) -> Result<Vec<Started>, String> {
    if !(2..=64).contains(&contexts.len()) {
        return Err("KV overlap requires 2..64 concurrent requests".into());
    }
    let barrier = Arc::new(Barrier::new(contexts.len()));
    let mut tasks = JoinSet::new();
    for (index, context) in contexts.into_iter().enumerate() {
        let barrier = barrier.clone();
        let http = http.clone();
        tasks.spawn(async move {
            barrier.wait().await;
            let reply = http.chat(&context.payload, false).await;
            (index, Started { context, reply })
        });
    }
    let mut rows = Vec::new();
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(row) => rows.push(row),
            Err(_) => {
                tasks.abort_all();
                while tasks.join_next().await.is_some() {}
                return Err("KV overlap task did not complete".into());
            }
        }
    }
    rows.sort_by_key(|(index, _)| *index);
    Ok(rows.into_iter().map(|(_, row)| row).collect())
}

#[cfg(test)]
#[path = "kv_overlap_tests.rs"]
mod tests;

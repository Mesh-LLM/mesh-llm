//! Recorder and durable-spill channel ownership; close both before joining workers.
use super::*;

#[derive(Debug)]
pub(crate) struct ExactStateRecordWorker {
    sender: Mutex<Option<SyncSender<PendingExactStateRecord>>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    pub(super) durable_sender: Mutex<Option<config::durable_spill::DurableSpillSender>>,
}

impl ExactStateRecordWorker {
    pub(super) fn new(
        sender: SyncSender<PendingExactStateRecord>,
        record_task: JoinHandle<()>,
        durable_task: Option<JoinHandle<()>>,
        durable_sender: Option<config::durable_spill::DurableSpillSender>,
    ) -> Self {
        let mut tasks = vec![record_task];
        tasks.extend(durable_task);
        Self {
            sender: Mutex::new(Some(sender)),
            tasks: Mutex::new(tasks),
            durable_sender: Mutex::new(durable_sender),
        }
    }

    pub(super) fn with_sender<T>(
        &self,
        use_sender: impl FnOnce(Option<&SyncSender<PendingExactStateRecord>>) -> T,
    ) -> T {
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        use_sender(sender.as_ref())
    }
}

impl Drop for ExactStateRecordWorker {
    fn drop(&mut self) {
        self.sender
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        self.durable_sender
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        for task in self
            .tasks
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
        {
            let _ = task.join();
        }
    }
}

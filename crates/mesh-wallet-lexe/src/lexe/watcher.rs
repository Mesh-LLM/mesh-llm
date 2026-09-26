//! Tracks the wallet's payments by hash through the SDK's update feed.
//!
//! The SDK caches payments by created index, so the watcher records each
//! payment's index to serve lookups by hash. Waiters subscribe by hash, and
//! while anyone waits, a single task polls the feed and publishes each update.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lexe::types::command::{
    GetPaymentRequest, GetPaymentResponse, GetUpdatedPaymentsRequest, GetUpdatedPaymentsResponse,
};
use lexe::types::payment::{PaymentCreatedIndex, PaymentHash, PaymentUpdatedIndex};
use lexe::wallet::LexeWallet;
use mesh_llm_wallet::provider::Transaction;
use tokio::sync::{Notify, watch};
use tokio::time::Instant;

use super::LexeProvider;

/// Cadence while a watched payment is likely to change soon.
const FAST_POLL: Duration = Duration::from_millis(250);
/// How long the fast cadence lasts after a new waiter or payment update.
const FAST_WINDOW: Duration = Duration::from_secs(10);
/// Slowest cadence, reached by doubling once the fast window lapses.
const MAX_POLL: Duration = Duration::from_secs(4);

/// The wallet API methods the watcher uses. Lets tests mock out the wallet.
pub(super) trait PaymentsApi: Send + Sync + 'static {
    fn get_updated_payments(
        &self,
        req: GetUpdatedPaymentsRequest,
    ) -> impl Future<Output = anyhow::Result<GetUpdatedPaymentsResponse>> + Send;

    fn get_payment(
        &self,
        req: GetPaymentRequest,
    ) -> impl Future<Output = anyhow::Result<GetPaymentResponse>> + Send;
}

/// Both methods first ask Lexe's gateway for the latest payment update and
/// only fetch from the user node when the SDK's local cache is behind, then
/// serve results from the cache. So calling them is cheap and stays within
/// Lexe's rate limits, unlike querying the user node directly.
impl PaymentsApi for LexeWallet {
    fn get_updated_payments(
        &self,
        req: GetUpdatedPaymentsRequest,
    ) -> impl Future<Output = anyhow::Result<GetUpdatedPaymentsResponse>> + Send {
        LexeWallet::get_updated_payments(self, req)
    }

    fn get_payment(
        &self,
        req: GetPaymentRequest,
    ) -> impl Future<Output = anyhow::Result<GetPaymentResponse>> + Send {
        LexeWallet::get_payment(self, req)
    }
}

pub(super) struct PaymentWatcher<W = LexeWallet> {
    wallet: Arc<W>,
    /// Exclusive update-feed position. Held across each feed read, so reads
    /// apply in order.
    cursor: tokio::sync::Mutex<Option<PaymentUpdatedIndex>>,
    state: Mutex<State>,
    /// Triggers an immediate poll and restarts the fast cadence.
    kick: Notify,
}

#[derive(Default)]
struct State {
    /// The created index of every payment read from the feed.
    indexes: HashMap<PaymentHash, PaymentCreatedIndex>,
    /// The waiters on each watched payment.
    /// A poller is running iff this is non-empty.
    watched: HashMap<PaymentHash, watch::Sender<Option<Transaction>>>,
}

impl PaymentWatcher {
    pub(super) fn new(wallet: Arc<LexeWallet>) -> Self {
        Self {
            wallet,
            cursor: Default::default(),
            state: Default::default(),
            kick: Notify::new(),
        }
    }
}

impl<W: PaymentsApi> PaymentWatcher<W> {
    /// This payment's current state, if the wallet has it.
    pub(super) async fn lookup(&self, hash: PaymentHash) -> anyhow::Result<Option<Transaction>> {
        let mut index = self.index_of(hash);
        if index.is_none() {
            self.refresh().await?;
            index = self.index_of(hash);
        }
        let Some(index) = index else {
            return Ok(None);
        };
        let response = self.wallet.get_payment(GetPaymentRequest { index }).await?;
        Ok(response.payment.map(LexeProvider::transaction))
    }

    fn index_of(&self, hash: PaymentHash) -> Option<PaymentCreatedIndex> {
        self.state.lock().unwrap().indexes.get(&hash).copied()
    }

    /// Watch this payment's latest state. The receiver only sees states the
    /// watcher publishes after this call, so look up the current state after
    /// subscribing, not before, or an update in between is missed.
    pub(super) fn subscribe(
        self: &Arc<Self>,
        hash: PaymentHash,
    ) -> watch::Receiver<Option<Transaction>> {
        let mut locked_state = self.state.lock().unwrap();
        if locked_state.watched.is_empty() {
            tokio::spawn(Arc::clone(self).poll_while_watched());
        }
        let updates = match locked_state.watched.entry(hash) {
            Entry::Occupied(entry) => entry.get().subscribe(),
            Entry::Vacant(entry) => entry.insert(watch::Sender::new(None)).subscribe(),
        };
        drop(locked_state);
        self.kick.notify_one();
        updates
    }

    /// Read new updates from the feed, recording each payment's index and
    /// sending it to its waiters. Returns whether there were any.
    async fn refresh(&self) -> anyhow::Result<bool> {
        let mut cursor = self.cursor.lock().await;
        let request = GetUpdatedPaymentsRequest {
            start_index: *cursor,
            limit: None,
        };
        let response = self.wallet.get_updated_payments(request).await?;
        if response.payments.is_empty() {
            return Ok(false);
        }
        *cursor = response.updated_index;

        let mut locked_state = self.state.lock().unwrap();
        let state = &mut *locked_state;
        for payment in response.payments {
            let Some(hash) = payment.hash else { continue };
            state.indexes.insert(hash, payment.index);
            if let Some(updates) = state.watched.get(&hash) {
                updates.send_replace(Some(LexeProvider::transaction(payment)));
            }
        }
        Ok(true)
    }

    async fn poll_while_watched(self: Arc<Self>) {
        let mut delay = FAST_POLL;
        let mut fast_until = Instant::now();
        loop {
            if tokio::time::timeout(delay, self.kick.notified())
                .await
                .is_ok()
            {
                fast_until = Instant::now() + FAST_WINDOW;
            }
            if !self.is_watched() {
                return;
            }

            match self.refresh().await {
                Ok(true) => fast_until = Instant::now() + FAST_WINDOW,
                Ok(false) => {}
                Err(error) => tracing::warn!("Lexe payment update poll failed: {error:#}"),
            }

            delay = if Instant::now() < fast_until {
                FAST_POLL
            } else {
                (delay * 2).min(MAX_POLL)
            };
        }
    }

    /// Drop payments nobody waits on. Returns whether any are left, and if
    /// not, the poller must exit.
    fn is_watched(&self) -> bool {
        let mut locked_state = self.state.lock().unwrap();
        locked_state
            .watched
            .retain(|_, updates| updates.receiver_count() > 0);
        !locked_state.watched.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ops::Bound;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use lexe::types::bitcoin::Amount;
    use lexe::types::payment::{
        Payment, PaymentCreatedIndex, PaymentDirection, PaymentId, PaymentKind, PaymentRail,
        PaymentStatus,
    };
    use lexe::types::util::TimestampMs;
    use mesh_llm_wallet::provider::PaymentStatus as TransactionStatus;
    use tokio::time::sleep;

    use super::*;

    /// A wallet whose payment updates are set by the test.
    #[derive(Default)]
    struct MockWallet {
        /// The latest state of each payment, keyed by its update index.
        payments: Mutex<BTreeMap<PaymentUpdatedIndex, Payment>>,
        /// When each poll happened, and its `start_index`.
        polls: Mutex<Vec<(Instant, Option<PaymentUpdatedIndex>)>>,
        /// The number of upcoming polls that fail.
        failures: AtomicUsize,
    }

    impl MockWallet {
        /// Replace a payment's state, as the node does when it persists one.
        fn update(&self, payment: Payment) {
            let mut payments = self.payments.lock().unwrap();
            payments.retain(|_, existing| existing.index.id != payment.index.id);
            payments.insert(payment.updated_index(), payment);
        }

        fn poll_times(&self) -> Vec<Instant> {
            let polls = self.polls.lock().unwrap();
            polls.iter().map(|(time, _)| *time).collect()
        }
    }

    impl PaymentsApi for MockWallet {
        fn get_updated_payments(
            &self,
            req: GetUpdatedPaymentsRequest,
        ) -> impl Future<Output = anyhow::Result<GetUpdatedPaymentsResponse>> + Send {
            self.polls
                .lock()
                .unwrap()
                .push((Instant::now(), req.start_index));
            let fail = self
                .failures
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
                .is_ok();
            let result = if fail {
                Err(anyhow::anyhow!("mock poll failure"))
            } else {
                // Like the SDK, return every payment updated after `start_index`.
                let start = req.start_index.map_or(Bound::Unbounded, Bound::Excluded);
                let payments = self
                    .payments
                    .lock()
                    .unwrap()
                    .range((start, Bound::Unbounded))
                    .map(|(_, payment)| payment.clone())
                    .collect::<Vec<_>>();
                let updated_index = payments.last().map(Payment::updated_index);
                Ok(GetUpdatedPaymentsResponse {
                    payments,
                    updated_index,
                })
            };
            std::future::ready(result)
        }

        fn get_payment(
            &self,
            req: GetPaymentRequest,
        ) -> impl Future<Output = anyhow::Result<GetPaymentResponse>> + Send {
            let payment = self
                .payments
                .lock()
                .unwrap()
                .values()
                .find(|payment| payment.index == req.index)
                .cloned();
            std::future::ready(Ok(GetPaymentResponse { payment }))
        }
    }

    /// Lexe's status and display message for each stage of a payment.
    type Stage = (PaymentStatus, &'static str);
    const PENDING: Stage = (PaymentStatus::Pending, "invoice generated");
    const CLAIMING: Stage = (PaymentStatus::Pending, "claiming");
    const COMPLETED: Stage = (PaymentStatus::Completed, "completed");

    fn hash(byte: u8) -> PaymentHash {
        format!("{byte:02x}").repeat(32).parse().unwrap()
    }

    /// An inbound invoice payment at `stage`, last updated at `updated_at_ms`.
    fn payment(hash: PaymentHash, updated_at_ms: u64, (status, status_msg): Stage) -> Payment {
        let created_at = TimestampMs::from_millis(1).unwrap();
        Payment {
            index: PaymentCreatedIndex {
                created_at,
                id: PaymentId::Lightning(hash),
            },
            rail: PaymentRail::Invoice,
            kind: PaymentKind::Invoice,
            direction: PaymentDirection::Inbound,
            hash: Some(hash),
            preimage: None,
            offer_id: None,
            txid: None,
            amount: Some(Amount::from_msat(1_000)),
            fees: Amount::from_msat(0),
            partner_pk: None,
            client_pk: None,
            partner_prop_fee: None,
            partner_base_fee: None,
            status,
            status_msg: status_msg.into(),
            address: None,
            invoice: None,
            tx: None,
            payer_name: None,
            message: None,
            personal_note: None,
            priority: None,
            expires_at: None,
            finalized_at: None,
            created_at,
            updated_at: TimestampMs::from_millis(updated_at_ms).unwrap(),
        }
    }

    fn setup() -> (Arc<MockWallet>, Arc<PaymentWatcher<MockWallet>>) {
        let wallet = Arc::new(MockWallet::default());
        let watcher = Arc::new(PaymentWatcher {
            wallet: Arc::clone(&wallet),
            cursor: Default::default(),
            state: Default::default(),
            kick: Notify::new(),
        });
        (wallet, watcher)
    }

    fn num_watched(watcher: &PaymentWatcher<MockWallet>) -> usize {
        watcher.state.lock().unwrap().watched.len()
    }

    fn num_alive_tasks() -> usize {
        tokio::runtime::Handle::current()
            .metrics()
            .num_alive_tasks()
    }

    fn latest(updates: &mut watch::Receiver<Option<Transaction>>) -> Transaction {
        updates.borrow_and_update().clone().unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn publishes_each_update_to_its_subscribers() {
        let (wallet, watcher) = setup();
        wallet.update(payment(hash(1), 10, PENDING));
        let mut updates = watcher.subscribe(hash(1));
        let unrelated = watcher.subscribe(hash(2));

        sleep(FAST_POLL).await;
        assert_eq!(latest(&mut updates).status, TransactionStatus::Pending);

        wallet.update(payment(hash(1), 20, CLAIMING));
        sleep(FAST_POLL).await;
        assert!(latest(&mut updates).is_claiming());

        wallet.update(payment(hash(1), 30, COMPLETED));
        sleep(FAST_POLL).await;
        assert_eq!(latest(&mut updates).status, TransactionStatus::Succeeded);

        assert!(unrelated.borrow().is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn subscribers_to_one_payment_share_an_entry() {
        let (wallet, watcher) = setup();
        let mut first = watcher.subscribe(hash(1));
        let mut second = watcher.subscribe(hash(1));
        assert_eq!(num_watched(&watcher), 1);

        wallet.update(payment(hash(1), 10, COMPLETED));
        sleep(FAST_POLL).await;

        // Each receiver tracks what it has seen independently.
        assert_eq!(latest(&mut first).status, TransactionStatus::Succeeded);
        assert!(second.has_changed().unwrap());
        assert_eq!(latest(&mut second).status, TransactionStatus::Succeeded);
    }

    #[tokio::test(start_paused = true)]
    async fn drops_unwatched_payments_and_stops_polling() {
        let (_, watcher) = setup();
        let first = watcher.subscribe(hash(1));
        let second = watcher.subscribe(hash(2));
        sleep(FAST_POLL).await;
        assert_eq!(num_watched(&watcher), 2);
        assert_eq!(num_alive_tasks(), 1);

        // Each entry is dropped within one poll of its last receiver leaving.
        drop(first);
        sleep(MAX_POLL).await;
        assert_eq!(num_watched(&watcher), 1);
        drop(second);
        sleep(MAX_POLL).await;
        assert_eq!(num_watched(&watcher), 0);

        // With nothing watched, the poller task has exited and released its
        // reference to the watcher.
        assert_eq!(num_alive_tasks(), 0);
        assert_eq!(Arc::strong_count(&watcher), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn restarted_poller_resumes_from_the_cursor() {
        let (wallet, watcher) = setup();
        let first_payment = payment(hash(1), 10, COMPLETED);
        let first_index = first_payment.updated_index();
        wallet.update(first_payment);

        // Consume the first update, then let the poller exit.
        let updates = watcher.subscribe(hash(1));
        sleep(FAST_POLL).await;
        drop(updates);
        sleep(MAX_POLL).await;
        assert_eq!(num_watched(&watcher), 0);
        let num_polls = wallet.poll_times().len();

        let mut updates = watcher.subscribe(hash(2));
        wallet.update(payment(hash(2), 20, COMPLETED));
        sleep(FAST_POLL).await;

        assert_eq!(wallet.polls.lock().unwrap()[num_polls].1, Some(first_index));
        assert_eq!(latest(&mut updates).status, TransactionStatus::Succeeded);
    }

    #[tokio::test(start_paused = true)]
    async fn idle_polling_backs_off_until_something_changes() {
        let (wallet, watcher) = setup();
        let start = Instant::now();
        let _updates = watcher.subscribe(hash(1));

        // Fast polls for the window, then doubling up to the slowest cadence.
        sleep(Duration::from_secs(30)).await;
        let gaps = |times: &[Instant]| {
            times
                .windows(2)
                .map(|pair| pair[1] - pair[0])
                .collect::<Vec<_>>()
        };
        let times = wallet.poll_times();
        assert_eq!(times[0], start);
        let idle_gaps = gaps(&times);
        let num_fast = (FAST_WINDOW.as_millis() / FAST_POLL.as_millis()) as usize;
        assert!(idle_gaps[..num_fast].iter().all(|gap| *gap == FAST_POLL));
        assert!(idle_gaps.iter().all(|gap| *gap <= MAX_POLL));
        assert_eq!(idle_gaps.last(), Some(&MAX_POLL));

        // A new subscriber is polled for immediately.
        let subscribed_at = Instant::now();
        let _other = watcher.subscribe(hash(2));
        sleep(Duration::from_millis(1)).await;
        assert_eq!(wallet.poll_times().last(), Some(&subscribed_at));

        // Once backed off again, an update restores the fast cadence.
        sleep(Duration::from_secs(30)).await;
        wallet.update(payment(hash(1), 10, PENDING));
        let num_polls = wallet.poll_times().len();
        sleep(Duration::from_secs(5)).await;
        let times = wallet.poll_times();
        assert!(
            gaps(&times[num_polls..])
                .iter()
                .all(|gap| *gap == FAST_POLL)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn poll_failures_do_not_stop_the_watcher() {
        let (wallet, watcher) = setup();
        wallet.failures.store(3, Ordering::Relaxed);
        wallet.update(payment(hash(1), 10, COMPLETED));

        let mut updates = watcher.subscribe(hash(1));
        sleep(FAST_POLL * 4).await;

        assert_eq!(latest(&mut updates).status, TransactionStatus::Succeeded);
    }

    #[tokio::test]
    async fn lookup_reads_payments_from_before_startup() {
        let (wallet, watcher) = setup();
        wallet.update(payment(hash(1), 10, COMPLETED));

        let found = watcher.lookup(hash(1)).await.unwrap().unwrap();
        assert_eq!(found.status, TransactionStatus::Succeeded);
        assert_eq!(wallet.poll_times().len(), 1);
    }

    #[tokio::test]
    async fn lookup_reads_known_payments_by_index() {
        let (wallet, watcher) = setup();
        wallet.update(payment(hash(1), 10, PENDING));
        watcher.lookup(hash(1)).await.unwrap();

        wallet.update(payment(hash(1), 20, COMPLETED));
        let found = watcher.lookup(hash(1)).await.unwrap().unwrap();
        assert_eq!(found.status, TransactionStatus::Succeeded);
        assert_eq!(wallet.poll_times().len(), 1);
    }

    #[tokio::test]
    async fn lookup_miss_finds_the_payment_once_it_exists() {
        let (wallet, watcher) = setup();
        assert!(watcher.lookup(hash(1)).await.unwrap().is_none());

        wallet.update(payment(hash(1), 10, PENDING));
        let found = watcher.lookup(hash(1)).await.unwrap().unwrap();
        assert_eq!(found.status, TransactionStatus::Pending);
    }
}

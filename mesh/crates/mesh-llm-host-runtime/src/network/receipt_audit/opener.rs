//! Sends a challenge to the peer that served a turn, over a mesh tunnel to
//! its OpenAI ingress (`POST /v1/receipts/open`, then
//! `POST /v1/receipts/rows` for the attention rows), as a client of that
//! peer would. `x-mesh-target` names the peer the request was sent to, so
//! that peer opens it itself.
//!
//! A peer that does not answer is struck, so this node must not take its
//! own slowness for the peer's silence, and one slow peer must not hold up
//! the audits of the others:
//!
//! - The node keeps a turn's replay state for the challenge window after
//!   the receipt. The window is checked when the challenge is sent, after
//!   the claimed model is open. A challenge this node is too late to send
//!   is not sent, and that is no audit ([`Late::Challenge`]).
//! - The answer may take what is left of the window and [`OPEN_GRACE`], no
//!   more: a peer that slow-walks its answer costs this node one window.
//! - The node keeps the attention rows for [`ROWS_PIN`] after it opened.
//!   This node asks for them after its own checks of the opening, which
//!   may take long the first time (the head key is built then). Asked too
//!   late, the rows are not asked for, and that is no audit of them
//!   ([`Late::Rows`]). The rows' answer may take [`ROWS_WAIT`].
//! - An answer is read up to the size its challenge can produce
//!   (`answer_limits`).

use std::cell::Cell;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use iroh::EndpointId;
use skippy_receipt_audit::turn_audit::{OpenChallenge, Opened, OpenedRows};
use skippy_receipts::attention::opening::{BlockChallenge, ROWS_PIN};
use skippy_receipts::wire::Challenge;

use super::answer_limits::AnswerLimits;
use super::http_answer::{Answer, read_answer};
use super::peer_text::shown_bytes;
use crate::mesh;

/// A challenge sent later than this after the receipt may find the node's
/// replay state gone.
pub(super) const CHALLENGE_WINDOW: Duration = Duration::from_secs(10);
/// How long past the window an opening may still be on its way.
pub(super) const OPEN_GRACE: Duration = Duration::from_secs(5);
/// How long a peer has to answer the request for attention rows.
pub(super) const ROWS_WAIT: Duration = Duration::from_secs(10);
/// The part of [`ROWS_PIN`] left for the rows request to reach the node.
const ROWS_TRANSIT: Duration = Duration::from_secs(2);

/// How this node itself was too late to ask a peer for something.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Late {
    /// The challenge was ready this long after the receipt.
    Challenge(Duration),
    /// The rows request was ready this long after the challenge was sent.
    Rows(Duration),
}

impl std::fmt::Display for Late {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Challenge(waited) => write!(
                f,
                "this node was ready to send the challenge {:.1} s after the receipt, past the {} s challenge window",
                waited.as_secs_f64(),
                CHALLENGE_WINDOW.as_secs()
            ),
            Self::Rows(waited) => write!(
                f,
                "the attention rows were not checked: this node was ready to ask for them {:.1} s after its challenge, and the node keeps them {} s",
                waited.as_secs_f64(),
                ROWS_PIN.as_secs()
            ),
        }
    }
}

/// Carries one request to the peer and its answer back.
pub(super) trait Transport {
    /// Posts `body` to `/v1/receipts/{action}` on the peer. The answer is
    /// waited for at most `wait`, and its body read up to `max_body`
    /// bytes. An error is no answer.
    fn post(
        &self,
        action: &'static str,
        body: &[u8],
        wait: Duration,
        max_body: usize,
    ) -> Result<Answer>;
}

/// Opens the challenges of one turn, within the times the node keeps what
/// they need.
pub(super) struct WindowedOpener<'a> {
    transport: &'a dyn Transport,
    /// The model the request named, which the peer routes the challenge by.
    model: &'a str,
    limits: &'a AnswerLimits,
    /// When the receipt arrived.
    received: Instant,
    clock: &'a dyn Fn() -> Instant,
    late: Cell<Option<Late>>,
    challenged: Cell<Option<Instant>>,
}

impl<'a> WindowedOpener<'a> {
    pub(super) fn new(
        transport: &'a dyn Transport,
        model: &'a str,
        limits: &'a AnswerLimits,
        received: Instant,
        clock: &'a dyn Fn() -> Instant,
    ) -> Self {
        Self {
            transport,
            model,
            limits,
            received,
            clock,
            late: Cell::new(None),
            challenged: Cell::new(None),
        }
    }

    /// How this node was too late to ask, if it was.
    pub(super) fn late(&self) -> Option<Late> {
        self.late.get()
    }

    /// Posts `request`, naming the model, and returns the peer's answer
    /// when its status is 200.
    fn post<B: serde::Serialize>(
        &self,
        action: &'static str,
        request: &B,
        wait: Duration,
        max_body: usize,
    ) -> Result<Answer> {
        let mut body = serde_json::to_value(request)?;
        body["model"] = self.model.into();
        let answer = self
            .transport
            .post(action, &serde_json::to_vec(&body)?, wait, max_body)?;
        if answer.status != 200 {
            bail!("{}: {}", answer.status, shown_bytes(&answer.body));
        }
        Ok(answer)
    }
}

impl OpenChallenge for WindowedOpener<'_> {
    fn open(&self, challenge: &Challenge) -> Result<Opened> {
        let now = (self.clock)();
        let waited = now.saturating_duration_since(self.received);
        if waited > CHALLENGE_WINDOW {
            self.late.set(Some(Late::Challenge(waited)));
            bail!("the challenge was not sent");
        }
        let wait = CHALLENGE_WINDOW - waited + OPEN_GRACE;
        self.challenged.set(Some(now));
        let answer = self.post("open", challenge, wait, self.limits.opening(challenge))?;
        let opening = serde_json::from_slice(&answer.body).context("decode the opening")?;
        Ok(Opened {
            opening,
            bytes: answer.body,
            signature: answer.signature,
            seconds: (self.clock)().saturating_duration_since(now).as_secs_f64(),
        })
    }

    fn rows(&self, blocks: &BlockChallenge) -> Result<OpenedRows> {
        let now = (self.clock)();
        let since = self
            .challenged
            .get()
            .map_or(Duration::ZERO, |at| now.saturating_duration_since(at));
        // the node opened no earlier than the challenge was sent
        if since > ROWS_PIN - ROWS_TRANSIT {
            self.late.set(Some(Late::Rows(since)));
            bail!("the attention rows were not asked for");
        }
        let answer = self.post("rows", blocks, ROWS_WAIT, self.limits.rows(blocks))?;
        let rows = serde_json::from_slice(&answer.body).context("decode the attention rows")?;
        Ok(OpenedRows {
            rows,
            bytes: answer.body,
            signature: answer.signature,
            seconds: (self.clock)().saturating_duration_since(now).as_secs_f64(),
        })
    }
}

/// The mesh tunnel to the peer a turn was asked of.
pub(super) struct TunnelTransport {
    pub(super) node: mesh::Node,
    pub(super) host: EndpointId,
    pub(super) runtime: tokio::runtime::Handle,
}

impl Transport for TunnelTransport {
    fn post(
        &self,
        action: &'static str,
        body: &[u8],
        wait: Duration,
        max_body: usize,
    ) -> Result<Answer> {
        // the timer belongs to the runtime, so it is made inside it
        self.runtime
            .block_on(within(wait, action, self.exchange(action, body, max_body)))
    }
}

impl TunnelTransport {
    async fn exchange(&self, action: &str, body: &[u8], max_body: usize) -> Result<Answer> {
        let (mut send, recv) = self.node.open_http_tunnel(self.host).await?;
        let head = format!(
            "POST /v1/receipts/{action} HTTP/1.1\r\nHost: mesh\r\nContent-Type: application/json\r\nContent-Length: {}\r\nx-mesh-target: {}\r\nConnection: close\r\n\r\n",
            body.len(),
            hex::encode(self.host.as_bytes())
        );
        send.write_all(head.as_bytes()).await?;
        send.write_all(body).await?;
        send.finish()?;
        read_answer(recv, max_body).await
    }
}

/// `exchange`, or an error when it takes longer than `wait`: the whole
/// exchange, from opening the tunnel to the last byte of the answer.
async fn within<T>(
    wait: Duration,
    action: &str,
    exchange: impl Future<Output = Result<T>>,
) -> Result<T> {
    tokio::time::timeout(wait, exchange).await.map_err(|_| {
        anyhow!(
            "no whole answer to /v1/receipts/{action} within {:.1} s",
            wait.as_secs_f64()
        )
    })?
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use skippy_receipts::toy_turn::{self, Turn};
    use tokio::io::AsyncWriteExt;

    use super::*;
    use crate::network::receipt_audit::answer_limits::MAX_ANSWER_BYTES;

    /// A peer that answers as its script says, and a clock the script moves.
    struct Scripted {
        turn: Turn,
        /// What each request finds: how long the answer takes, and its status.
        takes: Duration,
        status: u16,
        now: Cell<Instant>,
        asked: RefCell<Vec<(&'static str, Duration, usize)>>,
    }

    impl Scripted {
        fn new(takes: Duration) -> Self {
            Self {
                turn: Turn::new(),
                takes,
                status: 200,
                now: Cell::new(Instant::now()),
                asked: RefCell::default(),
            }
        }
    }

    impl Transport for Scripted {
        fn post(
            &self,
            action: &'static str,
            body: &[u8],
            wait: Duration,
            max_body: usize,
        ) -> Result<Answer> {
            self.asked.borrow_mut().push((action, wait, max_body));
            if self.takes > wait {
                self.now.set(self.now.get() + wait);
                bail!("no whole answer to /v1/receipts/{action} within {wait:?}");
            }
            self.now.set(self.now.get() + self.takes);
            let request: serde_json::Value = serde_json::from_slice(body)?;
            assert_eq!(request["model"], "asked-model");
            let body = match action {
                "open" => {
                    let challenge: Challenge = serde_json::from_value(request)?;
                    serde_json::to_vec(&self.turn.open(&challenge))?
                }
                _ => br#"{"receipt_id":"toy-turn","stages":[]}"#.to_vec(),
            };
            Ok(Answer {
                status: self.status,
                signature: Some("ab".into()),
                body,
            })
        }
    }

    fn opened(peer: &Scripted, after: Duration) -> (Result<Opened>, Option<Late>) {
        let limits = AnswerLimits::of(&toy_turn::weights(), &peer.turn.receipt).unwrap();
        let received = peer.now.get();
        peer.now.set(received + after);
        let clock = || peer.now.get();
        let opener = WindowedOpener::new(peer, "asked-model", &limits, received, &clock);
        let challenge = peer.turn.challenge(&[(0, &[3])], &[]);
        let result = opener.open(&challenge);
        (result, opener.late())
    }

    #[test]
    fn the_answer_may_take_what_is_left_of_the_window_and_a_grace() {
        for (after, wait) in [(0, 15), (4, 11), (10, 5)] {
            let peer = Scripted::new(Duration::from_secs(1));
            let (result, late) = opened(&peer, Duration::from_secs(after));
            assert!(result.is_ok() && late.is_none(), "sent after {after} s");
            let asked = peer.asked.borrow();
            assert_eq!(asked[0].0, "open");
            assert_eq!(
                asked[0].1,
                Duration::from_secs(wait),
                "sent after {after} s"
            );
        }
    }

    #[test]
    fn a_challenge_this_node_is_late_to_send_is_not_sent() {
        let peer = Scripted::new(Duration::from_secs(1));
        let (result, late) = opened(&peer, Duration::from_millis(10_001));
        assert!(result.is_err());
        assert_eq!(late, Some(Late::Challenge(Duration::from_millis(10_001))));
        assert!(peer.asked.borrow().is_empty(), "nothing was asked");
        assert!(late.unwrap().to_string().contains("past the 10 s"));
    }

    #[test]
    fn a_peer_that_slow_walks_its_answer_costs_one_window() {
        // the peer would answer after a minute
        let peer = Scripted::new(Duration::from_secs(60));
        let started = peer.now.get();
        let (result, late) = opened(&peer, Duration::ZERO);
        assert!(result.is_err());
        assert_eq!(late, None, "the peer's silence is not this node's lateness");
        let spent = peer.now.get() - started;
        assert_eq!(spent, CHALLENGE_WINDOW + OPEN_GRACE);
        assert!(spent < Duration::from_secs(20));
    }

    #[test]
    fn an_opening_is_read_up_to_what_its_challenge_can_produce() {
        let peer = Scripted::new(Duration::ZERO);
        let (result, _) = opened(&peer, Duration::ZERO);
        let opened = result.unwrap();
        let max_body = peer.asked.borrow()[0].2;
        assert!(opened.bytes.len() <= max_body);
        // the toy model's bound, not the cap for every model
        assert!(max_body < MAX_ANSWER_BYTES / 100, "{max_body}");
    }

    #[test]
    fn an_error_status_is_no_answer_and_its_body_is_shown_cut() {
        let mut peer = Scripted::new(Duration::ZERO);
        peer.status = 503;
        let (result, late) = opened(&peer, Duration::ZERO);
        let error = format!("{:#}", result.err().expect("no answer"));
        assert!(error.starts_with("503: "), "{error}");
        assert!(error.chars().count() < 300, "{}", error.len());
        assert_eq!(late, None);
    }

    fn rows_after(checks_took: Duration) -> (Result<OpenedRows>, Option<Late>, usize) {
        let peer = Scripted::new(Duration::from_secs(1));
        let limits = AnswerLimits::of(&toy_turn::weights(), &peer.turn.receipt).unwrap();
        let received = peer.now.get();
        let clock = || peer.now.get();
        let opener = WindowedOpener::new(&peer, "asked-model", &limits, received, &clock);
        opener
            .open(&peer.turn.challenge(&[(0, &[3])], &[]))
            .expect("an opening");
        // this node's own checks of the opening
        peer.now.set(peer.now.get() + checks_took);
        let blocks = BlockChallenge {
            receipt_id: "toy-turn".into(),
            picks: Vec::new(),
            proof: None,
        };
        let result = opener.rows(&blocks);
        (result, opener.late(), peer.asked.borrow().len())
    }

    #[test]
    fn rows_asked_for_past_the_nodes_pin_are_this_nodes_lateness() {
        // the opening took 1 s to arrive; the pin leaves 28 s for the checks
        let (result, late, asked) = rows_after(Duration::from_secs(26));
        assert!(result.is_ok() && late.is_none());
        assert_eq!(asked, 2);
        let (result, late, asked) = rows_after(Duration::from_secs(40));
        assert!(result.is_err());
        assert_eq!(late, Some(Late::Rows(Duration::from_secs(41))));
        assert_eq!(asked, 1, "the rows were not asked for");
    }

    #[tokio::test(start_paused = true)]
    async fn an_exchange_that_stalls_ends_at_its_wait() {
        // a peer that sends a head, then nothing, and keeps the stream open
        let (mut peer, node) = tokio::io::duplex(1024);
        peer.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{")
            .await
            .unwrap();
        let started = tokio::time::Instant::now();
        let wait = Duration::from_secs(15);
        let result = within(wait, "open", read_answer(node, 1 << 20)).await;
        let error = format!("{:#}", result.expect_err("no whole answer"));
        assert!(
            error.contains("no whole answer to /v1/receipts/open"),
            "{error}"
        );
        assert_eq!(started.elapsed(), wait);
        drop(peer);
    }
}

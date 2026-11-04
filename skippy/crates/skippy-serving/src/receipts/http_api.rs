//! The receipts HTTP routes: a stage's own API on its receipts address, and
//! the routes a node serves on its OpenAI surface for clients that reach it
//! only there.
//!
//! Every answer that is not an opening says which kind it is by its status:
//!
//! - 404: the stage keeps no record of the receipt id;
//! - 403: the route opens only bound receipts, and this one is not bound;
//! - 409 or 410: the request is one the stage would answer, but it can no
//!   longer open the turn ([`CannotOpen`]); 410 when the turn left the
//!   opening window, 409 when a later request overwrote its cells or the
//!   stage lost track of them. The body holds `cannot_open` with the reason
//!   and the stage, and for a bound receipt the node signs it;
//! - 503: the stage was unloaded;
//! - 400: anything else, a refused or malformed request among them.
//!
//! The routes hold the stage only weakly, so a host that keeps a router
//! after it unloads the model keeps neither the model nor the receipts.

use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;

use anyhow::{Context, Result, anyhow};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::commitment::{Hash, hex_array};
use skippy_receipts::turn;
use skippy_receipts::wire::{CannotOpen, CannotOpenReason, Challenge, Receipt, StageCommitment};

use super::downstream::CommitRequest;
use super::refusal::{NotFound, Unavailable};
use super::{OPENING_DIGEST_HEADER, SessionTurns, StageReceipts, TURN_ROOT_HEADER};
use crate::runtime_state::RuntimeState;

/// What the routes reach: the stage's receipts and its sessions, while the
/// stage is loaded.
#[derive(Clone)]
pub(super) struct ApiState {
    pub receipts: Weak<StageReceipts>,
    pub sessions: Weak<dyn SessionTurns>,
}

impl ApiState {
    pub(super) fn of(receipts: &Arc<StageReceipts>, runtime: &Arc<Mutex<RuntimeState>>) -> Self {
        let sessions: Weak<Mutex<RuntimeState>> = Arc::downgrade(runtime);
        Self {
            receipts: Arc::downgrade(receipts),
            sessions,
        }
    }

    fn receipts(&self) -> Result<Arc<StageReceipts>> {
        self.receipts
            .upgrade()
            .ok_or_else(|| anyhow::Error::new(Unavailable("this stage was unloaded".to_owned())))
    }
}

/// The receipts API thread of a stage; it stops serving when this is
/// dropped.
pub(super) struct ApiHandle {
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl ApiHandle {
    /// The API thread, for a test to wait for its end.
    #[cfg(test)]
    pub(super) fn take_thread(&mut self) -> JoinHandle<()> {
        self.thread.take().expect("the receipts API thread")
    }
}

impl Drop for ApiHandle {
    /// Tells the thread to stop. It is not waited for: a request it still
    /// answers may be what drops the stage.
    fn drop(&mut self) {
        self.stop.take();
        self.thread.take();
    }
}

pub(super) fn spawn_api(listener: std::net::TcpListener, router: Router) -> Result<ApiHandle> {
    listener.set_nonblocking(true)?;
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let thread = std::thread::Builder::new()
        .name("skippy-receipts-api".into())
        .spawn(move || serve(listener, router, stopped))
        .context("spawn receipts API thread")?;
    Ok(ApiHandle {
        stop: Some(stop),
        thread: Some(thread),
    })
}

fn serve(
    listener: std::net::TcpListener,
    router: Router,
    stopped: tokio::sync::oneshot::Receiver<()>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::error!("receipts API runtime: {error}");
            return;
        }
    };
    runtime.block_on(async move {
        let listener = match tokio::net::TcpListener::from_std(listener) {
            Ok(listener) => listener,
            Err(error) => {
                tracing::error!("receipts API listener: {error}");
                return;
            }
        };
        // the sender is dropped, never used: either way the stage is gone
        let stopped = async move {
            let _ = stopped.await;
        };
        if let Err(error) = axum::serve(listener, router)
            .with_graceful_shutdown(stopped)
            .await
        {
            tracing::error!("receipts API stopped: {error}");
        }
    });
}

/// A stage's own receipts API.
pub(super) fn router(state: ApiState) -> Router {
    Router::new()
        .route("/receipts/commit", post(commit_handler))
        .route("/receipts/open", post(open_handler))
        .route("/receipts/rows", post(rows_handler))
        .route("/receipts/{id}", get(receipt_handler))
        .with_state(state)
}

/// The routes of the OpenAI surface: `POST /v1/receipts/open` and
/// `POST /v1/receipts/rows`, for bound receipts only.
pub(super) fn client_routes(state: ApiState) -> Router {
    Router::new()
        .route("/v1/receipts/open", post(bound_open_handler))
        .route("/v1/receipts/rows", post(bound_rows_handler))
        .with_state(state)
}

/// An answer that is not the one asked for: its status, its JSON body, and
/// the node's signature of the body when it signs one.
struct ApiError {
    status: StatusCode,
    body: serde_json::Value,
    cannot_open: Option<CannotOpen>,
}

impl ApiError {
    fn new(status: StatusCode, error: &anyhow::Error) -> Self {
        Self {
            status,
            body: serde_json::json!({ "error": format!("{error:#}") }),
            cannot_open: None,
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        if let Some(cannot) = error.downcast_ref::<CannotOpen>() {
            let status = match cannot.reason {
                CannotOpenReason::Expired => StatusCode::GONE,
                _ => StatusCode::CONFLICT,
            };
            return Self {
                status,
                body: serde_json::json!({ "error": cannot.to_string(), "cannot_open": cannot }),
                cannot_open: Some(cannot.clone()),
            };
        }
        let status = if error.downcast_ref::<NotFound>().is_some() {
            StatusCode::NOT_FOUND
        } else if error.downcast_ref::<Unavailable>().is_some() {
            StatusCode::SERVICE_UNAVAILABLE
        } else {
            StatusCode::BAD_REQUEST
        };
        Self::new(status, &error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| ApiError::from(anyhow!("receipts worker failed: {error}")))?
        .map_err(ApiError::from)
}

/// The hash a stage before this one sent along in `header`, if any.
fn forwarded_hash(headers: &HeaderMap, header: &str) -> Result<Option<Hash>, ApiError> {
    headers
        .get(header)
        .map(|value| {
            let text = value.to_str().context("a forwarded hash is hex")?;
            hex_array(text, "a forwarded hash")
        })
        .transpose()
        .map_err(ApiError::from)
}

async fn commit_handler(
    State(state): State<ApiState>,
    Json(request): Json<CommitRequest>,
) -> Result<Json<Vec<StageCommitment>>, ApiError> {
    blocking(move || {
        let receipts = state.receipts()?;
        let sessions = state
            .sessions
            .upgrade()
            .ok_or_else(|| anyhow::Error::new(Unavailable("this stage was unloaded".into())))?;
        receipts.commit_session(sessions.as_ref(), &request)
    })
    .await
    .map(Json)
}

/// Answers with the opening's bytes and, for a bound receipt, the node's
/// signature of exactly those bytes.
async fn open_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(challenge): Json<Challenge>,
) -> Result<Response, ApiError> {
    let root = forwarded_hash(&headers, TURN_ROOT_HEADER)?;
    opening_response(state, challenge, root).await
}

/// Refuses a receipt the client routes do not open: one that is not bound.
fn bound_only(state: &ApiState, id: &str) -> Result<(), ApiError> {
    match state.receipts().and_then(|receipts| receipts.is_bound(id)) {
        Ok(true) => Ok(()),
        Ok(false) => Err(ApiError::new(
            StatusCode::FORBIDDEN,
            &anyhow!("receipt {id} is not bound to a client key"),
        )),
        Err(error) => Err(ApiError::from(error)),
    }
}

/// [`open_handler`] for a bound receipt only. The body may also name the
/// `model`, which a Mesh node routes the request by.
async fn bound_open_handler(
    State(state): State<ApiState>,
    Json(challenge): Json<Challenge>,
) -> Result<Response, ApiError> {
    bound_only(&state, &challenge.receipt_id)?;
    opening_response(state, challenge, None).await
}

/// The JSON `answer` with the node's signature, or the error of a turn the
/// node cannot open, signed likewise when the receipt is one it signs for.
fn signed_answer(
    receipts: &StageReceipts,
    receipt_id: &str,
    request: &Hash,
    answer: Result<(Vec<u8>, Option<String>)>,
) -> Result<(StatusCode, Vec<u8>, Option<String>)> {
    match answer {
        Ok((bytes, signature)) => Ok((StatusCode::OK, bytes, signature)),
        Err(error) => {
            let refused = ApiError::from(error);
            let bytes = serde_json::to_vec(&refused.body)?;
            let signature = refused
                .cannot_open
                .and_then(|_| receipts.sign_cannot_open(receipt_id, request, &bytes));
            Ok((refused.status, bytes, signature))
        }
    }
}

async fn opening_response(
    state: ApiState,
    challenge: Challenge,
    forwarded_root: Option<Hash>,
) -> Result<Response, ApiError> {
    let (status, bytes, signature) = blocking(move || {
        let receipts = state.receipts()?;
        let opened = receipts
            .open(&challenge, forwarded_root)
            .and_then(|opening| {
                let bytes = serde_json::to_vec(&opening)?;
                let signature = receipts.sign_opening(&challenge, &bytes)?;
                Ok((bytes, signature))
            });
        signed_answer(
            &receipts,
            &challenge.receipt_id,
            &challenge.digest(),
            opened,
        )
    })
    .await?;
    signed_response(status, bytes, signature)
}

/// A JSON body with the node's signature of it, when it signed it.
fn signed_response(
    status: StatusCode,
    bytes: Vec<u8>,
    signature: Option<String>,
) -> Result<Response, ApiError> {
    let mut response = (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        bytes,
    )
        .into_response();
    if let Some(signature) = signature {
        response.headers_mut().insert(
            turn::OPENING_SIGNATURE_HEADER,
            signature
                .parse()
                .context("a hex signature is a header value")?,
        );
    }
    Ok(response)
}

/// Answers with the rows' bytes and, for a bound receipt, the node's
/// signature of exactly those bytes.
async fn rows_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(challenge): Json<BlockChallenge>,
) -> Result<Response, ApiError> {
    let opening = forwarded_hash(&headers, OPENING_DIGEST_HEADER)?;
    rows_response(state, challenge, opening).await
}

/// [`rows_handler`] for a bound receipt that was challenged.
async fn bound_rows_handler(
    State(state): State<ApiState>,
    Json(challenge): Json<BlockChallenge>,
) -> Result<Response, ApiError> {
    bound_only(&state, &challenge.receipt_id)?;
    rows_response(state, challenge, None).await
}

async fn rows_response(
    state: ApiState,
    blocks: BlockChallenge,
    forwarded_opening: Option<Hash>,
) -> Result<Response, ApiError> {
    let (status, bytes, signature) = blocking(move || {
        let receipts = state.receipts()?;
        let opened = receipts.rows(&blocks, forwarded_opening).and_then(|rows| {
            let bytes = serde_json::to_vec(&rows)?;
            let signature = receipts.sign_rows(&blocks, &bytes)?;
            Ok((bytes, signature))
        });
        signed_answer(&receipts, &blocks.receipt_id, &blocks.digest(), opened)
    })
    .await?;
    signed_response(status, bytes, signature)
}

async fn receipt_handler(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Receipt>, ApiError> {
    // stage 0 commits a turn as its response ends, so a client may ask
    // before the receipt exists
    match state.receipts()?.receipt(&id) {
        Some(Ok(receipt)) => Ok(Json(Receipt::clone(&receipt))),
        Some(Err(reason)) => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            &anyhow!("turn {id} got no receipt: {reason}"),
        )),
        None => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            &anyhow!("no receipt {id}"),
        )),
    }
}

#[cfg(test)]
#[path = "http_api_tests.rs"]
mod tests;

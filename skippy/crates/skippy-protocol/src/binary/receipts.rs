//! The receipts messages of the stage connection: a request with a JSON
//! body ([`WireMessageKind::Receipts`]), and a reply that says what kind of
//! answer it is ([`WireReplyKind::Receipts`]) and is read only up to what
//! its request can produce.

use std::io::{self, Read, Write};

use super::codec::{RawBytes, read_i32, read_u32, write_i32, write_u32};
use super::{
    StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind, invalid_data, invalid_input,
};

/// A receipts request: a JSON commit or challenge.
pub const MAX_STAGE_RECEIPTS_REQUEST_BYTES: usize = 1024 * 1024;
/// The most a receipts reply can hold. A stage that waits for one bounds it
/// further, by what its own request can produce ([`recv_receipts_reply`]).
pub const MAX_STAGE_RECEIPTS_REPLY_BYTES: usize = 256 * 1024 * 1024;
/// A receipts reply that is not an answer: why a stage refused, or its
/// "cannot open" result.
pub const MAX_STAGE_RECEIPTS_REFUSAL_BYTES: usize = 64 * 1024;

/// The bound on a receipts request's body in `raw_bytes`, and how its
/// violations read.
pub(super) const REQUEST_BODY: RawBytes = RawBytes {
    max: MAX_STAGE_RECEIPTS_REQUEST_BYTES,
    mismatch: "receipts request byte count mismatch",
    too_large: "receipts request byte count exceeds maximum",
    count_too_large: "receipts request byte count exceeds maximum",
    over_budget: "receipts request byte count exceeds frame budget",
};

/// A stage's reply to a receipts request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptsReply {
    /// The JSON answer: commitments, an opening or rows.
    Answer(Vec<u8>),
    /// Why the stage refused the request.
    Refused(String),
    /// The stage knows no receipt of that id, and why.
    NotFound(String),
    /// The stage can no longer open the turn, though the request is one it
    /// would answer: its JSON "cannot open" result.
    CannotOpen(Vec<u8>),
    /// The stage issues no receipts, and why: asking it again is no use.
    Unavailable(String),
}

impl ReceiptsReply {
    fn status(&self) -> u32 {
        match self {
            Self::Answer(_) => 0,
            Self::Refused(_) => 1,
            Self::NotFound(_) => 2,
            Self::CannotOpen(_) => 3,
            Self::Unavailable(_) => 4,
        }
    }

    fn body(&self) -> &[u8] {
        match self {
            Self::Answer(body) | Self::CannotOpen(body) => body,
            Self::Refused(reason) | Self::NotFound(reason) | Self::Unavailable(reason) => {
                reason.as_bytes()
            }
        }
    }

    fn from_wire(status: u32, body: Vec<u8>) -> io::Result<Self> {
        let text = |body: Vec<u8>| String::from_utf8_lossy(&body).into_owned();
        match status {
            0 => Ok(Self::Answer(body)),
            1 => Ok(Self::Refused(text(body))),
            2 => Ok(Self::NotFound(text(body))),
            3 => Ok(Self::CannotOpen(body)),
            4 => Ok(Self::Unavailable(text(body))),
            _ => Err(invalid_data("unknown receipts reply status")),
        }
    }
}

impl StageWireMessage {
    /// A receipts request with a JSON `body`, for the stage that receives it
    /// and every later one.
    pub fn receipts(body: Vec<u8>) -> Self {
        Self {
            kind: WireMessageKind::Receipts,
            pos_start: 0,
            token_count: i32::try_from(body.len()).unwrap_or(i32::MAX),
            state: StageStateHeader::new(WireMessageKind::Receipts),
            request_id: 0,
            session_id: 0,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            readout_tokens: Vec::new(),
            raw_bytes: body,
        }
    }
}

/// Sends a stage's reply to a receipts request.
pub fn send_receipts_reply(mut writer: impl Write, reply: &ReceiptsReply) -> io::Result<()> {
    let body = reply.body();
    let max = match reply {
        ReceiptsReply::Answer(_) => MAX_STAGE_RECEIPTS_REPLY_BYTES,
        _ => MAX_STAGE_RECEIPTS_REFUSAL_BYTES,
    };
    if body.len() > max {
        return Err(invalid_input("receipts reply exceeds maximum"));
    }
    write_i32(&mut writer, WireReplyKind::Receipts as i32)?;
    write_u32(&mut writer, reply.status())?;
    write_u32(
        &mut writer,
        u32::try_from(body.len()).map_err(|_| invalid_input("receipts reply exceeds maximum"))?,
    )?;
    writer.write_all(body)
}

/// Reads a stage's reply to a receipts request, whose answer may hold at
/// most `max_answer_bytes`: what the request can produce, as its sender
/// bounds it. A longer answer is refused from its declared length, before
/// any of it is read. The body grows with the bytes that arrive, so a
/// declared length alone allocates nothing.
pub fn recv_receipts_reply(
    mut reader: impl Read,
    max_answer_bytes: usize,
) -> io::Result<ReceiptsReply> {
    if WireReplyKind::try_from(read_i32(&mut reader)?)? != WireReplyKind::Receipts {
        return Err(invalid_data("expected a receipts reply"));
    }
    let status = read_u32(&mut reader)?;
    let len = usize::try_from(read_u32(&mut reader)?)
        .map_err(|_| invalid_data("receipts reply exceeds maximum"))?;
    let max = match status {
        0 => max_answer_bytes.min(MAX_STAGE_RECEIPTS_REPLY_BYTES),
        _ => MAX_STAGE_RECEIPTS_REFUSAL_BYTES,
    };
    if len > max {
        return Err(invalid_data(
            "receipts reply exceeds what its request can produce",
        ));
    }
    let mut body = Vec::new();
    let read = (&mut reader).take(len as u64).read_to_end(&mut body)?;
    if read != len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "receipts reply ended early",
        ));
    }
    ReceiptsReply::from_wire(status, body)
}

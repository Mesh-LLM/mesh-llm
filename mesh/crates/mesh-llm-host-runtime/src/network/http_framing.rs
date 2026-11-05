//! HTTP/1.1 message framing that the OpenAI ingress, the HTTP tunnel and the
//! receipt audits share: where a message head ends, and chunked bodies.

pub(crate) mod chunked;

pub(crate) const CRLF: &[u8] = b"\r\n";
const LF: &[u8] = b"\n";
pub(crate) const CRLF_HEADER_TERMINATOR: &[u8] = b"\r\n\r\n";
const LF_HEADER_TERMINATOR: &[u8] = b"\n\n";

/// Where the message head in `prefix` ends, just past its blank line, and
/// the line ending it uses. A head may end lines with CRLF or a bare LF; the
/// first blank line of either kind ends it.
pub(crate) fn http_header_terminator(prefix: &[u8]) -> Option<(usize, &'static [u8])> {
    let crlf = prefix
        .windows(CRLF_HEADER_TERMINATOR.len())
        .position(|window| window == CRLF_HEADER_TERMINATOR)
        .map(|offset| (offset + CRLF_HEADER_TERMINATOR.len(), CRLF));
    let lf = prefix
        .windows(LF_HEADER_TERMINATOR.len())
        .position(|window| window == LF_HEADER_TERMINATOR)
        .map(|offset| (offset + LF_HEADER_TERMINATOR.len(), LF));

    match (crlf, lf) {
        (Some(crlf), Some(lf)) => Some(if crlf.0 <= lf.0 { crlf } else { lf }),
        (Some(terminator), None) | (None, Some(terminator)) => Some(terminator),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_blank_line_of_either_ending_ends_the_head() {
        assert_eq!(
            http_header_terminator(b"GET / HTTP/1.1\r\nHost: x\r\n\r\nbody"),
            Some((27, CRLF))
        );
        assert_eq!(
            http_header_terminator(b"GET / HTTP/1.1\nHost: x\n\nbody"),
            Some((24, LF))
        );
        // a bare-LF blank line inside a CRLF head ends it first
        assert_eq!(
            http_header_terminator(b"GET / HTTP/1.1\n\nHost: x\r\n\r\n"),
            Some((16, LF))
        );
        assert_eq!(
            http_header_terminator(b"GET / HTTP/1.1\r\nHost: x\r\n"),
            None
        );
    }
}

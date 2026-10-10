use super::{cli::run, protocol::Plan};

macro_rules! framing {
    ($name:ident, $wire:literal, $failure:expr) => {
        #[test]
        fn $name() {
            let plan = Plan {
                wire: Some($wire.to_vec()),
                ..Plan::default()
            };
            run(plan, $failure);
        }
    };
}

framing!(
    d13_chunked,
    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\n0\r\n\r\n",
    None
);
framing!(
    d13_short_length,
    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nx",
    Some("models_transfer_failed")
);
framing!(
    d13_short_chunk,
    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\n",
    Some("models_transfer_failed")
);
framing!(d13_close_delimited, b"HTTP/1.1 200 OK\r\n\r\nx", None);

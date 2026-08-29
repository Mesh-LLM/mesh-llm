//! `skippy-kv/1` — the network backend of the L3 stream contract.
//!
//! A peer serves its `HandoffSegmentStore` over a framed byte stream;
//! clients pull manifests and content-addressed segments by digest. Pulls
//! are idempotent: segments already present locally are skipped, and every
//! fetched segment is digest-verified before it lands in the local store,
//! so a corrupt or malicious peer cannot poison it. Ordering and
//! completeness come from the manifest, exactly as on disk.
//!
//! Transport is any `Read + Write` pair. In the mesh this rides an iroh
//! QUIC stream under the `skippy-kv/1` ALPN (`skippy_protocol::KV_ALPN_V1`)
//! bridged to a local socket, the same pattern the stage transport uses;
//! the harness drives it over plain TCP.
//!
//! **The server side has no authentication**: any process that can reach
//! the port can enumerate and drain the store. Digest verification protects
//! the *client* from a malicious peer, not this server from disclosure.
//! Plain-TCP serving is for the lab harness on trusted networks only — do
//! not wire it to a mesh-reachable listener; mesh exposure goes through the
//! `skippy-kv/1` iroh ALPN with mesh-membership auth.

use std::{
    io::{BufReader, BufWriter, Read, Write},
    net::{TcpListener, TcpStream},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::l3::{HandoffManifest, HandoffSegmentStore, segment_digest};

const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SEGMENT_BYTES: u64 = 256 * 1024 * 1024;
const STREAM_BUFFER_BYTES: usize = 1024 * 1024;

mod frame_kind {
    pub const GET_MANIFEST: u8 = 1;
    pub const MANIFEST: u8 = 2;
    pub const GET_SEGMENT: u8 = 3;
    pub const SEGMENT: u8 = 4;
    pub const LIST_MANIFESTS: u8 = 5;
    pub const MANIFEST_LIST: u8 = 6;
}

#[derive(Serialize, Deserialize)]
struct GetManifestHeader {
    /// Manifest key (payload digest); `None` asks for the newest.
    key: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct ManifestHeader {
    found: bool,
    manifest: Option<HandoffManifest>,
}

#[derive(Serialize, Deserialize)]
struct GetSegmentHeader {
    digest: String,
}

#[derive(Serialize, Deserialize)]
struct SegmentReplyHeader {
    found: bool,
    digest: String,
}

#[derive(Serialize, Deserialize)]
struct ManifestListHeader {
    keys: Vec<String>,
}

/// Statistics for one `fetch_into_store` pull.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FetchStats {
    pub segments_fetched: usize,
    pub segments_skipped: usize,
    pub bytes_fetched: u64,
}

/// Serve the store to one connected peer until it disconnects.
pub fn serve_connection(store: &HandoffSegmentStore, stream: TcpStream) -> Result<()> {
    stream.set_nodelay(true).ok();
    let mut reader = BufReader::with_capacity(STREAM_BUFFER_BYTES, stream.try_clone()?);
    let mut writer = BufWriter::with_capacity(STREAM_BUFFER_BYTES, stream);
    loop {
        let (kind, header, _) = match read_frame(&mut reader) {
            Ok(frame) => frame,
            // A closed connection between requests is a normal end.
            Err(_) => return Ok(()),
        };
        match kind {
            frame_kind::GET_MANIFEST => {
                let request: GetManifestHeader =
                    serde_json::from_value(header).context("malformed manifest request")?;
                let key = match request.key {
                    Some(key) => Some(key),
                    None => store.list_manifests()?.into_iter().next(),
                };
                let manifest = key.and_then(|key| store.load_manifest(&key).ok());
                write_frame(
                    &mut writer,
                    frame_kind::MANIFEST,
                    &ManifestHeader {
                        found: manifest.is_some(),
                        manifest,
                    },
                    &[],
                )?;
            }
            frame_kind::GET_SEGMENT => {
                let request: GetSegmentHeader =
                    serde_json::from_value(header).context("malformed segment request")?;
                match store.read_segment(&request.digest) {
                    Ok(bytes) => write_frame(
                        &mut writer,
                        frame_kind::SEGMENT,
                        &SegmentReplyHeader {
                            found: true,
                            digest: request.digest,
                        },
                        &bytes,
                    )?,
                    Err(_) => write_frame(
                        &mut writer,
                        frame_kind::SEGMENT,
                        &SegmentReplyHeader {
                            found: false,
                            digest: request.digest,
                        },
                        &[],
                    )?,
                }
            }
            frame_kind::LIST_MANIFESTS => {
                write_frame(
                    &mut writer,
                    frame_kind::MANIFEST_LIST,
                    &ManifestListHeader {
                        keys: store.list_manifests()?,
                    },
                    &[],
                )?;
            }
            other => bail!("unexpected skippy-kv frame kind {other}"),
        }
        writer.flush().context("failed to flush skippy-kv reply")?;
    }
}

/// Serve the store on a listener for `accept_count` connections
/// (0 = until the process dies).
pub fn serve_store(
    store: &HandoffSegmentStore,
    listener: &TcpListener,
    accept_count: usize,
) -> Result<()> {
    let mut served = 0usize;
    loop {
        let (stream, _) = listener.accept().context("skippy-kv accept failed")?;
        served += 1;
        if let Err(error) = serve_connection(store, stream) {
            eprintln!("skippy-kv connection failed: {error:#}");
        }
        if accept_count != 0 && served >= accept_count {
            return Ok(());
        }
    }
}

/// A client connection to a peer's store.
pub struct KvFetchClient {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
}

impl KvFetchClient {
    pub fn connect(peer: &str) -> Result<Self> {
        let stream = TcpStream::connect(peer)
            .with_context(|| format!("failed to connect to skippy-kv peer {peer}"))?;
        stream.set_nodelay(true).ok();
        Ok(Self {
            reader: BufReader::with_capacity(STREAM_BUFFER_BYTES, stream.try_clone()?),
            writer: BufWriter::with_capacity(STREAM_BUFFER_BYTES, stream),
        })
    }

    pub fn list_manifests(&mut self) -> Result<Vec<String>> {
        write_frame(
            &mut self.writer,
            frame_kind::LIST_MANIFESTS,
            &serde_json::json!({}),
            &[],
        )?;
        self.writer.flush()?;
        let (header, _) =
            read_frame_expect::<ManifestListHeader>(&mut self.reader, frame_kind::MANIFEST_LIST)?;
        Ok(header.keys)
    }

    pub fn fetch_manifest(&mut self, key: Option<&str>) -> Result<HandoffManifest> {
        write_frame(
            &mut self.writer,
            frame_kind::GET_MANIFEST,
            &GetManifestHeader {
                key: key.map(str::to_string),
            },
            &[],
        )?;
        self.writer.flush()?;
        let (header, _) =
            read_frame_expect::<ManifestHeader>(&mut self.reader, frame_kind::MANIFEST)?;
        header.manifest.context("peer has no matching manifest")
    }

    pub fn fetch_segment(&mut self, digest: &str) -> Result<Vec<u8>> {
        write_frame(
            &mut self.writer,
            frame_kind::GET_SEGMENT,
            &GetSegmentHeader {
                digest: digest.to_string(),
            },
            &[],
        )?;
        self.writer.flush()?;
        let (header, bytes) =
            read_frame_expect::<SegmentReplyHeader>(&mut self.reader, frame_kind::SEGMENT)?;
        if !header.found {
            bail!("peer does not hold segment {digest}");
        }
        if segment_digest(&bytes) != digest {
            bail!("segment {digest} from peer failed digest verification");
        }
        Ok(bytes)
    }

    /// Pull one manifest and every segment the local store is missing, then
    /// commit the manifest locally. Content addressing makes this
    /// idempotent: re-fetching an already-held manifest transfers nothing.
    pub fn fetch_into_store(
        &mut self,
        key: Option<&str>,
        store: &HandoffSegmentStore,
    ) -> Result<(HandoffManifest, FetchStats)> {
        let manifest = self.fetch_manifest(key)?;
        let mut stats = FetchStats::default();
        for segment in &manifest.segments {
            if store.has_segment(&segment.digest) {
                stats.segments_skipped += 1;
                continue;
            }
            let bytes = self.fetch_segment(&segment.digest)?;
            store.put_segment(&bytes)?;
            stats.segments_fetched += 1;
            stats.bytes_fetched += bytes.len() as u64;
        }
        store
            .commit(&manifest)
            .context("failed to commit fetched manifest")?;
        Ok((manifest, stats))
    }
}

fn write_frame(
    writer: &mut impl Write,
    kind: u8,
    header: &impl Serialize,
    payload: &[u8],
) -> Result<()> {
    let header_bytes = serde_json::to_vec(header).context("failed to encode frame header")?;
    if header_bytes.len() as u64 > MAX_HEADER_BYTES {
        bail!("frame header of {} bytes exceeds limit", header_bytes.len());
    }
    writer.write_all(&[kind])?;
    writer.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&header_bytes)?;
    writer.write_all(&(payload.len() as u64).to_le_bytes())?;
    writer.write_all(payload)?;
    Ok(())
}

fn read_frame(reader: &mut impl Read) -> Result<(u8, serde_json::Value, Vec<u8>)> {
    let mut kind = [0u8; 1];
    reader
        .read_exact(&mut kind)
        .context("skippy-kv stream closed")?;
    let mut header_len = [0u8; 4];
    reader.read_exact(&mut header_len)?;
    let header_len = u32::from_le_bytes(header_len) as u64;
    if header_len > MAX_HEADER_BYTES {
        bail!("frame header of {header_len} bytes exceeds limit");
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    reader.read_exact(&mut header_bytes)?;
    let header = serde_json::from_slice(&header_bytes).context("malformed frame header")?;
    let mut payload_len = [0u8; 8];
    reader.read_exact(&mut payload_len)?;
    let payload_len = u64::from_le_bytes(payload_len);
    if payload_len > MAX_SEGMENT_BYTES {
        bail!("frame payload of {payload_len} bytes exceeds limit");
    }
    let mut payload = vec![0u8; payload_len as usize];
    reader.read_exact(&mut payload)?;
    Ok((kind[0], header, payload))
}

fn read_frame_expect<T: DeserializeOwned>(
    reader: &mut impl Read,
    expected_kind: u8,
) -> Result<(T, Vec<u8>)> {
    let (kind, header, payload) = read_frame(reader)?;
    if kind != expected_kind {
        bail!("expected skippy-kv frame kind {expected_kind}, got {kind}");
    }
    Ok((
        serde_json::from_value(header).context("malformed frame header for expected kind")?,
        payload,
    ))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::l3::{HandoffManifest, HandoffSegmentRef};

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir()
            .join("skippy-kv-tests")
            .join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    fn seeded_store(root: &PathBuf, payload: &[u8]) -> (HandoffSegmentStore, HandoffManifest) {
        let store = HandoffSegmentStore::open(root, 0).expect("open store");
        let mut manifest =
            HandoffManifest::new("blake3:test-identity".to_string(), "full-state".into());
        for (index, chunk) in payload.chunks(1024).enumerate() {
            let (digest, _) = store.put_segment(chunk).expect("put");
            manifest.segments.push(HandoffSegmentRef {
                index: index as u32,
                offset: (index * 1024) as u64,
                bytes: chunk.len() as u64,
                digest,
                meta_json: None,
            });
        }
        manifest.total_bytes = payload.len() as u64;
        manifest.payload_digest = segment_digest(payload);
        store.commit(&manifest).expect("commit");
        (store, manifest)
    }

    #[test]
    fn fetch_into_store_pulls_verifies_and_is_idempotent() {
        let payload: Vec<u8> = (0..10_000u32).map(|value| (value % 251) as u8).collect();
        let server_root = temp_root("server");
        let client_root = temp_root("client");
        let (server_store, manifest) = seeded_store(&server_root, &payload);
        let client_store = HandoffSegmentStore::open(&client_root, 0).expect("open client");

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr").to_string();
        let server = std::thread::spawn(move || {
            serve_store(&server_store, &listener, 2).expect("serve");
        });

        let mut client = KvFetchClient::connect(&address).expect("connect");
        assert_eq!(
            client.list_manifests().expect("list"),
            vec![manifest.payload_digest.clone()]
        );
        let (fetched, stats) = client
            .fetch_into_store(None, &client_store)
            .expect("first fetch");
        assert_eq!(fetched.payload_digest, manifest.payload_digest);
        assert_eq!(stats.segments_fetched, manifest.segments.len());
        assert_eq!(stats.segments_skipped, 0);
        assert_eq!(client_store.assemble(&fetched).expect("assemble"), payload);
        drop(client);

        // Second pull on a fresh connection: everything present, nothing moves.
        let mut client = KvFetchClient::connect(&address).expect("reconnect");
        let (_, stats) = client
            .fetch_into_store(Some(&manifest.payload_digest), &client_store)
            .expect("second fetch");
        assert_eq!(stats.segments_fetched, 0);
        assert_eq!(stats.bytes_fetched, 0);
        assert_eq!(stats.segments_skipped, manifest.segments.len());
        drop(client);
        server.join().expect("server thread");
    }

    #[test]
    fn missing_segments_and_manifests_are_reported() {
        let server_root = temp_root("missing");
        let client_root = temp_root("missing-client");
        let payload = vec![7u8; 2048];
        let (server_store, _) = seeded_store(&server_root, &payload);
        let _client_store = HandoffSegmentStore::open(&client_root, 0).expect("open client");

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr").to_string();
        let server = std::thread::spawn(move || {
            serve_store(&server_store, &listener, 1).expect("serve");
        });

        let mut client = KvFetchClient::connect(&address).expect("connect");
        assert!(client.fetch_manifest(Some("no-such-key")).is_err());
        assert!(client.fetch_segment(&segment_digest(b"absent")).is_err());
        drop(client);
        server.join().expect("server thread");
    }
}

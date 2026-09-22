//! Versioned role setup precedes all message traffic, including return streams.
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::{ActivationAgreement, ActivationDimension, ActivationProfile, StageMessageIo};
use crate::{STAGE_PROTOCOL_GENERATION, StageConfig, StageTopology};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub type StageStream = StageMessageIo<std::net::TcpStream>;
const SETUP_MAGIC: u32 = 0x53545053;
const MAX_SETUP_BYTES: usize = 65536;
const POLL: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionRole {
    Activation,
    TokensAndControl,
    PredictionReturn,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Peer {
    run: String,
    topology: String,
    model: String,
    stage: String,
    index: u32,
    layer_start: u32,
    layer_end: u32,
}
impl From<&StageConfig> for Peer {
    fn from(c: &StageConfig) -> Self {
        Self {
            run: c.run_id.clone(),
            topology: c.topology_id.clone(),
            model: c.model_id.clone(),
            stage: c.stage_id.clone(),
            index: c.stage_index,
            layer_start: c.layer_start,
            layer_end: c.layer_end,
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Hello {
    role: ConnectionRole,
    sender: Option<Peer>,
    target: Option<(String, u32)>,
    output: Option<ActivationProfile>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Offer {
    role: ConnectionRole,
    receiver: Option<Peer>,
    agreement: ActivationAgreement,
}

fn compatible_input(offered: &ActivationProfile, required: &ActivationProfile) -> Result<()> {
    if offered.max_tokens > required.max_tokens || offered.max_sequences > required.max_sequences {
        bail!("offered activation counts exceed receiver limits");
    }
    if offered.frontier_identity != required.frontier_identity {
        bail!("activation frontier identity mismatch");
    }
    for expected in &required.parts {
        let Some(actual) = offered
            .parts
            .iter()
            .find(|p| p.identity == expected.identity)
        else {
            if expected.optional {
                continue;
            }
            bail!("required input binding missing from setup");
        };
        if actual.ggml_type != expected.ggml_type
            || actual.rank != expected.rank
            || actual.token_axis != expected.token_axis
            || (actual.optional && !expected.optional)
        {
            bail!("input binding type, rank, token axis or presence mismatch");
        }
        for (actual, expected) in actual.dimensions.iter().zip(&expected.dimensions) {
            let compatible = match expected {
                ActivationDimension::Fixed(value) => actual == &ActivationDimension::Fixed(*value),
                ActivationDimension::Tokens => actual == &ActivationDimension::Tokens,
                ActivationDimension::Dynamic { min, max } => match actual {
                    ActivationDimension::Fixed(value) => value >= min && value <= max,
                    ActivationDimension::Dynamic { min: lo, max: hi } => lo >= min && hi <= max,
                    ActivationDimension::Tokens => false,
                },
            };
            if !compatible {
                bail!("input dimension or dynamic bounds mismatch");
            }
        }
    }
    if offered
        .parts
        .iter()
        .any(|p| !p.optional && !required.parts.iter().any(|r| r.identity == p.identity))
    {
        bail!("unrecognized required activation binding");
    }
    Ok(())
}

pub fn client_setup(
    stream: &mut StageStream,
    role: ConnectionRole,
    config: Option<&StageConfig>,
    output: Option<ActivationProfile>,
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<()> {
    let hello = Hello {
        role,
        sender: config.map(Peer::from),
        target: config.and_then(|c| {
            c.downstream
                .as_ref()
                .map(|p| (p.stage_id.clone(), p.stage_index))
        }),
        output,
    };
    let result = (|| {
        write_packet(stream, &hello, deadline, shutdown)?;
        let offer: Offer = read_packet(stream, deadline, shutdown)?;
        if offer.role != role {
            bail!("setup role confirmation mismatch");
        }
        offer.agreement.validate()?;
        let mut expected = hello.output.iter().cloned().collect::<Vec<_>>();
        // The receiver may narrow counts, but may not alter the realized layout.
        if let (Some(source), [accepted]) =
            (expected.first_mut(), offer.agreement.profiles.as_slice())
        {
            if accepted.max_tokens > source.max_tokens
                || accepted.max_sequences > source.max_sequences
            {
                bail!("peer widened the offered count limits");
            }
            source.max_tokens = accepted.max_tokens;
            source.max_sequences = accepted.max_sequences;
        }
        if offer.agreement.profiles != expected {
            bail!("peer changed the offered output contract");
        }
        if let Some(config) = config {
            let receiver = offer
                .receiver
                .as_ref()
                .context("stage peer identity absent")?;
            let target = config
                .downstream
                .as_ref()
                .context("downstream identity absent")?;
            if receiver.run != config.run_id
                || receiver.topology != config.topology_id
                || receiver.model != config.model_id
                || receiver.stage != target.stage_id
                || receiver.index != target.stage_index
                || receiver.layer_start != config.layer_end
            {
                bail!("admitted downstream relationship mismatch");
            }
        }
        write_packet(stream, &offer.agreement.generation, deadline, shutdown)?;
        let ready: [u8; 16] = read_packet(stream, deadline, shutdown)?;
        if ready != offer.agreement.generation {
            bail!("setup confirmation generation mismatch");
        }
        stream.establish(offer.agreement)?;
        Ok(())
    })();
    clear_timeouts(stream, result)
}

pub fn server_setup(
    stream: &mut StageStream,
    config: Option<&StageConfig>,
    topology: Option<&StageTopology>,
    input: Option<&ActivationProfile>,
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<ConnectionRole> {
    server_setup_inner(
        stream,
        config,
        topology,
        input,
        deadline,
        shutdown,
        false,
        |_, _| Ok(()),
    )
}

/// Complete dependent outgoing agreements after upstream acceptance and before
/// confirming readiness. The callback must use the same absolute deadline.
pub fn server_setup_with_prepare(
    stream: &mut StageStream,
    config: &StageConfig,
    topology: Option<&StageTopology>,
    input: Option<&ActivationProfile>,
    deadline: Instant,
    shutdown: &AtomicBool,
    prepare: impl FnOnce(ConnectionRole, &ActivationAgreement) -> Result<()>,
) -> Result<ConnectionRole> {
    server_setup_inner(
        stream,
        Some(config),
        topology,
        input,
        deadline,
        shutdown,
        false,
        prepare,
    )
}

/// Diagnostic sinks record bounded typed payloads but never execute them. They
/// validate peer/plan identity and accept the producer's full advertised vocabulary.
pub fn recording_setup(
    stream: &mut StageStream,
    config: &StageConfig,
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<ConnectionRole> {
    server_setup_inner(
        stream,
        Some(config),
        None,
        None,
        deadline,
        shutdown,
        true,
        |_, _| Ok(()),
    )
}

#[allow(clippy::too_many_arguments)]
fn server_setup_inner(
    stream: &mut StageStream,
    config: Option<&StageConfig>,
    topology: Option<&StageTopology>,
    input: Option<&ActivationProfile>,
    deadline: Instant,
    shutdown: &AtomicBool,
    recording: bool,
    prepare: impl FnOnce(ConnectionRole, &ActivationAgreement) -> Result<()>,
) -> Result<ConnectionRole> {
    let result = (|| {
        let mut hello: Hello = read_packet(stream, deadline, shutdown)?;
        match hello.role {
            ConnectionRole::Activation => {
                let config = config.context("activation role on a return listener")?;
                let sender = hello
                    .sender
                    .as_ref()
                    .context("activation sender identity missing")?;
                let output = hello
                    .output
                    .as_mut()
                    .context("activation output contract missing")?;
                if sender.run != config.run_id
                    || sender.topology != config.topology_id
                    || sender.model != config.model_id
                    || hello.target != Some((config.stage_id.clone(), config.stage_index))
                    || sender.index.checked_add(1) != Some(config.stage_index)
                    || sender.layer_end != config.layer_start
                    || output.producer_stage_index != sender.index as i32
                    || output.layer_start != sender.layer_start as i32
                    || output.layer_end != sender.layer_end as i32
                {
                    bail!("admitted upstream relationship mismatch");
                }
                if let Some(expected) = config.upstream.as_ref()
                    && (expected.stage_id != sender.stage || expected.stage_index != sender.index)
                {
                    bail!("upstream identity differs from configured peer");
                }
                if let Some(topology) = topology {
                    let expected = topology
                        .stages
                        .iter()
                        .find(|p| p.stage_index == sender.index)
                        .context("upstream stage absent from plan")?;
                    if expected.stage_id != sender.stage
                        || expected.layer_start != sender.layer_start
                        || expected.layer_end != sender.layer_end
                    {
                        bail!("upstream identity differs from plan");
                    }
                }
                if !recording {
                    let required = input.context("receiver has no input boundary")?;
                    output.max_tokens = output.max_tokens.min(required.max_tokens);
                    output.max_sequences = output.max_sequences.min(required.max_sequences);
                    compatible_input(output, required)?;
                }
            }
            ConnectionRole::TokensAndControl => {
                if config.is_none() || hello.output.is_some() {
                    bail!("token/control ingress requires a stage and no activation table");
                }
            }
            ConnectionRole::PredictionReturn => {
                if hello.output.is_some() {
                    bail!("return connection cannot admit activations");
                }
            }
        }
        let agreement = ActivationAgreement {
            generation: *uuid::Uuid::new_v4().as_bytes(),
            profiles: hello.output.into_iter().collect(),
        };
        agreement.validate()?;
        write_packet(
            stream,
            &Offer {
                role: hello.role,
                receiver: config.map(Peer::from),
                agreement: agreement.clone(),
            },
            deadline,
            shutdown,
        )?;
        let accepted: [u8; 16] = read_packet(stream, deadline, shutdown)?;
        if accepted != agreement.generation {
            bail!("setup acceptance generation mismatch");
        }
        prepare(hello.role, &agreement)?;
        // Install before confirming readiness; no worker or clone is handed an unadmitted table.
        stream.establish(agreement)?;
        write_packet(stream, &accepted, deadline, shutdown)?;
        Ok(hello.role)
    })();
    clear_timeouts(stream, result)
}

fn clear_timeouts<T>(stream: &StageStream, result: Result<T>) -> Result<T> {
    stream
        .set_read_timeout(None)
        .context("clear setup read timeout")?;
    stream
        .set_write_timeout(None)
        .context("clear setup write timeout")?;
    result
}
fn timeout(stream: &StageStream, deadline: Instant, shutdown: &AtomicBool) -> Result<()> {
    if shutdown.load(Ordering::Acquire) {
        bail!("stage setup cancelled");
    }
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        bail!("stage setup deadline expired");
    }
    stream.set_read_timeout(Some(remaining.min(POLL)))?;
    stream.set_write_timeout(Some(remaining.min(POLL)))?;
    Ok(())
}
fn read_exact(
    stream: &mut StageStream,
    mut bytes: &mut [u8],
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<()> {
    while !bytes.is_empty() {
        timeout(stream, deadline, shutdown)?;
        match stream.read(bytes) {
            Ok(0) => bail!("stage peer closed during setup"),
            Ok(n) => bytes = &mut bytes[n..],
            Err(e) if retry(&e) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn write_all(
    stream: &mut StageStream,
    mut bytes: &[u8],
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<()> {
    while !bytes.is_empty() {
        timeout(stream, deadline, shutdown)?;
        match stream.write(bytes) {
            Ok(0) => bail!("stage peer closed during setup"),
            Ok(n) => bytes = &bytes[n..],
            Err(e) if retry(&e) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn retry(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}
fn write_packet(
    stream: &mut StageStream,
    value: &impl Serialize,
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > MAX_SETUP_BYTES {
        bail!("stage setup message exceeds limit");
    }
    write_all(stream, &SETUP_MAGIC.to_le_bytes(), deadline, shutdown)?;
    write_all(
        stream,
        &STAGE_PROTOCOL_GENERATION.to_le_bytes(),
        deadline,
        shutdown,
    )?;
    write_all(
        stream,
        &(bytes.len() as u32).to_le_bytes(),
        deadline,
        shutdown,
    )?;
    write_all(stream, &bytes, deadline, shutdown)
}
fn read_packet<T: DeserializeOwned>(
    stream: &mut StageStream,
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<T> {
    let mut header = [0; 12];
    read_exact(stream, &mut header, deadline, shutdown)?;
    if u32::from_le_bytes(header[..4].try_into()?) != SETUP_MAGIC
        || u32::from_le_bytes(header[4..8].try_into()?) != STAGE_PROTOCOL_GENERATION
    {
        bail!("incompatible stage setup protocol");
    }
    let len = u32::from_le_bytes(header[8..].try_into()?) as usize;
    if len > MAX_SETUP_BYTES {
        bail!("stage setup message exceeds limit");
    }
    let mut bytes = vec![0; len];
    read_exact(stream, &mut bytes, deadline, shutdown)?;
    serde_json::from_slice(&bytes).context("decode stage setup")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::{ActivationPartProfile, StageMessageContext};
    fn config(index: u32) -> StageConfig {
        serde_json::from_value(serde_json::json!({
            "run_id":"r", "topology_id":"p", "model_id":"m", "stage_id":format!("s{index}"),
            "stage_index":index, "layer_start":index*4, "layer_end":(index+1)*4,
            "execution_contract":"test", "load_mode":"runtime-slice", "bind_addr":"127.0.0.1:0",
            "downstream": if index==0 { serde_json::json!({"stage_id":"s1","stage_index":1,"endpoint":"unused"}) } else { serde_json::Value::Null }
        })).unwrap()
    }
    fn profile() -> ActivationProfile {
        ActivationProfile {
            id: 1,
            producer_stage_index: 0,
            layer_start: 0,
            layer_end: 4,
            frontier_identity: [3; 32],
            max_tokens: 32,
            max_sequences: 8,
            parts: vec![ActivationPartProfile {
                identity: [1; 32],
                ggml_type: 0,
                rank: 2,
                token_axis: 1,
                optional: false,
                dimensions: [
                    ActivationDimension::Fixed(4),
                    ActivationDimension::Tokens,
                    ActivationDimension::Fixed(1),
                    ActivationDimension::Fixed(1),
                ],
            }],
        }
    }
    #[test]
    fn setup_installs_shared_state_and_reconnect_has_a_new_generation() {
        let mut generations = Vec::new();
        for _ in 0..2 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let mut stream = StageStream::new(listener.accept().unwrap().0);
                let role = server_setup(
                    &mut stream,
                    Some(&config(1)),
                    None,
                    Some(&profile()),
                    Instant::now() + Duration::from_secs(2),
                    &AtomicBool::new(false),
                )
                .unwrap();
                assert_eq!(role, ConnectionRole::Activation);
                let clone = stream.try_clone().unwrap();
                assert!(std::ptr::eq(
                    stream.activation_agreement().unwrap(),
                    clone.activation_agreement().unwrap()
                ));
                let message = crate::binary::read_stage_message(&clone, 0).unwrap();
                assert_eq!(
                    message.activation_frame().unwrap().unwrap().payload,
                    vec![0; 32]
                );
                stream.activation_agreement().unwrap().clone()
            });
            let mut client = StageStream::connect(addr).unwrap();
            client_setup(
                &mut client,
                ConnectionRole::Activation,
                Some(&config(0)),
                Some(profile()),
                Instant::now() + Duration::from_secs(2),
                &AtomicBool::new(false),
            )
            .unwrap();
            let frame = crate::binary::StageActivationFrame {
                desc: crate::binary::StageActivationDesc {
                    version: crate::binary::STAGE_ACTIVATION_FRAME_VERSION,
                    producer_stage_index: 0,
                    layer_start: 0,
                    layer_end: 4,
                    token_count: 2,
                    sequence_count: 1,
                    payload_bytes: 32,
                    frontier_identity: [3; 32],
                    parts: vec![crate::binary::StageActivationPartDesc {
                        identity: [1; 32],
                        ggml_type: 0,
                        rank: 2,
                        token_axis: 1,
                        flags: 0,
                        dimensions: [4, 2, 1, 1],
                        byte_strides: [4, 16, 32, 32],
                        payload_offset: 0,
                        payload_bytes: 32,
                    }],
                },
                payload: vec![0; 32],
            };
            let mut message = crate::binary::StageWireMessage::stop();
            message.kind = crate::binary::WireMessageKind::DecodeEmbd;
            message.state = crate::binary::StageStateHeader::new(message.kind);
            message.state.source_stage_index = 0;
            message.token_count = 2;
            message.activation = crate::binary::encode_raw_activation_frame(&frame).unwrap();
            let mut forwarder = client.try_clone().unwrap();
            crate::binary::write_stage_message(&mut forwarder, &message).unwrap();
            let agreement = server.join().unwrap();
            assert_eq!(client.activation_agreement(), Some(&agreement));
            assert!(client.establish(agreement.clone()).is_err());
            generations.push(agreement.generation);
        }
        assert_ne!(generations[0], generations[1]);
    }
    #[test]
    fn prediction_return_role_works_on_dedicated_and_shared_listeners() {
        for shared in [false, true] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let mut stream = StageStream::new(listener.accept().unwrap().0);
                let config = shared.then(|| config(1));
                let role = server_setup(
                    &mut stream,
                    config.as_ref(),
                    None,
                    None,
                    Instant::now() + Duration::from_secs(2),
                    &AtomicBool::new(false),
                )
                .unwrap();
                assert_eq!(role, ConnectionRole::PredictionReturn);
                assert!(stream.activation_agreement().unwrap().profiles.is_empty());
            });
            let mut client = StageStream::connect(addr).unwrap();
            client_setup(
                &mut client,
                ConnectionRole::PredictionReturn,
                None,
                None,
                Instant::now() + Duration::from_secs(2),
                &AtomicBool::new(false),
            )
            .unwrap();
            server.join().unwrap();
        }
    }
    #[test]
    fn malformed_version_and_oversized_setup_rejected_before_body() {
        for (version, length) in [
            (STAGE_PROTOCOL_GENERATION - 1, 0u32),
            (STAGE_PROTOCOL_GENERATION, MAX_SETUP_BYTES as u32 + 1),
        ] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let mut stream = StageStream::new(listener.accept().unwrap().0);
                let error = server_setup(
                    &mut stream,
                    None,
                    None,
                    None,
                    Instant::now() + Duration::from_secs(2),
                    &AtomicBool::new(false),
                )
                .unwrap_err();
                assert!(!error.to_string().contains("deadline"));
                assert!(stream.activation_agreement().is_none());
            });
            let mut client = StageStream::connect(addr).unwrap();
            // Fragment the fixed header to exercise partial reads.
            let header = [
                SETUP_MAGIC.to_le_bytes(),
                version.to_le_bytes(),
                length.to_le_bytes(),
            ]
            .concat();
            for byte in header {
                client.write_all(&[byte]).unwrap();
            }
            server.join().unwrap();
        }
    }
    #[test]
    fn cancellation_interrupts_a_partial_setup() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let task_stop = stop.clone();
        let server = std::thread::spawn(move || {
            let mut stream = StageStream::new(listener.accept().unwrap().0);
            let start = Instant::now();
            let error = server_setup(
                &mut stream,
                None,
                None,
                None,
                start + Duration::from_secs(10),
                &task_stop,
            )
            .unwrap_err();
            assert!(error.to_string().contains("cancelled"));
            assert!(start.elapsed() < Duration::from_secs(1));
        });
        let mut client = StageStream::connect(addr).unwrap();
        client.write_all(&[0x53]).unwrap();
        stop.store(true, Ordering::Release);
        server.join().unwrap();
    }
    #[test]
    fn setup_rejects_wrong_plan_peer_and_output_contract_before_admission() {
        for fault in 0..5 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let mut stream = StageStream::new(listener.accept().unwrap().0);
                let result = server_setup(
                    &mut stream,
                    Some(&config(1)),
                    None,
                    Some(&profile()),
                    Instant::now() + Duration::from_secs(2),
                    &AtomicBool::new(false),
                );
                assert!(result.is_err());
                assert!(stream.activation_agreement().is_none());
            });
            let mut client = StageStream::connect(addr).unwrap();
            let mut source = config(0);
            let mut output = profile();
            match fault {
                0 => source.run_id = "other-run".into(),
                1 => source.model_id = "other-model".into(),
                2 => source.downstream.as_mut().unwrap().stage_id = "other-peer".into(),
                3 => output.parts[0].dimensions[0] = ActivationDimension::Fixed(8),
                _ => output.parts[0].optional = true,
            }
            assert!(
                client_setup(
                    &mut client,
                    ConnectionRole::Activation,
                    Some(&source),
                    Some(output),
                    Instant::now() + Duration::from_secs(2),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
            assert!(client.activation_agreement().is_none());
            server.join().unwrap();
        }
    }

    #[test]
    fn sender_rejects_a_receiver_that_changes_its_output_contract() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let mut stream = StageStream::new(listener.accept().unwrap().0);
            let deadline = Instant::now() + Duration::from_secs(2);
            let stop = AtomicBool::new(false);
            let hello: Hello = read_packet(&mut stream, deadline, &stop).unwrap();
            let mut profile = hello.output.unwrap();
            profile.parts[0].dimensions[0] = ActivationDimension::Fixed(8);
            write_packet(
                &mut stream,
                &Offer {
                    role: ConnectionRole::Activation,
                    receiver: Some(Peer::from(&config(1))),
                    agreement: ActivationAgreement {
                        generation: [5; 16],
                        profiles: vec![profile],
                    },
                },
                deadline,
                &stop,
            )
            .unwrap();
        });
        let mut client = StageStream::connect(addr).unwrap();
        let error = client_setup(
            &mut client,
            ConnectionRole::Activation,
            Some(&config(0)),
            Some(profile()),
            Instant::now() + Duration::from_secs(2),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("changed the offered output contract")
        );
        assert!(client.activation_agreement().is_none());
        server.join().unwrap();
    }

    #[test]
    fn receiver_narrows_sender_counts_and_both_install_the_intersection() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let mut stream = StageStream::new(listener.accept().unwrap().0);
            let mut small = profile();
            small.max_tokens = 3;
            small.max_sequences = 1;
            assert!(compatible_input(&profile(), &small).is_err());
            server_setup(
                &mut stream,
                Some(&config(1)),
                None,
                Some(&small),
                Instant::now() + Duration::from_secs(2),
                &AtomicBool::new(false),
            )
            .unwrap();
            stream.activation_agreement().unwrap().clone()
        });
        let mut client = StageStream::connect(addr).unwrap();
        client_setup(
            &mut client,
            ConnectionRole::Activation,
            Some(&config(0)),
            Some(profile()),
            Instant::now() + Duration::from_secs(2),
            &AtomicBool::new(false),
        )
        .unwrap();
        let agreement = server.join().unwrap();
        assert_eq!(client.activation_agreement(), Some(&agreement));
        assert_eq!(agreement.profiles[0].max_tokens, 3);
        assert_eq!(agreement.profiles[0].max_sequences, 1);
    }

    #[test]
    fn chain_agrees_outgoing_edge_before_upstream_ready_without_application_traffic() {
        let middle_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let middle_addr = middle_listener.local_addr().unwrap();
        let last_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let last_addr = last_listener.local_addr().unwrap();
        let ready = std::sync::Arc::new(AtomicBool::new(false));
        let last_ready = ready.clone();
        let last = std::thread::spawn(move || {
            let mut stream = StageStream::new(last_listener.accept().unwrap().0);
            let mut input = profile();
            input.producer_stage_index = 1;
            input.layer_start = 4;
            input.layer_end = 8;
            server_setup_with_prepare(
                &mut stream,
                &config(2),
                None,
                Some(&input),
                Instant::now() + Duration::from_secs(2),
                &AtomicBool::new(false),
                |_, _| {
                    last_ready.store(true, Ordering::Release);
                    Ok(())
                },
            )
            .unwrap();
        });
        let middle = std::thread::spawn(move || {
            let mut stream = StageStream::new(middle_listener.accept().unwrap().0);
            let deadline = Instant::now() + Duration::from_secs(2);
            let stop = AtomicBool::new(false);
            let mut middle_config = config(1);
            middle_config.downstream = Some(crate::PeerConfig {
                stage_id: "s2".into(),
                stage_index: 2,
                endpoint: last_addr.to_string(),
            });
            let mut outgoing = None;
            server_setup_with_prepare(
                &mut stream,
                &middle_config,
                None,
                Some(&profile()),
                deadline,
                &stop,
                |_, accepted| {
                    let mut output = accepted.profiles[0].clone();
                    output.producer_stage_index = 1;
                    output.layer_start = 4;
                    output.layer_end = 8;
                    let mut downstream = StageStream::connect(last_addr)?;
                    client_setup(
                        &mut downstream,
                        ConnectionRole::Activation,
                        Some(&middle_config),
                        Some(output),
                        deadline,
                        &stop,
                    )?;
                    outgoing = Some(downstream);
                    Ok(())
                },
            )
            .unwrap();
            assert!(outgoing.unwrap().activation_agreement().is_some());
        });
        let mut first = StageStream::connect(middle_addr).unwrap();
        client_setup(
            &mut first,
            ConnectionRole::Activation,
            Some(&config(0)),
            Some(profile()),
            Instant::now() + Duration::from_secs(2),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(ready.load(Ordering::Acquire));
        middle.join().unwrap();
        last.join().unwrap();
    }

    #[test]
    fn required_input_and_dynamic_bounds_are_checked() {
        let expected = profile();
        let mut bad = expected.clone();
        bad.parts[0].ggml_type = 1;
        assert!(compatible_input(&bad, &expected).is_err());
        let mut bad = expected.clone();
        bad.parts[0].optional = true;
        assert!(compatible_input(&bad, &expected).is_err());
        let mut expected = expected;
        expected.parts[0].dimensions[0] = ActivationDimension::Dynamic { min: 2, max: 4 };
        let mut bad = expected.clone();
        bad.parts[0].dimensions[0] = ActivationDimension::Dynamic { min: 1, max: 8 };
        assert!(compatible_input(&bad, &expected).is_err());
    }
}

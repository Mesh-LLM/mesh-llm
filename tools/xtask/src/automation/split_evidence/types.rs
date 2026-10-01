use super::Error;
use crate::automation::codepoint_json::{strings::JsonString, value::Value};
use num_bigint::BigUint;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct Text(pub(super) JsonString);

impl Text {
    pub(super) fn starts_with(&self, prefix: &Self) -> bool {
        let mut codes = self.0.codepoints();
        prefix.0.codepoints().all(|code| codes.next() == Some(code))
    }

    pub(super) fn display(&self) -> String {
        self.0
            .codepoints()
            .map(|code| char::from_u32(code).unwrap_or('\u{fffd}'))
            .collect()
    }

    pub(super) fn value(&self) -> Value {
        Value::Str(self.0.clone())
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Integer(pub(super) BigUint);

impl Integer {
    pub(super) fn parse(value: Option<&Value>, label: &str) -> Result<Self, Error> {
        let parsed = match value {
            Some(Value::Int(integer)) => u128::try_from(*integer).ok().map(BigUint::from),
            Some(Value::BigInt(decimal)) => BigUint::parse_bytes(decimal.as_bytes(), 10),
            _ => None,
        };
        parsed
            .map(Self)
            .ok_or_else(|| Error::Contract(format!("{label} must be a non-negative integer")))
    }

    pub(super) fn equals(&self, small: u32) -> bool {
        self.0 == BigUint::from(small)
    }
    pub(super) fn value(&self) -> Value {
        Value::BigInt(self.0.to_string())
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct Identity {
    pub(super) topology_id: Text,
    pub(super) run_id: Text,
    pub(super) model_id: Text,
    pub(super) package_ref: Text,
    pub(super) manifest_sha256: Text,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct Stage {
    pub(super) stage_id: Text,
    pub(super) stage_index: Integer,
    pub(super) node_id: Text,
    pub(super) layer_start: Integer,
    pub(super) layer_end: Integer,
    pub(super) bind_addr: Text,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct Topology {
    pub(super) identity: Identity,
    pub(super) stages: [Stage; 2],
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct Status {
    pub(super) identity: Identity,
    pub(super) stage: Stage,
    pub(super) state: Text,
}

pub(super) struct Observer {
    pub(super) node_id: Text,
    pub(super) mesh_id: Text,
    pub(super) peer_node_id: Text,
}

pub(super) struct Snapshot {
    pub(super) payload: Value,
    pub(super) basename: Text,
    pub(super) sha256: String,
}

pub(super) struct Ready {
    pub(super) topology: Topology,
    pub(super) seed: Observer,
    pub(super) worker: Observer,
    pub(super) model: Text,
}

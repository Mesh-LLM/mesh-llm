use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Destination {
    Stdout,
    Stderr,
}

#[derive(Serialize, Deserialize)]
pub struct Record {
    pub stream: Destination,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Behavior {
    Clean,
    Stubborn,
    Nonzero,
    EarlyZero,
    EarlyNonzero,
    OccupiedPort,
    LateReady,
    LateNewline,
    UnterminatedEof,
    DescendantAfterExit,
    Flood,
    StateDeletionFailure,
    HeldCleanup,
    LiveDescendant,
    StubbornDescendant,
}

#[derive(Serialize, Deserialize)]
pub struct Plan {
    pub behavior: Behavior,
    pub records: Vec<Record>,
}

#[derive(Serialize, Deserialize)]
pub struct Audit {
    pub pid: u32,
    pub cwd: PathBuf,
    pub arguments: Vec<String>,
    pub environment: BTreeMap<String, String>,
}

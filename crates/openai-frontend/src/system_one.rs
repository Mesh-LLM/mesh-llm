use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemOneRequest {
    pub state: Value,
    pub model: String,
    pub questions: BTreeMap<String, SystemOneQuestion>,
    #[serde(default)]
    pub images: Option<Vec<Value>>,
    #[serde(default)]
    pub steps: Option<u8>,
    #[serde(default)]
    pub samples: Option<u8>,
    #[serde(default)]
    pub think: Option<u32>,
    #[serde(default)]
    pub sequential: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneQuestion {
    Noul {
        #[serde(default)]
        instructions: Option<Value>,
        #[serde(default)]
        criteria: Option<SystemOneNoulCriteria>,
    },
    Choice {
        #[serde(default)]
        instructions: Option<Value>,
        criteria: BTreeMap<String, Value>,
    },
    Score {
        #[serde(default)]
        instructions: Option<Value>,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SystemOneNoulCriteria {
    #[serde(default)]
    pub r#true: Option<Value>,
    #[serde(default)]
    pub r#false: Option<Value>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, SystemOneAnswer>,
    pub usage: SystemOneUsage,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneAnswer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
    Score {
        score: f32,
        legend: BTreeMap<String, Value>,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct SystemOneUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

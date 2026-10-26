use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::DynResult;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Trajectory {
    pub session_id: String,
    pub source_dataset: String,
    #[serde(skip_serializing, default)]
    pub messages_json: String,
    pub n_turns: u64,
    pub max_isl: u64,
    pub total_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Selection {
    pub sources: Vec<String>,
    pub families: usize,
    pub min_isl: u64,
    pub max_isl_exclusive: u64,
    pub min_turns: u64,
}

impl Selection {
    pub fn validate(&self) -> DynResult<()> {
        if self.families == 0 || self.min_isl == 0 || self.min_turns == 0 {
            return Err("families, minimum ISL and minimum turns must be positive".into());
        }
        if self.max_isl_exclusive <= self.min_isl {
            return Err("maximum ISL must exceed minimum ISL".into());
        }
        if self.sources.is_empty() || self.sources.iter().any(|source| source.trim().is_empty()) {
            return Err("at least one nonempty source dataset is required".into());
        }
        Ok(())
    }

    fn eligible(&self, row: &Trajectory) -> bool {
        self.eligible_fields(&row.source_dataset, row.max_isl, row.n_turns)
    }

    pub fn eligible_fields(&self, source: &str, max_isl: u64, n_turns: u64) -> bool {
        max_isl >= self.min_isl
            && max_isl < self.max_isl_exclusive
            && n_turns >= self.min_turns
            && self.sources.iter().any(|allowed| allowed == source)
    }
}

type SessionOrder = ([u8; 16], String);

// Session rank never changes. A session outside the current smallest K ranks
// cannot re-enter as more rows arrive, so only K complete trajectories are held.
pub struct Selector<'a> {
    selection: &'a Selection,
    rows: BTreeMap<SessionOrder, Trajectory>,
}

impl<'a> Selector<'a> {
    pub fn new(selection: &'a Selection) -> DynResult<Self> {
        selection.validate()?;
        Ok(Self {
            selection,
            rows: BTreeMap::new(),
        })
    }

    pub fn observe(&mut self, row: Trajectory) {
        if !self.selection.eligible(&row) {
            return;
        }
        let key = (
            md5::compute(row.session_id.as_bytes()).0,
            row.session_id.clone(),
        );
        if let Some(prior) = self.rows.get_mut(&key) {
            if preferred(&row, prior) {
                *prior = row;
            }
            return;
        }
        self.rows.insert(key, row);
        if self.rows.len() > self.selection.families {
            self.rows.pop_last();
        }
    }

    pub fn finish(self) -> DynResult<Vec<Trajectory>> {
        if self.rows.len() != self.selection.families {
            return Err(format!(
                "selected {} trajectories, expected {}",
                self.rows.len(),
                self.selection.families
            )
            .into());
        }
        Ok(self.rows.into_values().collect())
    }
}

fn preferred(candidate: &Trajectory, prior: &Trajectory) -> bool {
    let rank = |row: &Trajectory| {
        (
            std::cmp::Reverse(row.max_isl),
            std::cmp::Reverse(row.total_tokens),
            md5::compute(row.messages_json.as_bytes()).0,
        )
    };
    rank(candidate) < rank(prior)
        || (rank(candidate) == rank(prior)
            && (
                &candidate.source_dataset,
                candidate.n_turns,
                &candidate.messages_json,
            ) < (&prior.source_dataset, prior.n_turns, &prior.messages_json))
}

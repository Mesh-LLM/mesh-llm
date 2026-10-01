mod cli;
mod metadata;
mod names;
mod successors;

#[cfg(test)]
mod tests;

pub(crate) use cli::run;
pub(crate) use names::{PackageList, PackageName};
use serde::Deserialize;
use std::collections::BTreeSet;
use successors::SUCCESSORS;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("package list must be a nonempty array")]
    EmptyPackages,
    #[error("invalid Cargo package name")]
    InvalidName,
    #[error("duplicate Cargo package name")]
    DuplicatePackage,
    #[error("batch matrix must be a nonempty array")]
    EmptyBatches,
    #[error("requested batch contains packages outside the protected plan")]
    OutsidePlan,
    #[error("unsupported planner package generation")]
    Generation,
    #[error("planned package {package:?} has missing source owners: {missing:?}")]
    MissingOwners {
        package: PackageName,
        missing: Vec<PackageName>,
    },
    #[error("package resolves more than once: {0:?}")]
    DuplicateTranslation(PackageName),
    #[error("invalid package JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Generation {
    Legacy,
    Current,
}

impl Generation {
    pub(crate) fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "legacy" => Ok(Self::Legacy),
            "current" => Ok(Self::Current),
            _ => Err(Error::Generation),
        }
    }
}

#[derive(Deserialize)]
struct Batch {
    crates: PackageList,
}

pub(crate) struct TranslationRequest {
    requested: PackageList,
    generation: Generation,
}

impl TranslationRequest {
    pub(crate) fn parse(
        crates: &str,
        batches: Option<&str>,
        generation: Generation,
    ) -> Result<Self, Error> {
        let requested = PackageList::parse(crates)?;
        let planned = match batches {
            None => requested.clone(),
            Some(json) => {
                let batches: Vec<Batch> = serde_json::from_str(json)?;
                if batches.is_empty() {
                    return Err(Error::EmptyBatches);
                }
                PackageList::try_from(
                    batches
                        .into_iter()
                        .flat_map(|batch| batch.crates.names().to_vec())
                        .collect::<Vec<_>>(),
                )?
            }
        };
        if requested
            .names()
            .iter()
            .any(|name| !planned.names().contains(name))
        {
            return Err(Error::OutsidePlan);
        }
        Ok(Self {
            requested,
            generation,
        })
    }

    pub(crate) fn resolve(&self, available: &BTreeSet<PackageName>) -> Result<PackageList, Error> {
        let migrating = match self.generation {
            Generation::Legacy => available
                .iter()
                .any(|name| name.as_str() == "skippy-package-builder"),
            Generation::Current => false,
        };
        let mut result = Vec::new();
        for name in self.requested.names() {
            let successors = migrating
                .then(|| SUCCESSORS.iter().find(|(old, _)| *old == name.as_str()))
                .flatten();
            let candidates = match successors {
                Some((_, owners)) => {
                    let candidates = owners
                        .iter()
                        .map(|owner| PackageName::try_from((*owner).to_owned()))
                        .collect::<Result<Vec<_>, _>>()?;
                    let missing = candidates
                        .iter()
                        .filter(|owner| !available.contains(*owner))
                        .cloned()
                        .collect::<BTreeSet<_>>();
                    if !missing.is_empty() {
                        return Err(Error::MissingOwners {
                            package: name.clone(),
                            missing: missing.into_iter().collect(),
                        });
                    }
                    candidates
                }
                None => vec![name.clone()],
            };
            for candidate in candidates {
                if result.contains(&candidate) {
                    return Err(Error::DuplicateTranslation(candidate));
                }
                result.push(candidate);
            }
        }
        PackageList::try_from(result)
    }
}

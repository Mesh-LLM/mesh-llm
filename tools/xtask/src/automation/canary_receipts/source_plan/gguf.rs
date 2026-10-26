use crate::automation::replay_matrix::model_preflight::dimensions::{
    self, Dimensions, DimensionsError,
};
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub(super) enum AdmissionError {
    #[error("GGUF shard {path} metadata is invalid: {source}")]
    Metadata {
        path: PathBuf,
        source: DimensionsError,
    },
    #[error("artifact has no metadata-bearing GGUF shard")]
    MissingDimensions,
    #[error("target shard {path} disagrees with planned architecture, layers, width, or MTP count")]
    Mismatch { path: PathBuf },
}

pub(super) enum Expected<'a> {
    Target(&'a Dimensions),
    Draft,
}

pub(super) fn verify(paths: &[PathBuf], expected: Expected<'_>) -> Result<(), AdmissionError> {
    let mut found = false;
    for path in paths {
        if let Some(actual) =
            dimensions::inspect(path).map_err(|source| AdmissionError::Metadata {
                path: path.clone(),
                source,
            })?
        {
            found = true;
            match expected {
                Expected::Target(planned) if planned != &actual => {
                    return Err(AdmissionError::Mismatch { path: path.clone() });
                }
                Expected::Target(_) | Expected::Draft => {}
            }
        }
    }
    if found {
        Ok(())
    } else {
        Err(AdmissionError::MissingDimensions)
    }
}

//! Provider-neutral CI control: planning, lane result admission, the
//! repository checks hosted planners and summaries run, and automation
//! bootstrap identity.

pub mod bootstrap;
pub mod child_output;
pub mod lane_results;
pub mod plan;
pub mod repository;
pub mod temp_dir;
pub mod prepared_input {
    pub mod value_format;
}

pub type DynError = Box<dyn std::error::Error>;
pub type DynResult<T> = Result<T, DynError>;

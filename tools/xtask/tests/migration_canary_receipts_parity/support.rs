#[path = "fixture_files.rs"]
mod fixture_files;
#[path = "package.rs"]
mod package;
#[path = "package_admission.rs"]
mod package_admission;

pub(crate) use package::CaseFixture;

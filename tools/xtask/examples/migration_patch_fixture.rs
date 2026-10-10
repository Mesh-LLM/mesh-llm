#[path = "../tests/migration_patch_bytes/fixture.rs"]
mod fixture;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fixture::run()
}

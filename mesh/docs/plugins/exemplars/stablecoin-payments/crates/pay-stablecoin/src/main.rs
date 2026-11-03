use anyhow::Result;
#[tokio::main]
async fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.iter().any(|s| s == "--manifest") {
        use std::io::Write;
        writeln!(
            std::io::stdout(),
            "{}",
            mesh_llm_plugin::package_manifest_json(&mesh_llm_pay_stablecoin::runtime::manifest())?
        )?;
        return Ok(());
    }
    mesh_llm_pay_stablecoin::runtime::run().await
}

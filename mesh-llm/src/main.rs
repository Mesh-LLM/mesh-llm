#![recursion_limit = "256"]

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    mesh_llm::run().await
}
